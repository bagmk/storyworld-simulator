"""tools/inline_optimizer.py — Per-episode inline distill+prose+polish optimizer.

Replaces step_chapter_gen in !novel-daily with a 5-trial mini-Optuna loop.
Phase 1 (trials 0,1): parallel exploratory
Phase 2 (trials 2,3,4): parallel TPE-guided

Logs results to data/policy_score_log.jsonl.
Every 5 completed episodes triggers a background full Optuna re-tune.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("inline_optimizer")

REVIEWER_MODEL = "gpt-4o-mini"
POLICY_SCORE_LOG = _REPO_ROOT / "data" / "policy_score_log.jsonl"

# ── Quality review → LLM scorer weight mapping ───────────────────────────────

# Maps quality review dimension keys → which LLM scoring criteria they relate to
_QUALITY_TO_LLM: dict[str, list[str]] = {
    "thrill":         ["emotional_tension", "pacing_tension_balance"],
    "style":          ["literary_quality", "readability", "sentence_diversity"],
    "causality":      ["pacing_tension_balance"],
    "character":      ["dialogue_effectiveness"],
    "scene_function": ["paragraph_rhythm", "prose_vividness"],
}

_ALL_LLM_CRITERIA = [
    "literary_quality", "emotional_tension", "readability", "sentence_diversity",
    "paragraph_rhythm", "prose_vividness", "dialogue_effectiveness", "pacing_tension_balance",
]


def _build_criterion_weights(quality_focus: dict | None) -> dict[str, float]:
    """Return per-LLM-criterion weight multipliers.
    Criteria mapped to low-scoring quality dimensions get higher weight.
    Default=1.0; boost = 1.0 + (7-score)/7 for score < 7."""
    weights = {k: 1.0 for k in _ALL_LLM_CRITERIA}
    if not quality_focus:
        return weights
    for q_dim, llm_keys in _QUALITY_TO_LLM.items():
        score = float(quality_focus.get(q_dim, 10.0))
        if score < 7.0:
            boost = 1.0 + (7.0 - score) / 7.0  # range: 1.0 (score=7) → 2.0 (score=0)
            for k in llm_keys:
                weights[k] = max(weights.get(k, 1.0), boost)
    return weights


# ── LLM scorer constants (reused from optuna_prose_test.py) ──────────────────

_LLM_SCORER_SYSTEM = (
    "You are a strict literary editor for serialized Korean thriller novels. "
    "Evaluate prose STYLE and STRUCTURE only — not story content. "
    "Focus on how the text is written, not what happens. "
    "Return ONLY valid JSON, no markdown."
)

_LLM_SCORER_CRITERIA = """Rate each criterion from 0 to 10 based on the prose excerpt below.
Evaluate STYLE/STRUCTURE, not content:

  literary_quality      : prose richness, word choice, imagery depth
  emotional_tension     : pacing feel, suspense rhythm, interiority weight
  readability           : clarity, sentence flow, no awkward phrasing
  sentence_diversity    : variety in sentence LENGTH and STRUCTURE (short/long mix, not all same pattern)
  paragraph_rhythm      : variation in paragraph length; does it breathe? or does it feel monotonous?
  prose_vividness       : concrete sensory details vs abstract statements (higher = more concrete)
  dialogue_effectiveness: does dialogue feel natural and serve the scene? (0 if no dialogue)
  pacing_tension_balance: tension/release balance — not too flat, not constantly peak

Return ONLY this JSON (no text before or after):
{
  "literary_quality": X,
  "emotional_tension": X,
  "readability": X,
  "sentence_diversity": X,
  "paragraph_rhythm": X,
  "prose_vividness": X,
  "dialogue_effectiveness": X,
  "pacing_tension_balance": X
}
"""


# ── Parameter space ───────────────────────────────────────────────────────────

def _param_space(trial, base_policy: dict) -> dict:
    return {
        "distiller_temperature":         trial.suggest_float("distiller_temperature", 0.05, 0.45),
        "target_scenes":                 trial.suggest_int("target_scenes", 2, 6),
        "dialogue_compaction_strength":  trial.suggest_float("dialogue_compaction_strength", 0.5, 1.0),
        "prose_scene_temperature":       trial.suggest_float("prose_scene_temperature", 0.55, 0.85),
        "prose_paragraph_min_sentences": trial.suggest_int("prose_paragraph_min_sentences", 2, 4),
        "prose_paragraph_max_sentences": trial.suggest_int("prose_paragraph_max_sentences", 3, 5),
        "prose_transition_temperature":  trial.suggest_float("prose_transition_temperature", 0.3, 0.7),
        "prose_polish_temperature":      trial.suggest_float("prose_polish_temperature", 0.2, 0.6),
        "hold_pressure_peak":            trial.suggest_categorical("hold_pressure_peak", [0, 1]),
        "scene_closure_aggressiveness":  trial.suggest_float("scene_closure_aggressiveness", 0.05, 0.5),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_deterministic(chapter_text: str, episode_id: str) -> float:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from quality_analyzer import QualityAnalyzer
    qa = QualityAnalyzer(chapter_text, episode_name=episode_id)
    return round(qa.analyze().get("overall_score", 0.0) * 10, 3)


def _score_llm(chapter_text: str, criterion_weights: dict[str, float] | None = None) -> float:
    from openai import OpenAI
    client = OpenAI()

    text = chapter_text.strip()
    if len(text) > 3000:
        excerpt = text[:2000] + "\n\n...[중략]...\n\n" + text[-1000:]
    else:
        excerpt = text

    prompt = _LLM_SCORER_CRITERIA + f"\n--- EXCERPT ---\n{excerpt}\n--- END ---"

    try:
        resp = client.chat.completions.create(
            model=REVIEWER_MODEL,
            messages=[
                {"role": "system", "content": _LLM_SCORER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        scores = json.loads(raw)
        numeric = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        if not numeric:
            return 0.0
        if criterion_weights:
            total_w = sum(criterion_weights.get(k, 1.0) for k in numeric)
            return round(sum(v * criterion_weights.get(k, 1.0) for k, v in numeric.items()) / total_w, 3)
        return round(sum(numeric.values()) / len(numeric), 3)
    except Exception as exc:
        logger.warning("LLM scorer error: %s", exc)
        return 0.0


def _score_chapter(
    chapter_text: str, episode_id: str, quality_focus: dict | None = None
) -> tuple[float, float, float]:
    det = _score_deterministic(chapter_text, episode_id)
    weights = _build_criterion_weights(quality_focus)
    llm = _score_llm(chapter_text, criterion_weights=weights)
    combined = round(0.4 * det + 0.6 * llm, 3)
    logger.info("[SCORE] det=%.2f llm=%.2f combined=%.3f focus=%s",
                det, llm, combined, list(quality_focus.keys()) if quality_focus else None)
    return combined, det, llm


# ── Single trial runner ───────────────────────────────────────────────────────

def _sync_run_trial(
    trial_idx: int,
    trial_params: dict,
    episode_id: str,
    episode_config: dict,
    trial_dir: Path,
    llm,
    protagonist_id: str,
    protagonist_name: str,
    target_words: int,
    character_profiles,
    reader_feedback: dict | None,
    guardian_briefing: str | None,
    base_policy: dict,
    quality_focus: dict | None = None,
) -> tuple[float, Path, dict]:
    """Synchronous trial: distill → prose. Returns (score, chapter_path, meta)."""
    from src.novel_writer.scene_distiller import SceneDistiller
    from src.novel_writer.prose_generator import ProseGenerator

    trial_policy = dict(base_policy)
    trial_policy.update(trial_params)

    try:
        distiller = SceneDistiller(
            llm=llm,
            episode_config=episode_config,
            runtime_policy=trial_policy,
            reader_feedback=reader_feedback,
        )
        scenes = distiller.distill(
            episode_id=episode_id,
            protagonist_id=protagonist_id,
            target_scenes=trial_params["target_scenes"],
        )
        scenes = distiller.normalize_scene_timeline(distiller.apply_scene_guards(scenes))
        if not scenes:
            logger.warning("Trial %d: distill returned no scenes", trial_idx)
            return 0.0, trial_dir / "empty.md", {
                "det": 0.0,
                "llm": 0.0,
                "scene_count": 0,
                "raw_turn_total": 0,
                "target_scenes": int(trial_params.get("target_scenes", 0) or 0),
            }
    except Exception as exc:
        logger.warning("Trial %d: distill failed: %s", trial_idx, exc)
        return 0.0, trial_dir / "empty.md", {
            "det": 0.0,
            "llm": 0.0,
            "scene_count": 0,
            "raw_turn_total": 0,
            "target_scenes": int(trial_params.get("target_scenes", 0) or 0),
        }

    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        prose_gen = ProseGenerator(
            llm=llm,
            episode_config=episode_config,
            output_dir=str(trial_dir),
            character_profiles=character_profiles,
            max_history_episodes=int(trial_policy.get("prose_history_max_episodes", 12)),
            runtime_policy=trial_policy,
            reader_feedback=reader_feedback,
            guardian_briefing=guardian_briefing,
        )
        chapter_path = prose_gen.generate_chapter(
            scenes=scenes,
            protagonist_name=protagonist_name,
            style="third_person_close",
            target_words=target_words,
        )
    except Exception as exc:
        logger.warning("Trial %d: prose generation failed: %s", trial_idx, exc)
        return 0.0, trial_dir / "empty.md", {
            "det": 0.0,
            "llm": 0.0,
            "scene_count": len(scenes),
            "raw_turn_total": int(sum(max(0, int(s.raw_turn_count or 0)) for s in scenes)),
            "target_scenes": int(trial_params.get("target_scenes", 0) or 0),
        }

    det = 0.0
    llm = 0.0
    word_count = 0
    try:
        chapter_text = Path(chapter_path).read_text(encoding="utf-8")
        word_count = len(chapter_text.split())
        score, det, llm = _score_chapter(chapter_text, episode_id, quality_focus=quality_focus)
    except Exception as exc:
        logger.warning("Trial %d: scoring failed: %s", trial_idx, exc)
        score = 0.0

    logger.info("Trial %d | score=%.3f | path=%s", trial_idx, score, chapter_path)
    return score, Path(chapter_path), {
        "det": round(det, 3),
        "llm": round(llm, 3),
        "scene_count": len(scenes),
        "raw_turn_total": int(sum(max(0, int(s.raw_turn_count or 0)) for s in scenes)),
        "target_scenes": int(trial_params.get("target_scenes", 0) or 0),
        "chapter_file": Path(chapter_path).name,
        "word_count": int(word_count),
    }


async def _run_single_trial(
    trial_idx: int,
    trial_params: dict,
    episode_id: str,
    episode_config: dict,
    trial_dir: Path,
    llm,
    protagonist_id: str,
    protagonist_name: str,
    target_words: int,
    character_profiles,
    reader_feedback: dict | None,
    guardian_briefing: str | None,
    base_policy: dict,
    quality_focus: dict | None = None,
) -> tuple[float, Path, dict]:
    """Async wrapper around _sync_run_trial using asyncio.to_thread."""
    return await asyncio.to_thread(
        _sync_run_trial,
        trial_idx,
        trial_params,
        episode_id,
        episode_config,
        trial_dir,
        llm,
        protagonist_id,
        protagonist_name,
        target_words,
        character_profiles,
        reader_feedback,
        guardian_briefing,
        base_policy,
        quality_focus,
    )


# ── Main optimizer ────────────────────────────────────────────────────────────

async def run_inline_optimize(
    episode_id: str,
    episode_config: dict,
    run_dir: Path,
    protagonist_id: str,
    protagonist_name: str,
    target_words: int,
    budget: float,
    character_profiles,
    reader_feedback: dict | None,
    guardian_briefing: str | None,
    base_policy: dict | None,
    base_model: str,
    premium_model: str,
    notify_fn=None,
    quality_focus: dict | None = None,
    final_upgrade_model: str | None = None,
) -> tuple[Path, dict, float, list[float]]:
    """
    Run 5-trial mini-Optuna loop (Phase 1: 2 parallel, Phase 2: 3 parallel).

    Returns (best_chapter_path, best_params, best_score, all_trial_scores).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise RuntimeError("optuna not installed — run: pip install optuna")

    from src.novel_writer.llm_client import LLMClient

    opt_dir = run_dir / "inline_opt"
    opt_dir.mkdir(parents=True, exist_ok=True)
    trial_budget = budget / 5

    study = optuna.create_study(
        study_name=f"inline_{episode_id}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=2, seed=42),
    )

    trial_scores: list[float] = []
    trial_paths: list[Path] = []
    trial_meta_by_idx: dict[int, dict] = {}

    # ── Phase 1: trials 0,1 (exploratory) ───────────────────────────────────
    if notify_fn:
        await notify_fn("[OPTIMIZE] 🔬 Trial 0,1 시작... (Phase 1/2)")
        await notify_fn("[OPTIMIZE] 📝 챕터 생성 시작: trial 0, trial 1")

    p1_trials = [study.ask() for _ in range(2)]
    effective_base_policy = dict(base_policy or {})
    p1_params = [_param_space(t, effective_base_policy) for t in p1_trials]

    p1_llms = [
        LLMClient(model=base_model, premium_model=premium_model, budget_usd=trial_budget)
        for _ in range(2)
    ]

    p1_results = await asyncio.gather(
        *[
            _run_single_trial(
                trial_idx=i,
                trial_params=p1_params[i],
                episode_id=episode_id,
                episode_config=episode_config,
                trial_dir=opt_dir / f"trial_{i}",
                llm=p1_llms[i],
                protagonist_id=protagonist_id,
                protagonist_name=protagonist_name,
                target_words=target_words,
                character_profiles=character_profiles,
                reader_feedback=reader_feedback,
                guardian_briefing=guardian_briefing,
                base_policy=effective_base_policy,
                quality_focus=quality_focus,
            )
            for i in range(2)
        ],
        return_exceptions=False,
    )

    for i, (score, path, meta) in enumerate(p1_results):
        study.tell(p1_trials[i], score)
        trial_scores.append(score)
        trial_paths.append(path)
        trial_meta_by_idx[i] = meta
        logger.info("Phase 1 trial %d → score=%.3f", i, score)
        if notify_fn:
            await notify_fn(
                f"[OPTIMIZE] 📝 trial {i} 챕터 생성 완료 | `{meta.get('chapter_file', path.name)}` "
                f"({int(meta.get('word_count', 0))}단어) | score {score:.2f}"
            )
    if notify_fn:
        p1_lines = []
        for i, score in enumerate(trial_scores[:2]):
            meta = trial_meta_by_idx.get(i, {})
            p1_lines.append(
                f"- trial {i}: {score:.2f} (det {float(meta.get('det', 0.0)):.2f} / llm {float(meta.get('llm', 0.0)):.2f}, "
                f"scenes {int(meta.get('scene_count', 0))}/{int(meta.get('target_scenes', 0))}, turns {int(meta.get('raw_turn_total', 0))})"
            )
        await notify_fn("[OPTIMIZE] 📊 Phase 1 결과\n" + "\n".join(p1_lines))

    # ── Phase 2: trials 2,3,4 (TPE-guided) ──────────────────────────────────
    if notify_fn:
        await notify_fn("[OPTIMIZE] 🔬 Trial 2,3,4 시작... (Phase 2/2)")
        await notify_fn("[OPTIMIZE] 📝 챕터 생성 시작: trial 2, trial 3, trial 4")

    p2_trials = [study.ask() for _ in range(3)]
    p2_params = [_param_space(t, effective_base_policy) for t in p2_trials]

    p2_llms = [
        LLMClient(model=base_model, premium_model=premium_model, budget_usd=trial_budget)
        for _ in range(3)
    ]

    p2_results = await asyncio.gather(
        *[
            _run_single_trial(
                trial_idx=2 + i,
                trial_params=p2_params[i],
                episode_id=episode_id,
                episode_config=episode_config,
                trial_dir=opt_dir / f"trial_{2 + i}",
                llm=p2_llms[i],
                protagonist_id=protagonist_id,
                protagonist_name=protagonist_name,
                target_words=target_words,
                character_profiles=character_profiles,
                reader_feedback=reader_feedback,
                guardian_briefing=guardian_briefing,
                base_policy=effective_base_policy,
                quality_focus=quality_focus,
            )
            for i in range(3)
        ],
        return_exceptions=False,
    )

    for i, (score, path, meta) in enumerate(p2_results):
        study.tell(p2_trials[i], score)
        trial_scores.append(score)
        trial_paths.append(path)
        trial_meta_by_idx[2 + i] = meta
        logger.info("Phase 2 trial %d → score=%.3f", 2 + i, score)
        if notify_fn:
            trial_idx = 2 + i
            await notify_fn(
                f"[OPTIMIZE] 📝 trial {trial_idx} 챕터 생성 완료 | `{meta.get('chapter_file', path.name)}` "
                f"({int(meta.get('word_count', 0))}단어) | score {score:.2f}"
            )
    if notify_fn:
        p2_lines = []
        for i, score in enumerate(trial_scores[2:], start=2):
            meta = trial_meta_by_idx.get(i, {})
            p2_lines.append(
                f"- trial {i}: {score:.2f} (det {float(meta.get('det', 0.0)):.2f} / llm {float(meta.get('llm', 0.0)):.2f}, "
                f"scenes {int(meta.get('scene_count', 0))}/{int(meta.get('target_scenes', 0))}, turns {int(meta.get('raw_turn_total', 0))})"
            )
        await notify_fn("[OPTIMIZE] 📊 Phase 2 결과\n" + "\n".join(p2_lines))

    # ── Pick best ────────────────────────────────────────────────────────────
    best_idx = int(max(range(len(trial_scores)), key=lambda i: trial_scores[i]))
    best_score = trial_scores[best_idx]
    best_params = (p1_params + p2_params)[best_idx]
    best_src = trial_paths[best_idx]

    if notify_fn:
        sorted_scores = sorted(enumerate(trial_scores), key=lambda item: item[1], reverse=True)
        ranking = " | ".join(f"t{idx}:{score:.2f}" for idx, score in sorted_scores)
        best_meta = trial_meta_by_idx.get(best_idx, {})
        await notify_fn(
            f"[OPTIMIZE] ✅ 5트라이얼 완료 | best={best_score:.2f} (trial {best_idx})\n"
            f"순위: {ranking}\n"
            f"best detail: det {float(best_meta.get('det', 0.0)):.2f} / llm {float(best_meta.get('llm', 0.0)):.2f}, "
            f"scenes {int(best_meta.get('scene_count', 0))}/{int(best_meta.get('target_scenes', 0))}, turns {int(best_meta.get('raw_turn_total', 0))}"
        )

    # ── Final one-pass upgrade generation (optional) ────────────────────────
    if final_upgrade_model:
        try:
            if notify_fn:
                await notify_fn(
                    f"[OPTIMIZE] ✨ 최종 1회 업그레이드 생성 시작 (model {final_upgrade_model})"
                )
            final_llm = LLMClient(
                model=base_model,
                premium_model=final_upgrade_model,
                budget_usd=trial_budget,
            )
            final_score, final_path, final_meta = await _run_single_trial(
                trial_idx=best_idx,
                trial_params=best_params,
                episode_id=episode_id,
                episode_config=episode_config,
                trial_dir=opt_dir / "final_upgrade",
                llm=final_llm,
                protagonist_id=protagonist_id,
                protagonist_name=protagonist_name,
                target_words=target_words,
                character_profiles=character_profiles,
                reader_feedback=reader_feedback,
                guardian_briefing=guardian_briefing,
                base_policy=effective_base_policy,
                quality_focus=quality_focus,
            )
            if notify_fn:
                await notify_fn(
                    f"[OPTIMIZE] ✨ 최종 1회 업그레이드 완료 | score {final_score:.2f} | "
                    f"`{final_meta.get('chapter_file', final_path.name)}` "
                    f"({int(final_meta.get('word_count', 0))}단어)"
                )
            if final_path.exists() and final_path.stat().st_size > 0 and final_score >= best_score:
                best_src = final_path
                best_score = final_score
                if notify_fn:
                    await notify_fn("[OPTIMIZE] ✨ 업그레이드 결과 채택 (점수 유지/상승)")
            elif notify_fn:
                await notify_fn("[OPTIMIZE] ℹ️ 업그레이드 결과 미채택 (기존 best 유지)")
        except Exception as exc:
            logger.warning("Final upgrade generation failed: %s", exc)
            if notify_fn:
                await notify_fn(f"[OPTIMIZE] ⚠️ 최종 업그레이드 생성 실패: {exc}")

    # Copy best chapter to run_dir
    best_chapter_path = run_dir / f"{episode_id}_chapter.md"
    if best_src.exists() and best_src.stat().st_size > 0:
        shutil.copy2(best_src, best_chapter_path)
    else:
        best_chapter_path = best_src

    logger.info(
        "Inline optimize done | best_trial=%d score=%.3f path=%s",
        best_idx, best_score, best_chapter_path,
    )
    return best_chapter_path, best_params, best_score, trial_scores


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_policy_score(
    episode_id: str,
    best_params: dict,
    best_score: float,
    all_trial_scores: list[float],
    log_path: Path | None = None,
    quality_review_scores: dict | None = None,
) -> None:
    log_path = log_path or POLICY_SCORE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date": str(date.today()),
        "episode_id": episode_id,
        "best_score": round(best_score, 4),
        "best_params": best_params,
        "trial_scores": [round(s, 4) for s in all_trial_scores],
    }
    if quality_review_scores:
        record["quality_review_scores"] = quality_review_scores
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Logged policy score for %s → %s", episode_id, log_path)


def update_policy_log_quality_scores(
    quality_scores: dict,
    log_path: Path | None = None,
) -> None:
    """Update the most recent policy_score_log entry with quality review scores."""
    log_path = log_path or POLICY_SCORE_LOG
    if not log_path.exists():
        return
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return
    try:
        last = json.loads(lines[-1])
        last["quality_review_scores"] = quality_scores
        lines[-1] = json.dumps(last, ensure_ascii=False)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Updated last policy log entry with quality_review_scores")
    except Exception as exc:
        logger.warning("Failed to update quality scores in policy log: %s", exc)


def should_trigger_full_optuna(log_path: Path | None = None) -> bool:
    log_path = log_path or POLICY_SCORE_LOG
    if not log_path.exists():
        return False
    count = sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return count > 0 and count % 5 == 0


def trigger_full_optuna_background(repo_root: Path, trials: int = 30) -> None:
    today = date.today().isoformat()
    log_file = repo_root / "output" / f"optuna_auto_retune_{today}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "tools/optuna_multi_study.py",
        "--trials", str(trials),
        "--study", "distiller", "orchestrator", "polisher",
    ]
    warmup_log = POLICY_SCORE_LOG
    if warmup_log.exists():
        cmd += ["--warmup-log", str(warmup_log)]

    with log_file.open("w") as lf:
        subprocess.Popen(
            ["nohup"] + cmd,
            stdout=lf,
            stderr=lf,
            cwd=str(repo_root),
            start_new_session=True,
        )
    logger.info("Full Optuna re-tune launched in background → %s", log_file)


def update_rl_policy(best_params: dict, best_score: float, episode_id: str) -> None:
    """Write inline optimizer best params back to rl_policy.json."""
    import json
    from datetime import date
    policy_path = _REPO_ROOT / "data" / "rl_policy.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
    except Exception:
        policy = {}

    policy["version"] = int(policy.get("version", 2)) + 1
    # Update only the params that inline optimizer controls
    INLINE_PARAM_KEYS = {
        "distiller_temperature", "target_scenes", "dialogue_compaction_strength",
        "prose_scene_temperature", "prose_paragraph_min_sentences", "prose_paragraph_max_sentences",
        "prose_transition_temperature", "prose_polish_temperature",
        "hold_pressure_peak", "scene_closure_aggressiveness",
    }
    for k, v in best_params.items():
        if k in INLINE_PARAM_KEYS:
            policy[k] = v
    policy["_last_inline_opt"] = {
        "date": str(date.today()),
        "episode_id": episode_id,
        "score": round(best_score, 4),
    }
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("rl_policy.json updated → version %s score=%.3f", policy["version"], best_score)


# ── Cycle score logging ────────────────────────────────────────────────────────

CYCLE_SCORE_LOG = _REPO_ROOT / "data" / "cycle_score_log.jsonl"


def log_cycle_score(
    episode_id: str,
    cycle_idx: int,
    current_params: dict,
    ai_review_scores: dict,
    subtrial_data: list[dict],
    log_path: Path | None = None,
) -> None:
    """Log one AUTO cycle's data: params, AI review scores, and subtrial results.

    ai_review_scores: {thrill, style, causality, character, scene_fn, avg}
    subtrial_data:    [{trial_idx, score, det, llm, params}, ...]
    """
    log_path = log_path or CYCLE_SCORE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date": str(date.today()),
        "episode_id": episode_id,
        "cycle_idx": cycle_idx,
        "ai_review": ai_review_scores,
        "cycle_params": current_params,
        "subtrials": subtrial_data,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Logged cycle score for %s cycle %d → %s", episode_id, cycle_idx, log_path)


# ── Narrow param space for per-cycle mini re-optimize ─────────────────────────

# Full search ranges mirroring _param_space()
_PARAM_FULL_RANGES: dict[str, tuple] = {
    "distiller_temperature":         ("float", 0.05, 0.45),
    "target_scenes":                 ("int",   2,    6),
    "dialogue_compaction_strength":  ("float", 0.5,  1.0),
    "prose_scene_temperature":       ("float", 0.55, 0.85),
    "prose_paragraph_min_sentences": ("int",   2,    4),
    "prose_paragraph_max_sentences": ("int",   3,    5),
    "prose_transition_temperature":  ("float", 0.3,  0.7),
    "prose_polish_temperature":      ("float", 0.2,  0.6),
    "hold_pressure_peak":            ("cat",   [0, 1]),
    "scene_closure_aggressiveness":  ("float", 0.05, 0.5),
}


def _narrow_width_from_n_trials(n_past_trials: int) -> float:
    """Compute search width_ratio based on accumulated trial count.

    Starts at 0.30 (wide) and shrinks toward 0.12 as evidence accumulates.
    Schedule: shrinks 15% every 10 trials, floor at 0.12.

    Examples:
      0  trials → 0.30 (full narrow band, TPE barely started)
     10  trials → 0.255
     20  trials → 0.217
     30  trials → 0.184
     50  trials → 0.133
    100+ trials → ~0.12 (floor)
    """
    return max(0.12, 0.30 * (0.85 ** (n_past_trials // 10)))


def _param_space_narrow(trial, base_policy: dict, width_ratio: float = 0.30) -> dict:
    """Sample params in a narrow band around base_policy values.

    width_ratio controls band size: 0.30 → ±15% of total range per side.
    Falls back to full range when current value is not in base_policy.
    Use _narrow_width_from_n_trials() to get a dynamic width_ratio.
    """
    params: dict = {}
    for name, spec in _PARAM_FULL_RANGES.items():
        kind = spec[0]
        current = base_policy.get(name)
        if kind == "float":
            lo, hi = float(spec[1]), float(spec[2])
            if current is not None:
                # Clamp current to valid range before computing narrow band
                cur = max(lo, min(hi, float(current)))
                half = (hi - lo) * width_ratio / 2.0
                c_lo = max(lo, cur - half)
                c_hi = min(hi, cur + half)
                if c_lo >= c_hi:           # degenerate range → center on valid midpoint
                    c_lo = max(lo, cur - (hi - lo) * 0.1)
                    c_hi = min(hi, cur + (hi - lo) * 0.1)
                    if c_lo >= c_hi:       # still degenerate → full range
                        c_lo, c_hi = lo, hi
            else:
                c_lo, c_hi = lo, hi
            params[name] = trial.suggest_float(name, c_lo, c_hi)
        elif kind == "int":
            lo, hi = int(spec[1]), int(spec[2])
            if current is not None:
                c_lo = max(lo, int(current) - 1)
                c_hi = min(hi, int(current) + 1)
            else:
                c_lo, c_hi = lo, hi
            params[name] = trial.suggest_int(name, c_lo, c_hi)
        elif kind == "cat":
            choices: list = list(spec[1])
            if current is not None and current in choices:
                # Pin to current value — no exploration for categoricals in narrow mode
                params[name] = trial.suggest_categorical(name, [current])
            else:
                params[name] = trial.suggest_categorical(name, choices)
    return params


# ── Per-cycle mini re-optimizer ───────────────────────────────────────────────

async def run_mini_reoptimize(
    episode_id: str,
    episode_config: dict,
    run_dir: Path,
    protagonist_id: str,
    protagonist_name: str,
    target_words: int,
    budget: float,
    character_profiles,
    reader_feedback: dict | None,
    guardian_briefing: str | None,
    current_params: dict,
    base_model: str,
    premium_model: str,
    notify_fn=None,
    quality_focus: dict | None = None,
    cycle_idx: int = 0,
    n_trials: int = 5,
    group_size: int = 5,
) -> tuple[Path | None, dict, float, list[dict]]:
    """Narrow-range param re-optimize using a persistent SQLite Optuna study.

    Runs n_trials total, in parallel groups of group_size.
    e.g. n_trials=25, group_size=5 → 5 groups of 5 parallel chapter generations.

    TPE learns across groups within a call AND across outer cycles (SQLite persistence).
    Search width narrows automatically as trial count accumulates.

    Returns:
        (best_chapter_path | None, best_params, best_score, subtrial_data_list)
        subtrial_data_list: [{trial_idx, score, det, llm, params}, ...]
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed — mini-reoptimize skipped")
        return None, current_params, 0.0, []

    from src.novel_writer.llm_client import LLMClient

    opt_dir = run_dir / f"mini_opt_cycle{cycle_idx}"
    opt_dir.mkdir(parents=True, exist_ok=True)
    trial_budget = budget / max(n_trials, 1)

    # ── Persistent SQLite study — accumulates across ALL outer cycles ─────────
    db_path = _REPO_ROOT / "data" / f"optuna_mini_{episode_id}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"mini_{episode_id}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5, seed=42),
    )
    n_past = len(study.trials)
    logger.info(
        "[MINI-OPT] episode=%s cycle=%d n_trials=%d group_size=%d past_trials=%d",
        episode_id, cycle_idx, n_trials, group_size, n_past,
    )

    trial_scores: list[float] = []
    trial_paths: list[Path] = []
    trial_meta_by_idx: dict[int, dict] = {}
    all_trial_params: list[dict] = []
    global_trial_idx = 0

    # ── Run trials in parallel groups, TPE learns between groups ─────────────
    n_groups = (n_trials + group_size - 1) // group_size
    for group_i in range(n_groups):
        group_n = min(group_size, n_trials - global_trial_idx)
        # Width narrows as evidence accumulates (past + already-done-this-call)
        width = _narrow_width_from_n_trials(n_past + global_trial_idx)

        if notify_fn:
            await notify_fn(
                f"[MINI-OPT] outer {cycle_idx} — group {group_i + 1}/{n_groups} "
                f"({group_n} trials parallel, 탐색 폭 ±{width * 50:.0f}%)"
            )

        g_optuna_trials = [study.ask() for _ in range(group_n)]
        g_params = [
            _param_space_narrow(t, current_params, width_ratio=width)
            for t in g_optuna_trials
        ]
        g_llms = [
            LLMClient(model=base_model, premium_model=premium_model, budget_usd=trial_budget)
            for _ in range(group_n)
        ]

        g_results = await asyncio.gather(
            *[
                _run_single_trial(
                    trial_idx=global_trial_idx + i,
                    trial_params=g_params[i],
                    episode_id=episode_id,
                    episode_config=episode_config,
                    trial_dir=opt_dir / f"trial_{global_trial_idx + i}",
                    llm=g_llms[i],
                    protagonist_id=protagonist_id,
                    protagonist_name=protagonist_name,
                    target_words=target_words,
                    character_profiles=character_profiles,
                    reader_feedback=reader_feedback,
                    guardian_briefing=guardian_briefing,
                    base_policy=current_params,
                    quality_focus=quality_focus,
                )
                for i in range(group_n)
            ],
            return_exceptions=False,
        )

        for i, (score, path, meta) in enumerate(g_results):
            tidx = global_trial_idx + i
            study.tell(g_optuna_trials[i], score)
            trial_scores.append(score)
            trial_paths.append(path)
            trial_meta_by_idx[tidx] = meta
            all_trial_params.append(g_params[i])

        if notify_fn:
            g_scores = trial_scores[global_trial_idx: global_trial_idx + group_n]
            g_lines = []
            for i, sc in enumerate(g_scores):
                meta = trial_meta_by_idx.get(global_trial_idx + i, {})
                g_lines.append(
                    f"  t{global_trial_idx + i}: {sc:.2f} "
                    f"(det {float(meta.get('det', 0)):.2f} / llm {float(meta.get('llm', 0)):.2f})"
                )
            await notify_fn(
                f"[MINI-OPT] group {group_i + 1} 결과:\n" + "\n".join(g_lines)
            )

        global_trial_idx += group_n

    # ── Pick best ─────────────────────────────────────────────────────────────
    best_idx = max(range(len(trial_scores)), key=lambda i: trial_scores[i])
    best_score = trial_scores[best_idx]
    best_params = all_trial_params[best_idx]
    best_path = trial_paths[best_idx]

    # ── Build subtrial data for logging ──────────────────────────────────────
    subtrial_data: list[dict] = []
    for i, (score, params) in enumerate(zip(trial_scores, all_trial_params)):
        meta = trial_meta_by_idx.get(i, {})
        subtrial_data.append({
            "trial_idx": i,
            "score": round(score, 4),
            "det": round(float(meta.get("det", 0.0)), 3),
            "llm": round(float(meta.get("llm", 0.0)), 3),
            "params": dict(params),
        })

    if notify_fn:
        sorted_scores = sorted(enumerate(trial_scores), key=lambda x: x[1], reverse=True)
        ranking = " | ".join(f"t{idx}:{sc:.2f}" for idx, sc in sorted_scores[:5])
        await notify_fn(
            f"[MINI-OPT] outer {cycle_idx} 전체 완료 | {n_trials} trials\n"
            f"best={best_score:.2f} (trial {best_idx}) | top5: {ranking}\n"
            f"누적 study trials: {n_past + n_trials}"
        )

    # ── Persist best params → rl_policy.json ─────────────────────────────────
    update_rl_policy(best_params, best_score, episode_id)

    resolved_path: Path | None = best_path if (best_path and best_path.exists()) else None
    return resolved_path, best_params, best_score, subtrial_data


# ── Param factor analysis (hyperparameter X → quality Y) ─────────────────────

def param_factor_analysis(
    n_datapoints: int | None = None,
    log_path: Path | None = None,
) -> str:
    """Ridge regression: param vector X → quality dimension Y.

    Reads subtrial entries from cycle_score_log.jsonl.
    Each entry is one subtrial (trial_idx, params, score).
    AI review scores (thrill/style/etc.) are taken from the parent cycle record.

    n_datapoints=None uses ALL accumulated data (default).
    Pass an integer to limit to the most recent N datapoints.

    Returns a formatted text report suitable for inclusion in manager prompt.
    """
    log_path = log_path or CYCLE_SCORE_LOG
    if not log_path.exists():
        return "cycle_score_log.jsonl 없음 — 파라미터 분석 데이터 없음."

    # ── Load flat subtrial rows ──────────────────────────────────────────────
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ai = rec.get("ai_review", {})
        for st in rec.get("subtrials", []):
            params = st.get("params", {})
            if not params:
                continue
            rows.append({
                "subtrial_score": float(st.get("score", 0.0)),
                "thrill":   float(ai.get("thrill", 0)),
                "style":    float(ai.get("style", 0)),
                "causality":float(ai.get("causality", 0)),
                "character":float(ai.get("character", 0)),
                "scene_fn": float(ai.get("scene_fn", 0)),
                **{k: float(v) for k, v in params.items() if isinstance(v, (int, float))},
            })

    if len(rows) < 5:
        return (
            f"파라미터 분석 데이터 부족 ({len(rows)}개 서브트라이얼, 최소 5개 필요).\n"
            f"데이터 축적 중입니다."
        )

    # Use all data by default; optionally cap to most recent n_datapoints
    if n_datapoints is not None:
        rows = rows[-n_datapoints:]
    param_keys = [k for k in _PARAM_FULL_RANGES if k in rows[0]]
    if not param_keys:
        return "cycle_score_log에 numeric 파라미터 컬럼 없음."

    try:
        import numpy as np
    except ImportError:
        return "numpy 미설치 — 파라미터 분석 생략."

    X = np.array([[r.get(k, 0.0) for k in param_keys] for r in rows])
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    y_dims = {
        "subtrial_score": np.array([r["subtrial_score"] for r in rows]),
        "thrill":         np.array([r["thrill"]   for r in rows]),
        "style":          np.array([r["style"]    for r in rows]),
        "causality":      np.array([r["causality"] for r in rows]),
        "character":      np.array([r["character"] for r in rows]),
        "scene_fn":       np.array([r["scene_fn"]  for r in rows]),
    }

    report_lines = [
        f"## 파라미터 Factor Analysis ({len(rows)}개 서브트라이얼 전체 누적)",
        f"분석 파라미터: {', '.join(param_keys)}",
        "",
    ]

    try:
        from sklearn.linear_model import Ridge
        for dim_name, y in y_dims.items():
            if float(y.std()) < 0.01:
                report_lines.append(f"{dim_name}: 분산 없음 (점수 고정)")
                continue
            model = Ridge(alpha=1.0)
            model.fit(X_norm, y)
            coefs = dict(zip(param_keys, model.coef_))
            top = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            top_str = ", ".join(f"{k}:{v:+.3f}" for k, v in top)
            report_lines.append(f"{dim_name}: {top_str}")
    except ImportError:
        # Fallback: Pearson correlation on subtrial_score
        report_lines.append("(sklearn 미설치 → Pearson 상관계수 fallback)")
        y = y_dims["subtrial_score"]
        cors: list[tuple[str, float]] = []
        for i, k in enumerate(param_keys):
            xi = X_norm[:, i]
            if float(xi.std()) < 1e-6:
                cors.append((k, 0.0))
                continue
            cor = float(np.corrcoef(xi, y)[0, 1])
            cors.append((k, cor))
        cors.sort(key=lambda x: abs(x[1]), reverse=True)
        report_lines.append(
            "subtrial_score 상관계수: "
            + ", ".join(f"{k}:{v:+.3f}" for k, v in cors[:8])
        )

    return "\n".join(report_lines)
