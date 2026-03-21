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
