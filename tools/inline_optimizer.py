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
import time
from collections import Counter
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("inline_optimizer")

REVIEWER_MODEL = "gpt-4o-mini"
POLICY_SCORE_LOG = _REPO_ROOT / "data" / "policy_score_log.jsonl"
CYCLE_SCORE_LOG = _REPO_ROOT / "data" / "cycle_score_log.jsonl"
SESSION_BENCHMARK_LOG = "benchmark_subtrials.jsonl"

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
_REPETITION_STOPWORDS = {
    "그리고", "그러나", "하지만", "그래서", "정말", "아주", "매우", "조금",
    "것", "수", "더", "또", "그", "이", "저", "때문", "정도",
}


def _extract_episode_stopwords(episode_id: str) -> set[str]:
    """Extract episode-specific vocabulary from its config YAML.

    Words that appear 3+ times in the episode config are domain/character vocabulary
    expected to repeat in the generated text — they should not count as prose repetition.
    """
    try:
        import yaml  # type: ignore
        config_path = _REPO_ROOT / "config" / "episodes" / f"{episode_id}.yaml"
        if not config_path.exists():
            return set()
        raw = config_path.read_text(encoding="utf-8")
        text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", raw.lower())
        counts = Counter(t for t in text.split() if len(t) >= 2)
        return {t for t, c in counts.items() if c >= 3}
    except Exception:
        return set()


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
    p_min = trial.suggest_int("prose_paragraph_min_sentences", 2, 4)
    p_max = trial.suggest_int("prose_paragraph_max_sentences", 3, 5)
    if p_min > p_max:
        p_min = p_max
    return {
        "distiller_temperature":         trial.suggest_float("distiller_temperature", 0.05, 0.45),
        "prose_scene_temperature":       trial.suggest_float("prose_scene_temperature", 0.55, 0.85),
        "prose_paragraph_min_sentences": p_min,
        "prose_paragraph_max_sentences": p_max,
        "prose_transition_temperature":  trial.suggest_float("prose_transition_temperature", 0.3, 0.7),
        "prose_polish_temperature":      trial.suggest_float("prose_polish_temperature", 0.2, 0.6),
        "hold_pressure_peak":            trial.suggest_categorical("hold_pressure_peak", [0, 1]),
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


def _score_llm_claude(chapter_text: str, criterion_weights: dict[str, float] | None = None) -> float:
    """Claude-based LLM scorer. Ensembles with GPT to reduce self-preference bias (Fix B)."""
    try:
        import anthropic
    except ImportError:
        return 0.0

    text = chapter_text.strip()
    if len(text) > 3000:
        excerpt = text[:2000] + "\n\n...[중략]...\n\n" + text[-1000:]
    else:
        excerpt = text

    prompt = _LLM_SCORER_CRITERIA + f"\n--- EXCERPT ---\n{excerpt}\n--- END ---"
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_LLM_SCORER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (msg.content[0].text if msg.content else "").strip()
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
        logger.warning("Claude LLM scorer error: %s", exc)
        return 0.0


_LLM_SCORE_RUNS = 3  # LLM 채점 반복 횟수 — 평균으로 분산 감소


def _score_chapter(
    chapter_text: str, episode_id: str, quality_focus: dict | None = None
) -> tuple[float, float, float, float]:
    det = _score_deterministic(chapter_text, episode_id)
    weights = _build_criterion_weights(quality_focus)

    # GPT + Claude 각 3회씩 채점 후 평균 (LLM 분산 감소)
    gpt_scores = [_score_llm(chapter_text, criterion_weights=weights) for _ in range(_LLM_SCORE_RUNS)]
    gpt_score = round(sum(gpt_scores) / len(gpt_scores), 3)

    claude_scores = [_score_llm_claude(chapter_text, criterion_weights=weights) for _ in range(_LLM_SCORE_RUNS)]
    valid_claude = [s for s in claude_scores if s > 0.0]
    claude_score = round(sum(valid_claude) / len(valid_claude), 3) if valid_claude else 0.0

    # Ensemble GPT + Claude to reduce self-preference bias (Fix B)
    if claude_score > 0.0:
        llm = round((gpt_score + claude_score) / 2, 3)
    else:
        llm = gpt_score

    episode_stopwords = _extract_episode_stopwords(episode_id)
    repetition_penalty = _repetition_penalty(chapter_text, extra_stopwords=episode_stopwords)
    # LLM 점수만 사용
    combined = round(max(0.0, float(llm or 0.0)), 3)
    logger.info(
        "[SCORE] det=%.2f gpt=%.2f(×%d) claude=%.2f(×%d) ensemble=%.2f rep_penalty=%.3f combined=%.3f focus=%s",
        det, gpt_score, _LLM_SCORE_RUNS, claude_score, _LLM_SCORE_RUNS,
        llm, repetition_penalty, combined,
        list(quality_focus.keys()) if quality_focus else None,
    )
    return combined, det, llm, repetition_penalty


def _repetition_penalty(chapter_text: str, extra_stopwords: set[str] | None = None) -> float:
    text = str(chapter_text or "").strip()
    if not text:
        return 0.0

    all_stopwords = _REPETITION_STOPWORDS | (extra_stopwords or set())
    normalized = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text.lower())
    tokens = [tok for tok in normalized.split() if len(tok) >= 2 and tok not in all_stopwords]
    if len(tokens) < 80:
        return 0.0

    token_counts = Counter(tokens)
    repeated_token_ratio = sum(1 for count in token_counts.values() if count >= 8) / max(1, len(token_counts))
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counts = Counter(bigrams)
    repeated_bigram_ratio = sum(1 for count in bigram_counts.values() if count >= 4) / max(1, len(bigram_counts))

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?…])\s+|(?<=다\.)\s+", text)
        if s.strip()
    ]
    local_repeat_hits = 0
    for prev, curr in zip(sentences, sentences[1:]):
        prev_norm = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", prev.lower()).split()
        curr_norm = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", curr.lower()).split()
        if not prev_norm or not curr_norm:
            continue
        prev_set, curr_set = set(prev_norm), set(curr_norm)
        overlap = len(prev_set & curr_set) / max(1, len(prev_set | curr_set))
        if overlap >= 0.80:
            local_repeat_hits += 1

    penalty = min(
        1.5,
        repeated_token_ratio * 2.0 + repeated_bigram_ratio * 2.5 + (local_repeat_hits / max(1, len(sentences))) * 3.0,
    )
    return round(penalty, 3)


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
            target_scenes=int(base_policy.get("target_scenes", 7)),
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
    repetition_penalty = 0.0
    word_count = 0
    try:
        chapter_text = Path(chapter_path).read_text(encoding="utf-8")
        word_count = len(chapter_text.split())
        score, det, llm, repetition_penalty = _score_chapter(
            chapter_text,
            episode_id,
            quality_focus=quality_focus,
        )
    except Exception as exc:
        logger.warning("Trial %d: scoring failed: %s", trial_idx, exc)
        score = 0.0

    logger.info("Trial %d | score=%.3f | path=%s", trial_idx, score, chapter_path)
    return score, Path(chapter_path), {
        "det": round(det, 3),
        "llm": round(llm, 3),
        "repetition_penalty": round(repetition_penalty, 3),
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
    cost_tracker: dict | None = None,
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
            if cost_tracker is not None:
                cost_tracker["phase_a_trials"] = (
                    float(cost_tracker.get("phase_a_trials", 0.0))
                    + float((final_llm.budget_summary() or {}).get("spent_usd", 0.0))
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

    # ── Accumulate trial LLM costs into cost_tracker ─────────────────────────
    if cost_tracker is not None:
        trial_llms_spent = sum(
            float((llm.budget_summary() or {}).get("spent_usd", 0.0))
            for llm in [*p1_llms, *p2_llms]
        )
        cost_tracker["phase_a_trials"] = (
            float(cost_tracker.get("phase_a_trials", 0.0)) + trial_llms_spent
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


def _recent_cycle_ai_avg(episode_id: str, n_cycles: int = 3) -> float | None:
    """Return the mean AI review avg from the last n_cycles entries for this episode.

    Returns None if no qualifying entries found.
    """
    import json as _json
    csl = CYCLE_SCORE_LOG
    if not csl.exists():
        return None
    try:
        lines = [l for l in csl.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return None
    avgs: list[float] = []
    for line in reversed(lines):
        if len(avgs) >= n_cycles:
            break
        try:
            row = _json.loads(line)
        except Exception:
            continue
        if row.get("episode_id") != episode_id:
            continue
        ai_rev = row.get("ai_review") or {}
        avg = ai_rev.get("avg")
        if avg is not None:
            avgs.append(float(avg))
    return sum(avgs) / len(avgs) if avgs else None


def update_rl_policy(best_params: dict, best_score: float, episode_id: str) -> None:
    """Write inline optimizer best params back to rl_policy.json.

    Guard: only writes if recent AI reviewer avg >= 7.0.
    Prevents proxy-metric wins from overwriting the policy when
    the final quality reviewer disagrees.
    """
    import json
    from datetime import date

    # ── AI reviewer quality guard ─────────────────────────────────────────────
    recent_avg = _recent_cycle_ai_avg(episode_id)
    if recent_avg is not None and recent_avg < 7.0:
        logger.warning(
            "update_rl_policy SKIPPED — recent AI review avg=%.2f < 7.0 for %s "
            "(proxy score=%.3f). Policy unchanged.",
            recent_avg, episode_id, best_score,
        )
        return

    policy_path = _REPO_ROOT / "data" / "rl_policy.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
    except Exception:
        policy = {}

    policy["version"] = int(policy.get("version", 2)) + 1
    # Update only the params that inline optimizer controls
    INLINE_PARAM_KEYS = {
        "distiller_temperature",
        "prose_scene_temperature", "prose_paragraph_min_sentences", "prose_paragraph_max_sentences",
        "prose_transition_temperature", "prose_polish_temperature",
        "hold_pressure_peak",
    }
    for k, v in best_params.items():
        if k in INLINE_PARAM_KEYS:
            policy[k] = v
    policy["_last_inline_opt"] = {
        "date": str(date.today()),
        "episode_id": episode_id,
        "score": round(best_score, 4),
        "ai_review_avg": round(recent_avg, 3) if recent_avg is not None else None,
    }
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "rl_policy.json updated → version %s score=%.3f ai_avg=%s",
        policy["version"], best_score,
        f"{recent_avg:.2f}" if recent_avg is not None else "n/a",
    )


# ── Cycle score logging ────────────────────────────────────────────────────────

def log_cycle_score(
    episode_id: str,
    cycle_idx: int,
    current_params: dict,
    ai_review_scores: dict,
    subtrial_data: list[dict],
    log_path: Path | None = None,
    cost_tracker: dict | None = None,
) -> None:
    """Log one AUTO cycle's data: params, AI review scores, and subtrial results.

    ai_review_scores: {thrill, style, causality, character, scene_fn, avg}
    subtrial_data:    [{trial_idx, score, det, llm, params}, ...]
    cost_tracker:     {guardian, simulation, chapter, auto_chapter, manager, auto_review, ...}
    """
    log_path = log_path or CYCLE_SCORE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _cost_breakdown: dict = {}
    _total_cost_usd = 0.0
    if cost_tracker:
        _cost_keys = [
            "guardian", "simulation", "chapter", "auto_chapter",
            "phase_a_trials", "manager", "auto_review", "code_review",
            "regen_check", "final_review", "feedback_parse",
        ]
        for k in _cost_keys:
            v = float(cost_tracker.get(k, 0.0))
            if v > 0.0:
                _cost_breakdown[k] = round(v, 6)
                _total_cost_usd += v
    record = {
        "date": str(date.today()),
        "episode_id": episode_id,
        "cycle_idx": cycle_idx,
        "ai_review": ai_review_scores,
        "cycle_params": current_params,
        "subtrials": subtrial_data,
        "cost_usd": round(_total_cost_usd, 6),
        "cost_breakdown": _cost_breakdown,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Logged cycle score for %s cycle %d → %s", episode_id, cycle_idx, log_path)


def append_session_benchmark_row(
    run_dir: Path,
    *,
    episode_id: str,
    cycle_idx: int,
    trial_idx: int,
    study_trial_count_before: int,
    score: float,
    det: float,
    llm: float,
    repetition_penalty: float,
    params: dict[str, object],
) -> None:
    """Append one completed subtrial to this session's benchmark log."""
    log_path = run_dir / SESSION_BENCHMARK_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": int(time.time()),
        "episode_id": episode_id,
        "cycle_idx": int(cycle_idx),
        "trial_idx": int(trial_idx),
        "global_trial_idx": int(study_trial_count_before + trial_idx),
        "score": round(float(score), 4),
        "det": round(float(det), 4),
        "llm": round(float(llm), 4),
        "repetition_penalty": round(float(repetition_penalty), 4),
        "params": dict(params),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Narrow param space for per-cycle mini re-optimize ─────────────────────────

# Full search ranges mirroring _param_space()
_PARAM_FULL_RANGES: dict[str, tuple] = {
    "distiller_temperature":         ("float", 0.05, 0.45),
    "prose_scene_temperature":       ("float", 0.55, 0.85),
    "prose_paragraph_min_sentences": ("int",   2,    4),
    "prose_paragraph_max_sentences": ("int",   3,    5),
    "prose_transition_temperature":  ("float", 0.3,  0.7),
    "prose_polish_temperature":      ("float", 0.2,  0.6),
    "hold_pressure_peak":            ("cat",   [0, 1]),
}

# Fix C: Split sim-level vs prose-level params for 2-study optimization.
# sim params → affect distillation (what content enters the scene pipeline)
# prose params → affect only how scenes are rendered as text
_PARAM_SIM_RANGES: dict[str, tuple] = {
    "distiller_temperature":         ("float", 0.05, 0.45),
}

_PARAM_PROSE_RANGES: dict[str, tuple] = {
    "prose_scene_temperature":       ("float", 0.45, 0.95),
    "prose_paragraph_min_sentences": ("int",   2,    4),
    "prose_paragraph_max_sentences": ("int",   3,    6),
    "prose_transition_temperature":  ("float", 0.30, 0.80),
    "prose_polish_temperature":      ("float", 0.15, 0.65),
    "hold_pressure_peak":            ("cat",   [0, 1]),
}


def _narrow_width_from_n_trials(n_past_trials: int, force_wide: bool = False) -> float:
    """Compute search width_ratio based on accumulated trial count.

    Fix A: start raised 0.30→0.50, floor raised 0.25→0.40.
    Prevents premature convergence in multimodal quality landscapes.
    Schedule: shrinks 15% every 10 trials.

    force_wide: if True (consecutive AI-score drops detected), reset to 0.50.

    Examples:
      0  trials → 0.50 (wide)
     10  trials → 0.50 (capped at floor=0.40 quickly)
     20  trials → 0.40 (floor)
     50+ trials → 0.40 (floor)
    """
    if force_wide:
        return 0.50
    return max(0.40, 0.50 * (0.85 ** (n_past_trials // 10)))


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
            # Always use full choices — pinning to [current] blocks exploration entirely
            params[name] = trial.suggest_categorical(name, choices)
    # Guard: prose_paragraph_min_sentences must not exceed max_sentences
    if "prose_paragraph_min_sentences" in params and "prose_paragraph_max_sentences" in params:
        if params["prose_paragraph_min_sentences"] > params["prose_paragraph_max_sentences"]:
            params["prose_paragraph_min_sentences"] = params["prose_paragraph_max_sentences"]
    return params


# ── Warmup helpers ────────────────────────────────────────────────────────────

def _enqueue_warmup_trials(study, psl_path: str, warmup_keys: set) -> int:
    """Add past best-param trials from policy_score_log.jsonl to an in-memory study.

    Each JSONL row that has best_params + best_score is added as a COMPLETE trial
    so TPE can start from accumulated cross-episode knowledge.

    Returns the number of trials successfully added.
    """
    import json as _json
    from datetime import datetime as _dt
    try:
        import optuna
        from optuna.distributions import (
            FloatDistribution,
            IntDistribution,
            CategoricalDistribution,
        )
        from optuna.trial import FrozenTrial, TrialState
    except ImportError:
        return 0

    # Build distribution map from _PARAM_FULL_RANGES
    dist_map: dict = {}
    for name, spec in _PARAM_FULL_RANGES.items():
        if name not in warmup_keys:
            continue
        kind = spec[0]
        if kind == "float":
            dist_map[name] = FloatDistribution(float(spec[1]), float(spec[2]))
        elif kind == "int":
            dist_map[name] = IntDistribution(int(spec[1]), int(spec[2]))
        elif kind == "cat":
            dist_map[name] = CategoricalDistribution(list(spec[1]))

    n_added = 0
    try:
        with open(psl_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        best_params = row.get("best_params", {})
        best_score = row.get("best_score")
        if best_score is None or not best_params:
            continue

        # Quality guard: skip warmup entries where the final AI reviewer avg < 7.0
        qr = row.get("quality_review_scores") or {}
        if qr:
            scores = [v for k, v in qr.items() if k != "avg" and isinstance(v, (int, float))]
            qr_avg = qr.get("avg") or (sum(scores) / len(scores) if scores else None)
            if qr_avg is not None and float(qr_avg) < 7.0:
                logger.debug(
                    "_enqueue_warmup_trials: skipping row date=%s score=%.3f qr_avg=%.2f < 7.0",
                    row.get("date", "?"), float(best_score), float(qr_avg),
                )
                continue

        # Filter to keys that are in both warmup_keys and dist_map
        trial_params: dict = {}
        trial_dists: dict = {}
        for k, dist in dist_map.items():
            if k not in best_params:
                continue
            val = best_params[k]
            # Type-coerce
            if isinstance(dist, FloatDistribution):
                val = float(val)
            elif isinstance(dist, IntDistribution):
                val = int(round(float(val)))
            trial_params[k] = val
            trial_dists[k] = dist

        if not trial_params:
            continue

        now = _dt.now()
        try:
            frozen = FrozenTrial(
                number=n_added,
                trial_id=n_added,
                state=TrialState.COMPLETE,
                value=float(best_score),
                datetime_start=now,
                datetime_complete=now,
                params=trial_params,
                distributions=trial_dists,
                intermediate_values={},
                user_attrs={},
                system_attrs={},
            )
            study.add_trial(frozen)
            n_added += 1
        except Exception:
            continue

    return n_added


# ── Fix C helpers: distill-only and prose-from-scenes ────────────────────────

def _sync_run_distill(
    sim_params: dict,
    episode_id: str,
    episode_config: dict,
    llm,
    protagonist_id: str,
    base_policy: dict,
    reader_feedback: dict | None,
) -> list:
    """Run distillation only with sim_params. Returns list of DistilledScene."""
    from src.novel_writer.scene_distiller import SceneDistiller
    trial_policy = dict(base_policy)
    trial_policy.update(sim_params)
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
            target_scenes=int(sim_params.get("target_scenes", base_policy.get("target_scenes", 6))),
        )
        scenes = distiller.normalize_scene_timeline(distiller.apply_scene_guards(scenes))
        return scenes or []
    except Exception as exc:
        logger.warning("Distill failed: %s", exc)
        return []


def _sync_run_prose_from_scenes(
    prose_params: dict,
    scenes: list,
    episode_id: str,
    episode_config: dict,
    trial_dir: Path,
    llm,
    protagonist_name: str,
    target_words: int,
    character_profiles,
    reader_feedback: dict | None,
    guardian_briefing: str | None,
    base_policy: dict,
    quality_focus: dict | None = None,
) -> tuple[float, Path, dict]:
    """Run prose generation from pre-distilled scenes. Returns (score, path, meta)."""
    from src.novel_writer.prose_generator import ProseGenerator
    if not scenes:
        return 0.0, trial_dir / "empty.md", {"det": 0.0, "llm": 0.0, "scene_count": 0}

    trial_policy = dict(base_policy)
    trial_policy.update(prose_params)
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
        logger.warning("Prose from scenes failed: %s", exc)
        return 0.0, trial_dir / "empty.md", {
            "det": 0.0, "llm": 0.0, "scene_count": len(scenes),
            "raw_turn_total": int(sum(max(0, int(s.raw_turn_count or 0)) for s in scenes)),
        }

    det = 0.0
    llm_val = 0.0
    repetition_penalty = 0.0
    word_count = 0
    score = 0.0
    try:
        chapter_text = Path(chapter_path).read_text(encoding="utf-8")
        word_count = len(chapter_text.split())
        score, det, llm_val, repetition_penalty = _score_chapter(
            chapter_text, episode_id, quality_focus=quality_focus,
        )
    except Exception as exc:
        logger.warning("Scoring failed: %s", exc)
    return score, Path(chapter_path), {
        "det": round(det, 3),
        "llm": round(llm_val, 3),
        "repetition_penalty": round(repetition_penalty, 3),
        "scene_count": len(scenes),
        "raw_turn_total": int(sum(max(0, int(s.raw_turn_count or 0)) for s in scenes)),
        "chapter_file": Path(chapter_path).name,
        "word_count": int(word_count),
    }


def _param_space_from_ranges(trial, param_ranges: dict, base_policy: dict, width_ratio: float) -> dict:
    """Generic narrow param sampler for any param_ranges dict."""
    params: dict = {}
    for name, spec in param_ranges.items():
        kind = spec[0]
        current = base_policy.get(name)
        if kind == "float":
            lo, hi = float(spec[1]), float(spec[2])
            if current is not None:
                cur = max(lo, min(hi, float(current)))
                half = (hi - lo) * width_ratio / 2.0
                c_lo = max(lo, cur - half)
                c_hi = min(hi, cur + half)
                if c_lo >= c_hi:
                    c_lo = max(lo, cur - (hi - lo) * 0.1)
                    c_hi = min(hi, cur + (hi - lo) * 0.1)
                    if c_lo >= c_hi:
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
            # Always use full choices — pinning to [current] blocks exploration entirely
            params[name] = trial.suggest_categorical(name, choices)
    # Guard: prose_paragraph_min_sentences must not exceed max_sentences
    if "prose_paragraph_min_sentences" in params and "prose_paragraph_max_sentences" in params:
        if params["prose_paragraph_min_sentences"] > params["prose_paragraph_max_sentences"]:
            params["prose_paragraph_min_sentences"] = params["prose_paragraph_max_sentences"]
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
    """Fix C: 2-study hierarchical param re-optimize.

    study_sim  : optimises distillation-level params (what scenes to keep).
    study_prose: optimises prose-level params (how scenes are rendered).

    Structure: n_sim_trials = n_trials // group_size distillation attempts,
    each with group_size parallel prose trials.
    Total chapter generations = n_trials (same as before).

    Fix A: CMA-ES sampler when warmup data ≥ 5 rows, TPE otherwise.
    Fix B: Claude ensemble scoring used inside _score_chapter.

    Returns:
        (best_chapter_path | None, best_params, best_score, subtrial_data_list)
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

    # Derive 2-study dimensions from n_trials / group_size
    n_prose_per_sim = max(1, group_size)
    n_sim_trials = max(1, n_trials // n_prose_per_sim)
    prose_budget = budget / max(n_trials, 1)

    _psl = _REPO_ROOT / "data" / "policy_score_log.jsonl"
    _n_psl_rows = sum(1 for ln in _psl.read_text(encoding="utf-8").splitlines() if ln.strip()) if _psl.exists() else 0

    def _make_sampler(prefix: str) -> "optuna.samplers.BaseSampler":
        if _n_psl_rows >= 5:
            return optuna.samplers.CmaEsSampler(
                n_startup_trials=10,
                restart_strategy="ipop",
                seed=42,
                independent_sampler=optuna.samplers.TPESampler(n_startup_trials=2, seed=42),
            )
        return optuna.samplers.TPESampler(n_startup_trials=5, seed=42)

    # ── Create two in-memory studies ─────────────────────────────────────────
    study_sim = optuna.create_study(
        study_name=f"sim_{episode_id}_outer{cycle_idx}",
        direction="maximize",
        sampler=_make_sampler("sim"),
    )
    study_prose = optuna.create_study(
        study_name=f"prose_{episode_id}_outer{cycle_idx}",
        direction="maximize",
        sampler=_make_sampler("prose"),
    )

    # ── Consecutive AI-score drop detection (Fix B-3) ────────────────────────
    # Read last 3 cycle entries for this episode; if AI avg is strictly declining
    # over 2+ steps, force wide search to escape the local optimum.
    _force_wide_search = False
    if cycle_idx >= 2:
        import json as _json_csl
        _csl = CYCLE_SCORE_LOG
        _recent_avgs: list[float] = []
        if _csl.exists():
            for _ln in reversed(_csl.read_text(encoding="utf-8").splitlines()):
                if len(_recent_avgs) >= 3:
                    break
                _ln = _ln.strip()
                if not _ln:
                    continue
                try:
                    _r = _json_csl.loads(_ln)
                except Exception:
                    continue
                if _r.get("episode_id") != episode_id:
                    continue
                _ai = (_r.get("ai_review") or {}).get("avg")
                if _ai is not None:
                    _recent_avgs.append(float(_ai))
        # Strictly declining: each successive value lower than the previous
        if len(_recent_avgs) >= 2 and all(
            _recent_avgs[i] > _recent_avgs[i + 1] for i in range(len(_recent_avgs) - 1)
        ):
            _force_wide_search = True
            logger.warning(
                "[MINI-OPT] consecutive AI-avg drops detected %s → forcing wide search (width=0.50)",
                list(reversed(_recent_avgs)),
            )

    # Warm-start both studies from accumulated cross-episode data
    n_warmup_sim = _enqueue_warmup_trials(study_sim, str(_psl), set(_PARAM_SIM_RANGES)) if _psl.exists() else 0
    n_warmup_prose = _enqueue_warmup_trials(study_prose, str(_psl), set(_PARAM_PROSE_RANGES)) if _psl.exists() else 0
    n_past = n_warmup_sim + n_warmup_prose

    logger.info(
        "[MINI-OPT] episode=%s cycle=%d n_sim=%d n_prose_per_sim=%d warmup_sim=%d warmup_prose=%d force_wide=%s",
        episode_id, cycle_idx, n_sim_trials, n_prose_per_sim, n_warmup_sim, n_warmup_prose, _force_wide_search,
    )

    if notify_fn:
        _wide_note = " ⚠️ force_wide=True" if _force_wide_search else ""
        await notify_fn(
            f"[MINI-OPT] outer {cycle_idx} — 2-study 시작 "
            f"({n_sim_trials} distill × {n_prose_per_sim} prose = {n_sim_trials * n_prose_per_sim}회, "
            f"warmup sim={n_warmup_sim} prose={n_warmup_prose}){_wide_note}"
        )

    trial_scores: list[float] = []
    trial_paths: list[Path] = []
    trial_meta_by_idx: dict[int, dict] = {}
    all_trial_params: list[dict] = []
    global_prose_idx = 0

    # ── Outer loop: sim params ─────────────────────────────────────────────
    for sim_i in range(n_sim_trials):
        width_sim = _narrow_width_from_n_trials(n_past // 2 + sim_i, force_wide=_force_wide_search)
        width_prose = _narrow_width_from_n_trials(n_past // 2 + global_prose_idx, force_wide=_force_wide_search)

        # Sample sim params
        sim_optuna_trial = study_sim.ask()
        sim_params = _param_space_from_ranges(sim_optuna_trial, _PARAM_SIM_RANGES, current_params, width_sim)

        if notify_fn:
            _sim_str = ", ".join(
                f"{k}={round(v, 3) if isinstance(v, float) else v}"
                for k, v in sim_params.items()
            )
            await notify_fn(
                f"[MINI-OPT-SIM] sim {sim_i + 1}/{n_sim_trials} — 증류 시작\n"
                f"파라미터: {_sim_str}"
            )

        # Distill once for this sim trial
        distill_llm = LLMClient(model=base_model, premium_model=premium_model, budget_usd=prose_budget)
        scenes = await asyncio.to_thread(
            _sync_run_distill,
            sim_params, episode_id, episode_config, distill_llm,
            protagonist_id, current_params, reader_feedback,
        )

        if not scenes:
            study_sim.tell(sim_optuna_trial, 0.0)
            if notify_fn:
                await notify_fn(f"[MINI-OPT-SIM] sim {sim_i + 1} 증류 실패 (0씬) — 스킵")
            continue

        if notify_fn:
            _scene_lines = []
            for _s in scenes[:5]:
                _title = getattr(_s, "title", None) or (
                    _s.get("title") if isinstance(_s, dict) else "—"
                )
                _arc = getattr(_s, "emotional_arc", None) or (
                    _s.get("emotional_arc") if isinstance(_s, dict) else ""
                )
                _pacing = getattr(_s, "pacing", None) or (
                    _s.get("pacing") if isinstance(_s, dict) else ""
                )
                _arc_short = str(_arc)[:40] if _arc else "—"
                _scene_lines.append(f"  · {_title} [{_pacing}] {_arc_short}")
            _scene_summary = "\n".join(_scene_lines)
            await notify_fn(
                f"[MINI-OPT-SIM] sim {sim_i + 1} 증류 완료 — {len(scenes)}씬 생성\n"
                f"{_scene_summary}\n"
                f"→ prose {n_prose_per_sim}개 병렬 생성 시작"
            )

        # Sample prose params for this group
        prose_optuna_trials = [study_prose.ask() for _ in range(n_prose_per_sim)]
        prose_params_list = [
            _param_space_from_ranges(t, _PARAM_PROSE_RANGES, current_params, width_prose)
            for t in prose_optuna_trials
        ]
        prose_llms = [
            LLMClient(model=base_model, premium_model=premium_model, budget_usd=prose_budget)
            for _ in range(n_prose_per_sim)
        ]

        # Run prose trials in parallel
        prose_results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _sync_run_prose_from_scenes,
                    prose_params_list[j],
                    scenes,
                    episode_id,
                    episode_config,
                    opt_dir / f"s{sim_i}_p{j}",
                    prose_llms[j],
                    protagonist_name,
                    target_words,
                    character_profiles,
                    reader_feedback,
                    guardian_briefing,
                    {**current_params, **sim_params},
                    quality_focus,
                )
                for j in range(n_prose_per_sim)
            ],
            return_exceptions=False,
        )

        prose_scores_this_sim: list[float] = []
        for j, (p_score, p_path, p_meta) in enumerate(prose_results):
            tidx = global_prose_idx + j
            study_prose.tell(prose_optuna_trials[j], p_score)
            trial_scores.append(p_score)
            trial_paths.append(p_path)
            trial_meta_by_idx[tidx] = p_meta
            combined_params = {**sim_params, **prose_params_list[j]}
            all_trial_params.append(combined_params)
            prose_scores_this_sim.append(p_score)
            append_session_benchmark_row(
                run_dir,
                episode_id=episode_id,
                cycle_idx=cycle_idx,
                trial_idx=tidx,
                study_trial_count_before=n_past,
                score=p_score,
                det=float(p_meta.get("det", 0.0)),
                llm=float(p_meta.get("llm", 0.0)),
                repetition_penalty=float(p_meta.get("repetition_penalty", 0.0)),
                params=combined_params,
            )

        # Tell study_sim the average prose score for this distillation
        sim_avg = sum(prose_scores_this_sim) / max(1, len(prose_scores_this_sim))
        study_sim.tell(sim_optuna_trial, sim_avg)

        if notify_fn:
            best_j = max(range(n_prose_per_sim), key=lambda j: prose_scores_this_sim[j])
            p_lines = []
            for j in range(n_prose_per_sim):
                meta_j = trial_meta_by_idx.get(global_prose_idx + j, {})
                marker = "★" if j == best_j else " "
                rep = float(meta_j.get("repetition_penalty", 0))
                p_lines.append(
                    f" {marker} p{j + 1}: {prose_scores_this_sim[j]:.3f} "
                    f"(결정적 {float(meta_j.get('det', 0)):.2f} / "
                    f"LLM {float(meta_j.get('llm', 0)):.2f} / "
                    f"반복패널티 -{rep:.2f})"
                )
            _best_prose_params = prose_params_list[best_j]
            _best_str = ", ".join(
                f"{k}={round(v, 3) if isinstance(v, float) else v}"
                for k, v in _best_prose_params.items()
            )
            await notify_fn(
                f"[MINI-OPT-SCORE] sim {sim_i + 1} 평가 완료 (avg={sim_avg:.3f})\n"
                + "\n".join(p_lines) + "\n"
                f"★ 베스트 prose 파라미터: {_best_str}"
            )

        global_prose_idx += n_prose_per_sim

    if not trial_scores:
        logger.warning("[MINI-OPT] No successful trials — returning current_params")
        return None, current_params, 0.0, []

    # ── Pick best ─────────────────────────────────────────────────────────────
    best_idx = max(range(len(trial_scores)), key=lambda i: trial_scores[i])
    best_score = trial_scores[best_idx]
    best_params = all_trial_params[best_idx]
    best_path = trial_paths[best_idx]

    if notify_fn:
        sorted_scores = sorted(enumerate(trial_scores), key=lambda x: x[1], reverse=True)
        top5 = sorted_scores[:5]
        score_bar = " > ".join(f"t{idx+1}:{sc:.3f}" for idx, sc in top5)
        _best_param_str = "\n".join(
            f"  {k} = {round(v, 4) if isinstance(v, float) else v}"
            for k, v in best_params.items()
        )
        await notify_fn(
            f"[MINI-OPT-PROG] outer {cycle_idx} 최적화 완료 — {len(trial_scores)}개 trial\n"
            f"점수 순위: {score_bar}\n"
            f"★ 베스트 파라미터 (trial {best_idx + 1}, score={best_score:.3f}):\n"
            f"{_best_param_str}\n"
            f"→ rl_policy.json 업데이트 예정"
        )

    # ── Build subtrial data for logging ──────────────────────────────────────
    subtrial_data: list[dict] = []
    for i, (score, params) in enumerate(zip(trial_scores, all_trial_params)):
        meta = trial_meta_by_idx.get(i, {})
        subtrial_data.append({
            "trial_idx": i,
            "score": round(score, 4),
            "det": round(float(meta.get("det", 0.0)), 3),
            "llm": round(float(meta.get("llm", 0.0)), 3),
            "repetition_penalty": round(float(meta.get("repetition_penalty", 0.0)), 3),
            "params": dict(params),
        })

    total_trials = len(trial_scores)
    if notify_fn:
        sorted_scores = sorted(enumerate(trial_scores), key=lambda x: x[1], reverse=True)
        ranking = " | ".join(f"t{idx}:{sc:.2f}" for idx, sc in sorted_scores[:5])
        await notify_fn(
            f"[MINI-OPT] outer {cycle_idx} 전체 완료 | {total_trials} trials\n"
            f"best={best_score:.2f} (trial {best_idx}) | top5: {ranking}\n"
            f"누적 study trials: {n_past + total_trials}"
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
