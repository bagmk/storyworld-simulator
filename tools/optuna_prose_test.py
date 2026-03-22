#!/usr/bin/env python3
"""
tools/optuna_prose_test.py — Standalone Optuna search for ProseGenerator parameters.

Goal: find the best `runtime_policy` combination for literary prose quality
      WITHOUT running a full simulation. Uses precomputed scenes.json as input.

Usage:
    python tools/optuna_prose_test.py [--trials 20] [--scenes-file PATH]

Output:
    output/optuna_prose/study_results.json   — all trial results + scores
    output/optuna_prose/best_params.json     — best params found

Design:
    - Imports ProseGenerator directly (no subprocess) so runtime_policy is passed cleanly
    - Loads precomputed scenes.json and trims to SCENE_LIMIT for speed
    - Scores each chapter with a single LLM call (3 criteria → average)
    - Modular: scoring fn and param space are easy to swap or extend

TODO (future pipeline integration):
    - Wire best_params.json back into rl_policy.json or daily_pipeline overrides
    - Add distiller param space in a second study (deferred for now)
    - Increase SCENE_LIMIT once latency budget is confirmed
    - Replace inline scorer with quality_reviewer.py when it becomes standalone
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
SCENE_FILE     = "output/20260307/007/ep01_conference_shadow_scenes.json"
EPISODE_ID     = "ep01_conference_shadow"
EPISODE_CONFIG = "config/episodes/ep01_academic_presentation.yaml"
PROTAGONIST    = "kim_sumin"
PROTAGONIST_NAME = "Kim Sumin"

SCENE_LIMIT    = 3          # trim input to this many scenes per trial
TARGET_WORDS   = 1200       # shorter target → faster LLM calls
PROSE_MODEL    = "gpt-4.1-mini"
BASE_MODEL     = "gpt-4o-mini"
REVIEWER_MODEL = "gpt-4o-mini"
_REPETITION_STOPWORDS = {
    "그리고", "그러나", "하지만", "그래서", "정말", "아주", "매우", "조금",
    "것", "수", "더", "또", "그", "이", "저", "때문", "정도",
}
PROSE_STYLE    = "third_person_close"
BUDGET_USD     = 2.0        # per-trial budget cap

OUTPUT_DIR     = "output/optuna_prose"
DB_PATH        = "data/simulation.db"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [optuna_prose] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optuna_prose")


# ── Parameter space ──────────────────────────────────────────────────────────
def _suggest_runtime_policy(trial) -> dict:
    """
    Define the Optuna search space for ProseGenerator.runtime_policy.

    Keys match what ProseGenerator reads from self.runtime_policy.
    Add more keys here as you discover ones worth tuning.

    TODO: verify each key's effect by checking ProseGenerator source
          before widening the ranges.
    """
    min_sent = trial.suggest_int("prose_paragraph_min_sentences", 1, 3)
    max_sent = trial.suggest_int("prose_paragraph_max_sentences", min_sent + 1, min_sent + 4)

    return {
        "prose_scene_temperature": trial.suggest_float(
            "prose_scene_temperature", 0.5, 1.0
        ),
        "prose_paragraph_min_sentences": min_sent,
        "prose_paragraph_max_sentences": max_sent,
        "prose_scene_readability_temperature": trial.suggest_float(
            "prose_scene_readability_temperature", 0.2, 0.7
        ),
        "prose_transition_temperature": trial.suggest_float(
            "prose_transition_temperature", 0.4, 1.0
        ),
        "prose_enable_term_gloss": trial.suggest_categorical(
            "prose_enable_term_gloss", [True, False]
        ),
        "prefer_concrete_offer_detail": trial.suggest_categorical(
            "prefer_concrete_offer_detail", [0, 1]
        ),
    }


# ── Scene loader ──────────────────────────────────────────────────────────────
def _load_scenes(scene_file: str, limit: int):
    """
    Load scenes.json and return the first `limit` DistilledScene objects.
    Mirrors the logic in generate_chapter._load_precomputed_scenes.
    """
    from src.novel_writer.scene_distiller import DistilledScene

    raw_list = json.loads(Path(scene_file).read_text(encoding="utf-8"))
    raw_list = raw_list[:limit]

    def _coerce_int(v, default=0) -> int:
        if isinstance(v, bool):
            return default
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, (list, tuple)):
            return _coerce_int(v[0] if v else default, default)
        raw = str(v or "").strip()
        raw = raw.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
        m = re.search(r"(?<!\d)(\d{1,5})(?!\d)", raw)
        return int(m.group(1)) if m else default

    def _coerce_str_list(v) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [p.strip() for p in re.split(r"[,/|]\s*|\n+", v) if p.strip()]
        return []

    def _norm_range(raw) -> tuple[int, int]:
        if isinstance(raw, str):
            vals = re.findall(r"\d{1,5}", raw.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")))[:2]
        else:
            vals = list(raw)[:2] if isinstance(raw, (list, tuple)) else [raw]
        if len(vals) < 2:
            fill = vals[0] if vals else 0
            vals = (vals + [fill, fill])[:2]
        s, e = _coerce_int(vals[0]), _coerce_int(vals[1])
        return (s, e) if s <= e else (e, s)

    scenes = []
    for item in raw_list:
        scenes.append(DistilledScene(
            scene_number=_coerce_int(item.get("scene_number", 0), default=len(scenes) + 1),
            title=str(item.get("title", "")),
            turn_range=_norm_range(item.get("turn_range", [0, 0])),
            location=str(item.get("location", "")),
            characters_present=_coerce_str_list(item.get("characters_present", [])),
            key_dialogue=item.get("key_dialogue", []) if isinstance(item.get("key_dialogue"), list) else [],
            key_actions=_coerce_str_list(item.get("key_actions", [])),
            discoveries=_coerce_str_list(item.get("discoveries", [])),
            emotional_arc=str(item.get("emotional_arc", "")),
            beat_references=_coerce_str_list(item.get("beat_references", [])),
            narrative_summary=str(item.get("narrative_summary", "")),
            pacing=str(item.get("pacing", "")),
            raw_turn_count=max(1, _coerce_int(item.get("raw_turn_count", 0), default=1)),
        ))

    logger.info("Loaded %d/%d scenes from %s", len(scenes), limit, scene_file)
    return scenes


# ── Deterministic scorer (QualityAnalyzer) ───────────────────────────────────
def _score_deterministic(chapter_text: str) -> tuple[float, dict]:
    """
    Run QualityAnalyzer (no LLM) on the chapter.
    Returns (overall_score_0_to_10, detail_dict).

    Metrics (all 0-1, weighted):
      sentence_complexity, paragraph_density, repetition_patterns,
      abstract_concrete_ratio, dialogue_ratio, scene_progression,
      information_novelty, subordinate_clause_depth,
      opening_line_quality, pov_consistency, timeline_coherence
    """
    sys.path.insert(0, str(_ROOT / "tools"))
    from quality_analyzer import QualityAnalyzer

    qa = QualityAnalyzer(chapter_text, episode_name=EPISODE_ID)
    results = qa.analyze()

    overall_01 = results.get("overall_score", 0.0)
    overall_10 = round(overall_01 * 10, 2)

    detail = {
        "sentence_complexity":      results.get("sentence_complexity", {}).get("score", 0),
        "paragraph_density":        results.get("paragraph_density", {}).get("score", 0),
        "repetition_patterns":      results.get("repetition_patterns", {}).get("score", 0),
        "abstract_concrete_ratio":  results.get("abstract_concrete_ratio", {}).get("score", 0),
        "dialogue_ratio":           results.get("dialogue_ratio", {}).get("score", 0),
        "scene_progression":        results.get("scene_progression", {}).get("score", 0),
        "information_novelty":      results.get("information_novelty", {}).get("score", 0),
        "subordinate_clause_depth": results.get("subordinate_clause_depth", {}).get("score", 0),
        "opening_line_quality":     results.get("opening_line_quality", {}).get("score", 0),
        "pov_consistency":          results.get("pov_consistency", {}).get("score", 0),
        "timeline_coherence":       results.get("timeline_coherence", {}).get("score", 0),
    }
    return overall_10, detail


# ── LLM scorer ────────────────────────────────────────────────────────────────
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

def _score_llm(chapter_text: str) -> tuple[float, dict]:
    """
    Score with LLM using 8 style-focused criteria (0–10 each).
    Returns (avg_score, detail_dict). Falls back to (0.0, {}) on error.
    """
    from openai import OpenAI
    client = OpenAI()

    # head + tail excerpt to cover opening and climax
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
        if not raw:
            logger.warning("LLM scorer: empty response")
            return 0.0, {}
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        scores = json.loads(raw)
        numeric = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        avg = sum(numeric.values()) / len(numeric) if numeric else 0.0
        return round(avg, 2), numeric
    except Exception as exc:
        logger.warning("LLM scorer error: %s", exc)
        return 0.0, {}


def _score_chapter(chapter_text: str) -> float:
    """
    Combined score = 40% deterministic (QualityAnalyzer) + 60% LLM (8 style criteria).
    Returns a float in [0, 10].
    """
    det_score, det_detail = _score_deterministic(chapter_text)
    llm_score, llm_detail = _score_llm(chapter_text)
    repetition_penalty = _repetition_penalty(chapter_text)
    combined = round(max(0.0, 0.4 * det_score + 0.6 * llm_score - repetition_penalty), 3)

    # Log all dimensions
    logger.info(
        "  [DET] sent_complex=%.2f par_density=%.2f repetition=%.2f abstract=%.2f "
        "dialogue=%.2f scene_prog=%.2f info_novel=%.2f sub_depth=%.2f "
        "opening=%.2f pov=%.2f timeline=%.2f → det_avg=%.2f",
        det_detail.get("sentence_complexity", 0),
        det_detail.get("paragraph_density", 0),
        det_detail.get("repetition_patterns", 0),
        det_detail.get("abstract_concrete_ratio", 0),
        det_detail.get("dialogue_ratio", 0),
        det_detail.get("scene_progression", 0),
        det_detail.get("information_novelty", 0),
        det_detail.get("subordinate_clause_depth", 0),
        det_detail.get("opening_line_quality", 0),
        det_detail.get("pov_consistency", 0),
        det_detail.get("timeline_coherence", 0),
        det_score,
    )
    logger.info(
        "  [LLM] literary=%.1f tension=%.1f readability=%.1f "
        "sent_diversity=%.1f par_rhythm=%.1f vividness=%.1f "
        "dialogue=%.1f pacing_bal=%.1f → llm_avg=%.2f",
        llm_detail.get("literary_quality", 0),
        llm_detail.get("emotional_tension", 0),
        llm_detail.get("readability", 0),
        llm_detail.get("sentence_diversity", 0),
        llm_detail.get("paragraph_rhythm", 0),
        llm_detail.get("prose_vividness", 0),
        llm_detail.get("dialogue_effectiveness", 0),
        llm_detail.get("pacing_tension_balance", 0),
        llm_score,
    )
    logger.info(
        "  [COMBINED] %.3f  (det=%.2f × 0.4 + llm=%.2f × 0.6 - rep=%.3f)",
        combined, det_score, llm_score, repetition_penalty,
    )

    return combined


def _repetition_penalty(chapter_text: str) -> float:
    text = str(chapter_text or "").strip()
    if not text:
        return 0.0

    normalized = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text.lower())
    tokens = [tok for tok in normalized.split() if len(tok) >= 2 and tok not in _REPETITION_STOPWORDS]
    if len(tokens) < 80:
        return 0.0

    token_counts = Counter(tokens)
    repeated_token_ratio = sum(1 for count in token_counts.values() if count >= 6) / max(1, len(token_counts))
    bigram_counts = Counter(zip(tokens, tokens[1:]))
    repeated_bigram_ratio = sum(1 for count in bigram_counts.values() if count >= 3) / max(1, len(bigram_counts))

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
        if overlap >= 0.72:
            local_repeat_hits += 1

    return round(
        min(
            1.5,
            repeated_token_ratio * 2.0 + repeated_bigram_ratio * 2.5 + (local_repeat_hits / max(1, len(sentences))) * 3.0,
        ),
        3,
    )


# ── Optuna objective ──────────────────────────────────────────────────────────
def _build_objective(scenes, episode_config: dict, output_dir: str):
    """Return a closure for optuna.Study.optimize()."""

    from src.novel_writer.llm_client import LLMClient
    from src.novel_writer.prose_generator import ProseGenerator
    from src.novel_writer.scene_distiller import SceneDistiller

    def objective(trial) -> float:
        runtime_policy = _suggest_runtime_policy(trial)
        logger.info("Trial %d | params: %s", trial.number, runtime_policy)

        trial_dir = Path(output_dir) / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        llm = LLMClient(
            model=BASE_MODEL,
            premium_model=PROSE_MODEL,
            budget_usd=BUDGET_USD,
        )

        prose_gen = ProseGenerator(
            llm=llm,
            episode_config=episode_config,
            output_dir=str(trial_dir),
            runtime_policy=runtime_policy,
        )

        try:
            chapter_path = prose_gen.generate_chapter(
                scenes=scenes,
                protagonist_name=PROTAGONIST_NAME,
                style=PROSE_STYLE,
                target_words=TARGET_WORDS,
            )
        except Exception as exc:
            logger.warning("Trial %d: prose generation failed: %s", trial.number, exc)
            return 0.0

        chapter_text = Path(chapter_path).read_text(encoding="utf-8")
        score = _score_chapter(chapter_text)

        trial.set_user_attr("chapter_path", chapter_path)
        trial.set_user_attr("score", score)
        trial.set_user_attr("word_count", len(chapter_text.split()))
        return score

    return objective


# ── Results saver ─────────────────────────────────────────────────────────────
def _save_results(study, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_trials = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
            "state": str(t.state),
        }
        for t in study.trials
    ]
    (out / "study_results.json").write_text(
        json.dumps(all_trials, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if study.best_trial:
        best = {
            "value": study.best_value,
            "params": study.best_params,
            "trial_number": study.best_trial.number,
        }
        (out / "best_params.json").write_text(
            json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Best score: %.4f  params: %s", study.best_value, study.best_params)

    logger.info("Results saved → %s", out)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna prose parameter search — standalone test"
    )
    p.add_argument("--trials", type=int, default=10,
                   help="Number of Optuna trials (default: 10)")
    p.add_argument("--scenes-file", default=SCENE_FILE,
                   help=f"Path to precomputed scenes.json (default: {SCENE_FILE})")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help=f"Output directory (default: {OUTPUT_DIR})")
    p.add_argument("--scene-limit", type=int, default=SCENE_LIMIT,
                   help=f"Max scenes per trial (default: {SCENE_LIMIT})")
    return p.parse_args()


def main() -> None:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna not installed — run: pip install optuna")
        sys.exit(1)

    # Env must be loaded before importing project modules that call OpenAI
    from src.novel_writer.env_loader import load_project_env
    load_project_env()

    args = parse_args()

    scene_file = Path(args.scenes_file)
    if not scene_file.exists():
        logger.error("Scenes file not found: %s", scene_file)
        logger.error("Run the daily pipeline first to produce a scenes.json.")
        sys.exit(1)

    from src.novel_writer.config_loader import load_episode
    episode_config = load_episode(EPISODE_CONFIG)

    scenes = _load_scenes(str(scene_file), args.scene_limit)

    study = optuna.create_study(
        study_name="optuna_prose_test",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = _build_objective(
        scenes=scenes,
        episode_config=episode_config,
        output_dir=args.output_dir,
    )

    logger.info(
        "Starting: %d trials × %d scenes × %d target_words (model=%s)",
        args.trials, args.scene_limit, TARGET_WORDS, PROSE_MODEL,
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    _save_results(study, args.output_dir)


if __name__ == "__main__":
    main()
