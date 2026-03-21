#!/usr/bin/env python3
"""
tools/optuna_multi_study.py — 3개 병렬 Optuna 스터디
  1. Distiller  — SceneDistiller 파라미터 최적화
  2. Orchestrator — SimulationOrchestrator 턴 품질 최적화
  3. Polisher   — ChapterPolisher 파라미터 최적화

Usage:
    python tools/optuna_multi_study.py [--trials 10]

Output:
    output/optuna_distiller/best_params.json
    output/optuna_orchestrator/best_params.json
    output/optuna_polisher/best_params.json
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import re
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 공통 상수 ─────────────────────────────────────────────────────────────────
EPISODE_ID      = "ep01_conference_shadow"
EPISODE_CONFIG  = "config/episodes/ep01_academic_presentation.yaml"
CHARACTERS      = "config/characters.yaml"
PROTAGONIST     = "kim_sumin"
PROTAGONIST_NAME = "Kim Sumin"
DB_PATH         = "data/simulation.db"
BASE_MODEL      = "gpt-4o-mini"
PREMIUM_MODEL   = "gpt-4.1-mini"
REVIEWER_MODEL  = "gpt-4o-mini"
BUDGET_USD      = 2.0

BEST_CHAPTER    = "output/optuna_prose/trial_0001/ep01_conference_shadow_chapter.txt"
SCENE_FILE      = "output/20260307/007/ep01_conference_shadow_scenes.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 공통: LLM 스코어러
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_score(text: str, criteria: dict[str, str], label: str) -> float:
    """단일 LLM 콜로 여러 기준 채점. 0-10 평균 반환."""
    from openai import OpenAI
    client = OpenAI()

    criteria_lines = "\n".join(f"  {k}: {v}" for k, v in criteria.items())
    keys_json = "{" + ", ".join(f'"{k}": X' for k in criteria) + "}"

    excerpt = text[:3000]
    prompt = (
        f"You are evaluating a Korean thriller novel pipeline component ({label}).\n"
        "Rate each criterion 0–10 based on the text below.\n"
        f"Criteria:\n{criteria_lines}\n\n"
        f"Return ONLY valid JSON: {keys_json}\n\n"
        f"--- TEXT ---\n{excerpt}\n--- END ---"
    )
    system = "Return ONLY valid JSON. No markdown, no explanation."

    try:
        resp = client.chat.completions.create(
            model=REVIEWER_MODEL,
            messages=[
                {"role": "system", "content": system},
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
        return round(sum(numeric.values()) / len(numeric), 3) if numeric else 0.0
    except Exception as exc:
        logging.getLogger(label).warning("LLM scorer error: %s", exc)
        return 0.0


def _save_study(study, out_dir: str, label: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
    ]
    (out / "study_results.json").write_text(
        json.dumps(trials, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if study.best_trial:
        best = {"value": study.best_value, "params": study.best_params,
                "trial_number": study.best_trial.number}
        (out / "best_params.json").write_text(
            json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logging.getLogger(label).info(
            "Best: %.4f  params: %s", study.best_value, study.best_params
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STUDY 1: Distiller
# ═══════════════════════════════════════════════════════════════════════════════

def _distiller_param_space(trial) -> dict:
    return {
        "distiller_temperature":         trial.suggest_float("distiller_temperature", 0.1, 0.6),
        "distiller_max_tokens":          trial.suggest_int("distiller_max_tokens", 2000, 6000, step=500),
        "target_scenes":                 trial.suggest_int("target_scenes", 3, 8),
        "prompt_char_limit":             trial.suggest_int("prompt_char_limit", 8000, 20000, step=2000),
        # soft params — 0.0=off, 1.0=max
        "confrontation_merge_strength":  trial.suggest_float("confrontation_merge_strength", 0.0, 1.0),
        "dialogue_compaction_strength":  trial.suggest_float("dialogue_compaction_strength", 0.0, 1.0),
        "role_cue_strength":             trial.suggest_float("role_cue_strength", 0.0, 1.0),
    }


def _distiller_score(scenes: list) -> float:
    if not scenes:
        return 0.0
    # serialise scenes to text for scoring
    text = json.dumps([s.to_dict() for s in scenes], ensure_ascii=False)
    criteria = {
        "beat_coverage":   "All key story beats and clues are represented across the scenes (0=missing, 10=full)",
        "scene_coherence": "Each scene has a clear location, characters, and narrative arc (no fragmented/empty scenes)",
        "dialogue_quality": "Retained dialogue is essential and non-redundant",
        "scene_variety":   "Scenes differ meaningfully in pacing, emotion, and content (not all the same)",
        "compactness":     "Scenes are tight — no verbose padding or repeated information",
    }
    return _llm_score(text[:4000], criteria, "distiller")


def run_distiller_study(n_trials: int) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = logging.getLogger("distiller")

    from src.novel_writer.env_loader import load_project_env
    load_project_env()
    from src.novel_writer.config_loader import load_episode
    from src.novel_writer.llm_client import LLMClient
    from src.novel_writer.scene_distiller import SceneDistiller
    from src.novel_writer import database as db

    episode_config = load_episode(EPISODE_CONFIG)
    db.DB_PATH = DB_PATH
    db.init_db()

    def objective(trial) -> float:
        params = _distiller_param_space(trial)
        log.info("Trial %d | %s", trial.number, params)

        llm = LLMClient(
            model=BASE_MODEL,
            premium_model=PREMIUM_MODEL,
            budget_usd=BUDGET_USD,
        )
        llm.temperature    = params["distiller_temperature"]
        llm.max_tokens     = params["distiller_max_tokens"]

        runtime_policy = {
            "distiller_temperature":        params["distiller_temperature"],
            "distiller_max_tokens":         params["distiller_max_tokens"],
            "prompt_char_limit":            params["prompt_char_limit"],
            "confrontation_merge_strength": params["confrontation_merge_strength"],
            "dialogue_compaction_strength": params["dialogue_compaction_strength"],
            "role_cue_strength":            params["role_cue_strength"],
        }

        distiller = SceneDistiller(
            llm=llm,
            episode_config=episode_config,
            runtime_policy=runtime_policy,
        )
        # Patch prompt_char_limit into the request builder
        distiller._optuna_prompt_char_limit = params["prompt_char_limit"]

        try:
            scenes = distiller.distill(
                episode_id=EPISODE_ID,
                protagonist_id=PROTAGONIST,
                target_scenes=params["target_scenes"],
            )
        except Exception as exc:
            log.warning("Trial %d distill failed: %s", trial.number, exc)
            return 0.0

        score = _distiller_score(scenes)
        log.info("Trial %d | scenes=%d score=%.3f", trial.number, len(scenes), score)
        trial.set_user_attr("n_scenes", len(scenes))
        return score

    study = optuna.create_study(
        study_name="optuna_distiller",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=11),
    )
    study.optimize(objective, n_trials=n_trials)
    _save_study(study, "output/optuna_distiller", "distiller")
    log.info("Distiller study done. Best: %.4f", study.best_value if study.best_trial else 0)


# ═══════════════════════════════════════════════════════════════════════════════
# STUDY 2: Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def _orchestrator_param_space(trial) -> dict:
    return {
        # word caps per turn section
        "action_cap":                        trial.suggest_int("action_cap", 30, 90, step=10),
        "dialogue_cap":                      trial.suggest_int("dialogue_cap", 20, 70, step=10),
        "inner_cap":                         trial.suggest_int("inner_cap", 10, 50, step=10),
        # structural caps
        "sentence_chars_max":                trial.suggest_int("sentence_chars_max", 60, 120, step=10),
        "max_sentences_per_paragraph":       trial.suggest_int("max_sentences_per_paragraph", 2, 6),
        "max_jargon_terms_per_paragraph":    trial.suggest_int("max_jargon_terms_per_paragraph", 1, 4),
        # boolean style flags
        "needs_role_cues":                   trial.suggest_categorical("needs_role_cues", [True, False]),
        "prefers_dialogue_compaction":       trial.suggest_categorical("prefers_dialogue_compaction", [True, False]),
        "prefers_analytical_wording_reduction": trial.suggest_categorical("prefers_analytical_wording_reduction", [True, False]),
        # repetition guard sensitivity (Jaccard threshold: lower = more sensitive)
        "repetition_jaccard_threshold":      trial.suggest_float("repetition_jaccard_threshold", 0.4, 0.85),
    }


def _build_reader_feedback_from_orchestrator_params(params: dict) -> dict:
    """Convert Optuna params → reader_feedback dict the orchestrator can consume."""
    boring = []
    if params["needs_role_cues"]:
        boring.append("role cues needed")
    if params["prefers_dialogue_compaction"]:
        boring.append("dialogue too long")
    if params["prefers_analytical_wording_reduction"]:
        boring.append("too analytical / abstract")
    return {
        "style_constraints": {
            "action_cap":                        params["action_cap"],
            "dialogue_cap":                      params["dialogue_cap"],
            "inner_cap":                         params["inner_cap"],
            "sentence_chars_max":                params["sentence_chars_max"],
            "max_sentences_per_paragraph":       params["max_sentences_per_paragraph"],
            "max_jargon_terms_per_paragraph":    params["max_jargon_terms_per_paragraph"],
        },
        "what_felt_boring_or_hard": boring,
    }


def _orchestrator_score(turns: list[dict]) -> float:
    """Score simulation turns with fine-grained per-turn criteria for Optuna signal."""
    from openai import OpenAI
    if not turns:
        return 0.0

    client = OpenAI()
    # Score all turns in one call with detailed continuous criteria
    text = "\n\n".join(
        f"[Turn {t.get('turn', i+1)} — {t.get('speaker_id', '?')}]\n{t.get('content', '')}"
        for i, t in enumerate(turns[:10])
    )
    prompt = (
        "You are evaluating a Korean thriller simulation. Score each of the 5 criteria "
        "on a CONTINUOUS scale 0.0–10.0 (decimals encouraged — avoid round numbers).\n\n"
        "Criteria:\n"
        "  word_economy: Turns are tight — no padding, no restating, no filler phrases (0=very verbose, 10=lean)\n"
        "  repetition_absence: Characters do NOT repeat the same ideas or phrasing across turns (0=repetitive, 10=fresh)\n"
        "  voice_distinctiveness: Characters sound clearly different from each other (0=same voice, 10=distinct)\n"
        "  tension_arc: Later turns feel more tense or complex than earlier ones (0=flat, 10=clear escalation)\n"
        "  specificity: Actions and dialogue reference specific details, not vague generalities (0=vague, 10=precise)\n\n"
        "Return ONLY valid JSON: "
        "{\"word_economy\": X, \"repetition_absence\": X, \"voice_distinctiveness\": X, \"tension_arc\": X, \"specificity\": X}\n\n"
        f"--- TURNS ---\n{text[:3500]}\n--- END ---"
    )
    try:
        resp = client.chat.completions.create(
            model=REVIEWER_MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON with float values. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        scores = json.loads(raw)
        numeric = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        return round(sum(numeric.values()) / len(numeric), 3) if numeric else 0.0
    except Exception as exc:
        logging.getLogger("orchestrator").warning("LLM scorer error: %s", exc)
        return 0.0


def run_orchestrator_study(n_trials: int) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = logging.getLogger("orchestrator")

    from src.novel_writer.env_loader import load_project_env
    load_project_env()
    from src.novel_writer.config_loader import (
        load_episode, load_characters, load_world_facts,
        load_storyline, build_world_state, build_clue_manager,
    )
    from src.novel_writer.llm_client import LLMClient
    from src.novel_writer.director import DirectorAI
    from src.novel_writer.orchestrator import SimulationOrchestrator, TURN_LOCAL_REPEAT_JACCARD
    from src.novel_writer import database as db
    import src.novel_writer.orchestrator as orch_module

    episode_config = load_episode(EPISODE_CONFIG)
    # 10 turns — enough for policy effects to accumulate and score to differentiate
    mini_config = dict(episode_config)
    mini_config["max_turns"] = 10

    def objective(trial) -> float:
        params = _orchestrator_param_space(trial)
        log.info("Trial %d | %s", trial.number, params)

        # Fresh temp DB per trial to avoid cross-contamination
        tmp_db = tempfile.mktemp(suffix=f"_orch_trial{trial.number}.db")
        db.DB_PATH = tmp_db
        db.init_db()

        reader_feedback = _build_reader_feedback_from_orchestrator_params(params)

        # Patch repetition Jaccard threshold for this trial
        original_jaccard = orch_module.TURN_LOCAL_REPEAT_JACCARD
        orch_module.TURN_LOCAL_REPEAT_JACCARD = params["repetition_jaccard_threshold"]

        try:
            agents = load_characters(CHARACTERS)
            world_facts = load_world_facts("config/world_facts.yaml")
            storyline = load_storyline("config/storyline.yaml")
            world = build_world_state(mini_config, world_facts, agents)
            clue_mgr = build_clue_manager(mini_config, world_facts)
            world.clue_manager = clue_mgr

            episode_id_test = f"{EPISODE_ID}_optuna_t{trial.number}"
            llm = LLMClient(model=BASE_MODEL, premium_model=PREMIUM_MODEL, budget_usd=BUDGET_USD)
            director = DirectorAI(
                episode_config=mini_config,
                world_facts=world_facts,
                clue_manager=clue_mgr,
                llm=llm,
                storyline=storyline,
            )
            orchestrator = SimulationOrchestrator(
                agents=agents,
                director=director,
                world=world,
                llm=llm,
                episode_id=episode_id_test,
                episode_config=mini_config,
                reader_feedback=reader_feedback,
            )
            turns = orchestrator.run_episode()
        except Exception as exc:
            log.warning("Trial %d orchestrator failed: %s", trial.number, exc)
            return 0.0
        finally:
            orch_module.TURN_LOCAL_REPEAT_JACCARD = original_jaccard
            try:
                Path(tmp_db).unlink(missing_ok=True)
            except OSError:
                pass

        score = _orchestrator_score(turns)
        log.info("Trial %d | turns=%d score=%.3f", trial.number, len(turns), score)
        trial.set_user_attr("n_turns", len(turns))
        return score

    study = optuna.create_study(
        study_name="optuna_orchestrator",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=22),
    )
    study.optimize(objective, n_trials=n_trials)
    _save_study(study, "output/optuna_orchestrator", "orchestrator")
    log.info("Orchestrator study done. Best: %.4f", study.best_value if study.best_trial else 0)


# ═══════════════════════════════════════════════════════════════════════════════
# STUDY 3: Polisher
# ═══════════════════════════════════════════════════════════════════════════════

def _polisher_param_space(trial) -> dict:
    return {
        # Prose polish LLM temperature
        "prose_polish_temperature":        trial.suggest_float("prose_polish_temperature", 0.2, 0.8),
        # Anchor fix temperature
        "prose_anchor_fix_temperature":    trial.suggest_float("prose_anchor_fix_temperature", 0.2, 0.6),
        # Clue injection timing: 0=early (first pass), 1=late (anchor-fix pass)
        "clue_injection_timing":           trial.suggest_categorical("clue_injection_timing", [0, 1]),
        # Fallback cast size (how many agents director keeps when cast is trimmed)
        "director_fallback_cast_size":     trial.suggest_int("director_fallback_cast_size", 1, 4),
        # Scene closure: how aggressively to end scenes (0=gentle, 1=hard cut)
        "scene_closure_aggressiveness":    trial.suggest_float("scene_closure_aggressiveness", 0.0, 1.0),
        # Pressure peak hold: keep tension at peak vs release after peak
        "hold_pressure_peak":              trial.suggest_categorical("hold_pressure_peak", [0, 1]),
        # Repeated concern loop reduction: 0=off, 1=on
        "repeated_concern_loop_reduction": trial.suggest_categorical("repeated_concern_loop_reduction", [0, 1]),
    }


def _polisher_deterministic_score(text: str) -> float:
    """Run QualityAnalyzer on polished text. Returns 0-10."""
    sys.path.insert(0, str(_ROOT / "tools"))
    from quality_analyzer import QualityAnalyzer
    qa = QualityAnalyzer(text, episode_name=EPISODE_ID)
    return round(qa.analyze().get("overall_score", 0.0) * 10, 3)


def _polisher_comparative_score(source: str, polished: str) -> float:
    """Compare before vs after polishing for differential signal (not absolute quality).
    Asks: how much did the polish *improve* each aspect? Returns 0-10."""
    from openai import OpenAI
    client = OpenAI()
    prompt = (
        "You are a Korean literary editor comparing two versions of the same chapter.\n"
        "Rate how much the POLISHED version IMPROVED over the ORIGINAL on each criterion "
        "(0.0 = no improvement or got worse, 5.0 = neutral/same, 10.0 = dramatic improvement). "
        "Use decimals — avoid round numbers.\n\n"
        "Criteria:\n"
        "  repetition_reduction: Were repeated phrases/ideas cleaned up? (0=more repetition, 10=greatly reduced)\n"
        "  sentence_variety: Did sentence lengths and structures become more varied? (0=more monotone, 10=more varied)\n"
        "  flow_smoothness: Do transitions between sentences and paragraphs feel smoother? (0=worse, 10=much smoother)\n"
        "  word_precision: Were vague words replaced with precise, concrete language? (0=vaguer, 10=much more precise)\n\n"
        "Return ONLY valid JSON: "
        "{\"repetition_reduction\": X, \"sentence_variety\": X, \"flow_smoothness\": X, \"word_precision\": X}\n\n"
        f"--- ORIGINAL (first 1200 chars) ---\n{source[:1200]}\n\n"
        f"--- POLISHED (first 1200 chars) ---\n{polished[:1200]}\n--- END ---"
    )
    try:
        resp = client.chat.completions.create(
            model=REVIEWER_MODEL,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON with float values. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        scores = json.loads(raw)
        numeric = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        return round(sum(numeric.values()) / len(numeric), 3) if numeric else 0.0
    except Exception as exc:
        logging.getLogger("polisher").warning("Comparative scorer error: %s", exc)
        return 0.0


def run_polisher_study(n_trials: int) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = logging.getLogger("polisher")

    from src.novel_writer.env_loader import load_project_env
    load_project_env()
    from src.novel_writer.config_loader import load_episode
    from src.novel_writer.llm_client import LLMClient
    from src.novel_writer.prose_generator import ProseGenerator

    # Load best chapter from prose study as input
    chapter_path = Path(BEST_CHAPTER)
    if not chapter_path.exists():
        # Fallback: find any trial chapter
        candidates = sorted(Path("output/optuna_prose").glob("trial_*/ep01_*.txt"))
        if not candidates:
            log.error("No best chapter found — run optuna_prose_test.py first")
            return
        chapter_path = candidates[0]

    source_text = chapter_path.read_text(encoding="utf-8")
    log.info("Polisher input: %s (%d words)", chapter_path, len(source_text.split()))

    episode_config = load_episode(EPISODE_CONFIG)

    def objective(trial) -> float:
        params = _polisher_param_space(trial)
        log.info("Trial %d | %s", trial.number, params)

        trial_dir = Path("output/optuna_polisher") / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        runtime_policy = {
            "prose_polish_temperature":        params["prose_polish_temperature"],
            "prose_anchor_fix_temperature":    params["prose_anchor_fix_temperature"],
            "clue_injection_timing":           params["clue_injection_timing"],
            "director_fallback_cast_size":     params["director_fallback_cast_size"],
            "scene_closure_aggressiveness":    params["scene_closure_aggressiveness"],
            "hold_pressure_peak":              params["hold_pressure_peak"],
            "repeated_concern_loop_reduction": params["repeated_concern_loop_reduction"],
            # Keep best prose params
            "prose_scene_temperature":         0.985,
            "prose_paragraph_min_sentences":   3,
            "prose_paragraph_max_sentences":   4,
            "prose_enable_term_gloss":         False,
        }

        llm = LLMClient(model=BASE_MODEL, premium_model=PREMIUM_MODEL, budget_usd=BUDGET_USD)
        prose_gen = ProseGenerator(
            llm=llm,
            episode_config=episode_config,
            output_dir=str(trial_dir),
            runtime_policy=runtime_policy,
        )

        try:
            polished = prose_gen.chapter_polisher.polish_chapter(
                text=source_text,
                target_words=1200,
                style="third_person_close",
                protagonist_name=PROTAGONIST_NAME,
                chapter_anchors=None,
                prose_adapter=prose_gen,
            )
        except Exception as exc:
            log.warning("Trial %d polisher failed: %s", trial.number, exc)
            return 0.0

        out_path = trial_dir / "polished_chapter.txt"
        out_path.write_text(polished, encoding="utf-8")

        det_score  = _polisher_deterministic_score(polished)
        llm_score  = _polisher_comparative_score(source_text, polished)

        combined = round(0.4 * det_score + 0.6 * llm_score, 3)
        log.info("Trial %d | det=%.2f llm=%.2f combined=%.3f", trial.number, det_score, llm_score, combined)
        trial.set_user_attr("det_score", det_score)
        trial.set_user_attr("llm_score", llm_score)
        return combined

    study = optuna.create_study(
        study_name="optuna_polisher",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=33),
    )
    study.optimize(objective, n_trials=n_trials)
    _save_study(study, "output/optuna_polisher", "polisher")
    log.info("Polisher study done. Best: %.4f", study.best_value if study.best_trial else 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 진행 모니터
# ═══════════════════════════════════════════════════════════════════════════════

def _monitor(n_trials: int, selected: set | None = None) -> None:
    """각 스터디의 best_params.json을 주기적으로 체크해 진행 상황 출력."""
    import time
    all_studies = {
        "distiller":    "output/optuna_distiller/study_results.json",
        "orchestrator": "output/optuna_orchestrator/study_results.json",
        "polisher":     "output/optuna_polisher/study_results.json",
    }
    studies = {k: v for k, v in all_studies.items() if selected is None or k in selected}
    log = logging.getLogger("monitor")
    seen = {k: -1 for k in studies}

    while True:
        all_done = True
        for name, path in studies.items():
            p = Path(path)
            if not p.exists():
                all_done = False
                continue
            try:
                trials = json.loads(p.read_text())
                completed = [t for t in trials if t["state"] == "TrialState.COMPLETE"]
                if len(completed) > seen[name]:
                    seen[name] = len(completed)
                    if completed:
                        best_v = max(t["value"] for t in completed if t["value"] is not None)
                        latest = completed[-1]
                        log.info(
                            "[%s] %d/%d | latest=%.3f | best=%.3f | params=%s",
                            name, len(completed), n_trials,
                            latest["value"] or 0, best_v, latest["params"]
                        )
                if len(completed) < n_trials:
                    all_done = False
            except Exception:
                all_done = False

        if all_done:
            log.info("모든 스터디 완료!")
            break
        time.sleep(15)


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="3개 병렬 Optuna 스터디 (distiller / orchestrator / polisher)")
    p.add_argument("--trials", type=int, default=10, help="각 스터디 trial 수 (default: 10)")
    p.add_argument(
        "--study", nargs="+", choices=["distiller", "orchestrator", "polisher"],
        default=["distiller", "orchestrator", "polisher"],
        help="실행할 스터디 지정 (default: 전체)",
    )
    return p.parse_args()


def main() -> None:
    try:
        import optuna  # noqa: F401
    except ImportError:
        print("ERROR: pip install optuna")
        sys.exit(1)

    from src.novel_writer.env_loader import load_project_env
    load_project_env()

    args = parse_args()
    n = args.trials
    selected = set(args.study)

    print(f"\n🚀 스터디 병렬 시작 ({', '.join(sorted(selected))}) — 각 {n} trials")
    for s in sorted(selected):
        print(f"   {s.capitalize()} → output/optuna_{s}/")
    print()

    study_map = {
        "distiller":    run_distiller_study,
        "orchestrator": run_orchestrator_study,
        "polisher":     run_polisher_study,
    }
    processes = [
        multiprocessing.Process(target=study_map[s], args=(n,), name=s)
        for s in ("distiller", "orchestrator", "polisher")
        if s in selected
    ]
    for p in processes:
        p.start()
        print(f"  ▶ {p.name} started (PID {p.pid})")

    # 모니터 (메인 프로세스에서 실행)
    try:
        _monitor(n, selected)
    except KeyboardInterrupt:
        print("\n중단 요청 — 프로세스 종료 중...")
        for p in processes:
            p.terminate()

    for p in processes:
        p.join()

    print("\n📊 최종 결과:")
    for name in ("distiller", "orchestrator", "polisher"):
        bp = Path(f"output/optuna_{name}/best_params.json")
        if bp.exists():
            d = json.loads(bp.read_text())
            print(f"  {name}: best={d['value']:.4f}  trial={d['trial_number']}")
            print(f"    params: {d['params']}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
