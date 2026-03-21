#!/usr/bin/env python3
"""
generate_chapter.py — Generate a literary novel chapter using the new pipeline.

Uses scene_distiller (turn compression) + prose_generator (YAML-aware literary prose)
instead of the old novel_generator that worked from raw turn logs.

Usage:
    python generate_chapter.py \\
        --episode  ep01_academic_presentation \\
        --episode-config config/episodes/ep01_academic_presentation.yaml \\
        --protagonist kim_sumin \\
        [--protagonist-name "Kim Sumin"] \\
        [--model   gpt-4o-mini] \\
        [--premium gpt-5-mini] \\
        [--budget  5.00] \\
        [--words   3800] \\
        [--scenes  8] \\
        [--style   first_person] \\
        [--output  output/] \\
        [--db      data/simulation.db]

Output:
    output/<episode_id>_chapter.md     Literary novel chapter
    output/<episode_id>_scenes.json    Distilled scene data (debug)

The old generate_chapter.py (using novel_generator.py) still works for comparison.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime

from src.novel_writer.config_loader import load_episode, load_characters
from src.novel_writer.llm_client import LLMClient
from src.novel_writer.scene_distiller import SceneDistiller
from src.novel_writer.scene_distiller import DistilledScene
from src.novel_writer.prose_generator import ProseGenerator
from src.novel_writer import database as db
from src.novel_writer.rl_policy import load_policy, tuned_scene_target, episode_runtime_policy
from src.novel_writer.env_loader import load_project_env
from src.novel_writer.reader_profile import (
    MAX_WORDS_PER_SCENE as PROFILE_MAX_WORDS_PER_SCENE,
    build_reader_profile,
)
from src.novel_writer.review_feedback import (
    ensure_jargon_watch_terms,
    ensure_repetition_watch_terms,
    load_reader_review,
    resolve_reader_review_path,
)


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Story Simulation Engine — literary chapter generator (v2)"
    )
    p.add_argument("--episode",        required=True,
                   help="Episode ID in the database (e.g., ep01_academic_presentation)")
    p.add_argument("--episode-config", required=True,
                   help="Path to episode YAML config file")
    p.add_argument("--protagonist",    required=True,
                   help="Agent ID for POV protagonist (e.g., kim_sumin)")
    p.add_argument("--protagonist-name", default="Kim Sumin",
                   help="Display name of protagonist for prose (default: Kim Sumin)")
    p.add_argument("--characters",      default="config/characters.yaml",
                   help="Path to character YAML for voice profiles (default: config/characters.yaml)")
    p.add_argument("--model",          default="gpt-4o-mini",
                   help="Default LLM model")
    p.add_argument("--premium",        default="gpt-5-mini",
                   help="Premium model for prose generation")
    p.add_argument("--budget",         type=float, default=5.0,
                   help="USD budget cap (default: $5.00)")
    p.add_argument("--words",          type=int, default=0,
                   help="Target word count (default: from episode config)")
    p.add_argument("--scenes",         type=int, default=0,
                   help="Target number of distilled scenes (default: auto-calculated from word count)")
    p.add_argument("--style",          default="third_person_close",
                   choices=["first_person", "third_person_close"],
                   help="Narrative POV style (default: third_person_close)")
    p.add_argument("--output",         default="output",
                   help="Output directory (default: output/)")
    p.add_argument("--db",             default="data/simulation.db",
                   help="SQLite database path")
    p.add_argument("--debug",          action="store_true",
                   help="Enable debug logging")
    p.add_argument("--track-run-id", default="",
                   help="Tracking run identifier (overrides NOVEL_RUN_ID)")
    p.add_argument("--track-iteration", type=int, default=None,
                   help="Tracking iteration number (overrides NOVEL_ITERATION)")
    p.add_argument("--track-phase", default="",
                   help="Tracking phase label (overrides NOVEL_PHASE)")
    p.add_argument("--reader-review-md", default="",
                   help="Optional reader review markdown for readability/style steering")
    p.add_argument("--guardian-briefing", default="",
                   help="Optional guardian GPT analysis text file for story continuity steering")
    p.add_argument("--precomputed-scenes", default="",
                   help="Optional precomputed scenes JSON path to skip scene distillation")
    return p.parse_args()


def _reader_feedback_corpus(reader_feedback: dict) -> str:
    return build_reader_profile(reader_feedback).corpus


def _reader_feedback_has_any(reader_feedback: dict, *tokens: str) -> bool:
    return build_reader_profile(reader_feedback).has_any(*tokens)


def _reader_feedback_mentions_stalled_progression(reader_feedback: dict) -> bool:
    return build_reader_profile(reader_feedback).reports_stalled_progression()


def _reader_feedback_needs_draft_cleanup(reader_feedback: dict) -> bool:
    return build_reader_profile(reader_feedback).needs_draft_cleanup()


def _reader_feedback_prefers_sumin_first_person(reader_feedback: dict) -> bool:
    return build_reader_profile(reader_feedback).prefers_sumin_first_person()


def _sanitize_chapter_draft_artifacts(chapter_text: str, reader_feedback: dict) -> str:
    return build_reader_profile(reader_feedback).sanitize_chapter_draft_artifacts(chapter_text)


# ── 씬 floor 계산 상수 ──────────────────────────────────────────────────────
# 씬당 최대 단어수를 초과하면 LLM 호출이 느려지고 타임아웃 위험이 생깁니다.
# floor = ceil(target_words / _MAX_WORDS_PER_SCENE) 으로 동적 계산됩니다.
# ⚠️ AI 코드 수정 도구(Fixer): 이 상수와 아래 floor 계산식을 수정하지 마세요.
# AI codefixer: DO NOT modify _MAX_WORDS_PER_SCENE or the floor formula in adjust_scene_target_for_feedback.
_MAX_WORDS_PER_SCENE = PROFILE_MAX_WORDS_PER_SCENE  # words — latency-tested upper bound per LLM call


def adjust_scene_target_for_feedback(
    target_scenes: int,
    target_words: int,
    reader_feedback: dict,
) -> int:
    return build_reader_profile(reader_feedback).adjusted_scene_target(
        target_scenes,
        target_words,
        max_words_per_scene=_MAX_WORDS_PER_SCENE,
    )


def _resolve_generation_style(cli_style: str, reader_feedback: dict) -> str:
    return build_reader_profile(reader_feedback).resolve_generation_style(cli_style)


def _apply_reader_feedback_pipeline_overrides(reader_feedback: dict) -> dict:
    return build_reader_profile(reader_feedback).as_dict()


def _chapter_runtime_policy(base_policy: dict, reader_feedback: dict) -> dict:
    policy = dict(base_policy or {})
    profile = build_reader_profile(reader_feedback)
    if profile.reports_stalled_progression():
        policy["hold_pressure_peak"] = 1
        policy["prefer_scene_exit_on_stall"] = 1
        policy["prefer_concrete_offer_detail"] = 1
        policy["prefer_concrete_threat_detail"] = 1
        policy["prefer_concrete_transition_cue"] = 1
    if profile.flags_stock_bridge_phrases():
        policy["prefer_concrete_transition_cue"] = 1
    if profile.prefers_stronger_scene_compaction():
        policy["prefer_concrete_offer_detail"] = 1
        policy["prefer_concrete_threat_detail"] = 1
        policy["prefer_concrete_transition_cue"] = 1
    if profile.prefers_dialogue_compaction():
        policy["prefer_concrete_offer_detail"] = 1
        policy["prefer_concrete_threat_detail"] = 1
    if profile.prefers_explicit_transition_cues():
        policy["prefer_concrete_transition_cue"] = 1
    return policy


def _load_precomputed_scenes(path: str) -> list[DistilledScene]:
    raw_list = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes: list[DistilledScene] = []
    for item in raw_list:
        turn_range = _normalize_turn_range(item.get("turn_range", [0, 0]))
        scenes.append(
            DistilledScene(
                scene_number=_coerce_intish(item.get("scene_number", 0), default=len(scenes) + 1),
                title=str(item.get("title", "")),
                turn_range=turn_range,
                location=str(item.get("location", "")),
                characters_present=_coerce_string_list(item.get("characters_present", [])),
                key_dialogue=_coerce_dialogue_rows(item.get("key_dialogue", [])),
                key_actions=_coerce_string_list(item.get("key_actions", [])),
                discoveries=_coerce_string_list(item.get("discoveries", [])),
                emotional_arc=str(item.get("emotional_arc", "")),
                beat_references=_coerce_string_list(item.get("beat_references", [])),
                narrative_summary=str(item.get("narrative_summary", "")),
                pacing=str(item.get("pacing", "")),
                raw_turn_count=max(1, _coerce_intish(item.get("raw_turn_count", 0), default=1)),
            )
        )
    return scenes


def _coerce_intish(value, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (list, tuple)):
        first = value[0] if value else default
        return _coerce_intish(first, default=default)
    raw = str(value or "")
    if not raw.strip():
        return default
    raw = raw.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")).strip()
    match = re.search(r"(?<!\d)(\d{1,5})(?!\d)", raw)
    if not match:
        return default
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return default


def _normalize_turn_range(raw_range) -> tuple[int, int]:
    if isinstance(raw_range, str):
        values = re.findall(
            r"\d{1,5}",
            raw_range.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")),
        )[:2]
    else:
        values = list(raw_range)[:2] if isinstance(raw_range, (list, tuple)) else [raw_range]
    if len(values) < 2:
        fill = values[0] if values else 0
        values = (values + [fill, fill])[:2]
    start = _coerce_intish(values[0], default=0)
    end = _coerce_intish(values[1], default=start)
    return (start, end) if start <= end else (end, start)


def _coerce_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,/|]\s*|\n+", value)
        return [part.strip() for part in parts if part.strip()]
    return []


def _coerce_dialogue_rows(value) -> list[dict]:
    if isinstance(value, dict):
        value = [value]
    elif isinstance(value, str):
        value = [value]
    elif not isinstance(value, list):
        return []

    rows: list[dict] = []
    for raw in value:
        if isinstance(raw, dict):
            speaker = str(raw.get("speaker", "") or raw.get("name", "")).strip() or "Unknown"
            line = str(raw.get("line", "") or raw.get("content", "")).strip()
        else:
            text = str(raw or "").strip()
            if not text:
                continue
            match = re.match(r"^\s*([^:：]{1,40})\s*[:：]\s*(.+?)\s*$", text)
            if match:
                speaker = match.group(1).strip() or "Unknown"
                line = match.group(2).strip()
            else:
                speaker = "Unknown"
                line = text
        if not line:
            continue
        rows.append({"speaker": speaker, "line": line})
    return rows


def _log_sentence_pattern_warnings(
    prose_gen: ProseGenerator,
    chapter_text: str,
    logger: logging.Logger,
) -> list[str]:
    warnings = prose_gen.detect_repetition_pattern_warnings(chapter_text)
    for warning in warnings:
        logger.warning("Sentence pattern guard: %s", warning)
    return warnings


def main() -> None:
    load_project_env()
    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("generate_chapter")

    logger.info("=" * 60)
    logger.info("  Literary Chapter Generator (v2: distill → prose)")
    logger.info("=" * 60)

    # Override DB path
    db.DB_PATH = args.db
    db.init_db()
    db.configure_tracking_from_env()
    if args.track_run_id:
        db.set_tracking_context(run_id=args.track_run_id)
    if args.track_iteration is not None:
        db.set_tracking_context(iteration=args.track_iteration)
    if args.track_phase:
        db.set_tracking_context(phase=args.track_phase)
    tracking = db.get_tracking_context()
    if tracking.get("run_id"):
        logger.info(
            "Tracking | run_id=%s iteration=%s phase=%s",
            tracking.get("run_id"),
            tracking.get("iteration"),
            tracking.get("phase"),
        )

    # Load episode config from YAML
    logger.info("Loading episode config: %s", args.episode_config)
    episode_config = load_episode(args.episode_config)
    episode_id = str(episode_config.get("id") or args.episode).strip()
    if args.episode and args.episode != episode_id:
        logger.info(
            "Episode ID normalized from CLI '%s' to config id '%s'",
            args.episode,
            episode_id,
        )
    rl_policy = load_policy()
    episode_config["_rl_runtime"] = episode_runtime_policy(rl_policy)
    reader_feedback: dict = {}
    review_path = resolve_reader_review_path(
        explicit_path=args.reader_review_md,
        episode_id=episode_id,
        output_dir=args.output,
        prefer_run_id=str(tracking.get("run_id") or ""),
    )
    if review_path:
        reader_feedback = load_reader_review(str(review_path))
        reader_feedback = ensure_repetition_watch_terms(reader_feedback)
        reader_feedback = ensure_jargon_watch_terms(reader_feedback)
        reader_feedback = _apply_reader_feedback_pipeline_overrides(reader_feedback)
        if reader_feedback:
            repeat_terms = reader_feedback.get("repetition_watch_terms", []) or []
            jargon_terms = reader_feedback.get("jargon_watch_terms", []) or []
            style_constraints = reader_feedback.get("style_constraints", {}) or {}
            fixer_actions = reader_feedback.get("fixer_priority_actions", []) or []
            logger.info(
                "Loaded reader review feedback from %s (weak=%d, fixer=%d, tips=%d, repeat_terms=%d, jargon_terms=%d, style_constraints=%d)",
                review_path,
                len(reader_feedback.get("what_felt_boring_or_hard", []) or []),
                len(fixer_actions),
                len(reader_feedback.get("style_tips", []) or []),
                len(repeat_terms),
                len(jargon_terms),
                len(style_constraints) if isinstance(style_constraints, dict) else 0,
            )
        else:
            logger.warning("Reader review file parsed but yielded no actionable guidance: %s", review_path)
    normalized_reader_feedback = _apply_reader_feedback_pipeline_overrides(reader_feedback)
    chapter_runtime_policy = _chapter_runtime_policy(rl_policy, normalized_reader_feedback)
    episode_config["_rl_runtime"] = episode_runtime_policy(chapter_runtime_policy)

    # Load guardian briefing for story continuity steering.
    guardian_briefing = ""
    if args.guardian_briefing:
        briefing_path = Path(args.guardian_briefing)
        if briefing_path.exists():
            guardian_briefing = briefing_path.read_text(encoding="utf-8").strip()
            logger.info("Loaded guardian briefing from %s (%d chars)", briefing_path, len(guardian_briefing))
        else:
            logger.warning("Guardian briefing file not found: %s", briefing_path)

    # Load character profiles for voice/style guidance in prose generation.
    character_profiles = []
    try:
        agents = load_characters(args.characters)
        for a in agents:
            character_profiles.append(
                {
                    "id": a.id,
                    "name": a.name,
                    "aliases": list(a.aliases or []),
                    "speech_profile": dict(a.speech_profile or {}),
                    "visual_profile": dict(a.visual_profile or {}),
                }
            )
        logger.info("Loaded %d character voice profiles from %s",
                    len(character_profiles), args.characters)
    except Exception as exc:
        logger.warning("Could not load character voice profiles from %s: %s",
                       args.characters, exc)

    # Determine target words
    target_words = args.words or episode_config.get("recommended_length", 3500)
    resolved_style = _resolve_generation_style(args.style, normalized_reader_feedback)
    if resolved_style != args.style:
        logger.info(
            "Reader feedback adjusted prose POV style: %s -> %s",
            args.style,
            resolved_style,
        )

    # Auto-calculate target scenes based on word count if not specified
    # Logic: shorter episodes need fewer scenes to avoid fragmentation
    # - Under 1000 words: 3-4 scenes (250-330 words/scene)
    # - 1000-2000 words: 4-6 scenes (250-400 words/scene)
    # - 2000-4000 words: 6-8 scenes (300-500 words/scene)
    # - Over 4000 words: 8-10 scenes (400-600 words/scene)
    if args.scenes > 0:
        target_scenes = args.scenes  # User override
    else:
        if target_words < 1000:
            target_scenes = 3
        elif target_words < 2000:
            target_scenes = 5
        elif target_words < 4000:
            target_scenes = 7
        else:
            target_scenes = 8
    feedback_adjusted_target_scenes = adjust_scene_target_for_feedback(
        target_scenes=target_scenes,
        target_words=target_words,
        reader_feedback=normalized_reader_feedback,
    )
    if feedback_adjusted_target_scenes != target_scenes:
        logger.info(
            "Reader feedback adjusted target scenes: %d -> %d",
            target_scenes,
            feedback_adjusted_target_scenes,
        )
        target_scenes = feedback_adjusted_target_scenes
    target_scenes = tuned_scene_target(target_scenes, rl_policy)

    target_scenes = max(1, int(target_scenes))
    logger.info("Target words: %d | Target scenes: %d (%.0f words/scene avg)",
                target_words, target_scenes, target_words / target_scenes)

    # Check episode exists in DB
    interactions = db.load_episode_interactions(episode_id)
    if not interactions:
        logger.error(
            "No interactions found for '%s' in database %s. "
            "Run simulate.py or trial_simulate.py first.",
            episode_id, args.db,
        )
        sys.exit(1)
    logger.info("Found %d interactions for '%s'", len(interactions), episode_id)

    # Build LLM client
    llm = LLMClient(
        model=args.model,
        premium_model=args.premium,
        budget_usd=args.budget,
    )

    # === Stage 1: Scene Distillation ===
    logger.info("─── Stage 1: Scene Distillation ───")
    distiller = SceneDistiller(
        llm=llm,
        episode_config=episode_config,
        runtime_policy=chapter_runtime_policy,
        reader_feedback=normalized_reader_feedback,
    )
    if args.precomputed_scenes:
        scenes = _load_precomputed_scenes(args.precomputed_scenes)
        distill_elapsed = 0.0
        logger.info(
            "Reused %d precomputed scenes from %s (distill skipped)",
            len(scenes),
            args.precomputed_scenes,
        )
    else:
        distill_start = datetime.utcnow()
        distill_fallback_used = False
        try:
            scenes = distiller.distill(
                episode_id=episode_id,
                protagonist_id=args.protagonist,
                target_scenes=target_scenes,
            )
            if not scenes:
                raise ValueError("scene distiller returned no scenes")
        except Exception as exc:
            logger.exception(
                "Scene distillation failed for '%s'; falling back to deterministic chunking: %s",
                episode_id,
                exc,
            )
            pov = distiller._filter_perspective(interactions, args.protagonist)
            beats = distiller._extract_beats()
            scenes = distiller._fallback_chunk(pov, beats, target_scenes)
            distill_fallback_used = True
            if not scenes:
                raise
        distill_elapsed = (datetime.utcnow() - distill_start).total_seconds()

        logger.info(
            "Distilled %d turns into %d scenes (%.1fs%s)",
            len(interactions), len(scenes), distill_elapsed,
            " | fallback chunking" if distill_fallback_used else "",
        )
    canonical_speakers = distiller._build_canonical_speaker_map(interactions)
    scenes = distiller.normalize_scene_timeline(
        distiller.apply_scene_guards(scenes, canonical_speakers)
    )
    for s in scenes:
        logger.info(
            "  Scene %d: '%s' [T%d-%d] %s — %s",
            s.scene_number, s.title, s.turn_range[0], s.turn_range[1],
            s.pacing, s.emotional_arc[:60] if s.emotional_arc else "",
        )

    # Save distilled scenes for debugging
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes_path = output_dir / f"{episode_id}_scenes.json"
    with scenes_path.open("w", encoding="utf-8") as f:
        json.dump(
            [s.to_dict() for s in scenes],
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Scene data → %s", scenes_path)

    # === Stage 2: Prose Generation ===
    logger.info("─── Stage 2: Prose Generation ───")
    prose_gen = ProseGenerator(
        llm=llm,
        episode_config=episode_config,
        output_dir=args.output,
        character_profiles=character_profiles,
        max_history_episodes=int(chapter_runtime_policy.get("prose_history_max_episodes", 12) or 12),
        runtime_policy=chapter_runtime_policy,
        reader_feedback=normalized_reader_feedback,
        guardian_briefing=guardian_briefing,
    )

    prose_start = datetime.utcnow()
    chapter_path = prose_gen.generate_chapter(
        scenes=scenes,
        protagonist_name=args.protagonist_name,
        style=resolved_style,
        target_words=target_words,
    )
    prose_elapsed = (datetime.utcnow() - prose_start).total_seconds()

    # === Report ===
    chapter_text = Path(chapter_path).read_text(encoding="utf-8")
    cleaned_chapter_text = _sanitize_chapter_draft_artifacts(chapter_text, normalized_reader_feedback)
    if cleaned_chapter_text != chapter_text:
        Path(chapter_path).write_text(cleaned_chapter_text, encoding="utf-8")
        chapter_text = cleaned_chapter_text
        logger.info("Applied deterministic draft-artifact cleanup to %s", chapter_path)
    pattern_warnings = _log_sentence_pattern_warnings(prose_gen, chapter_text, logger)
    word_count = len(chapter_text.split())
    total_elapsed = distill_elapsed + prose_elapsed

    budget = llm.budget_summary() or {}
    spent_usd = float(budget.get("spent_usd", 0.0) or 0.0)
    budget_usd = float(budget.get("budget_usd", 0.0) or 0.0)
    call_count = int(budget.get("call_count", 0) or 0)
    meta_path = Path(chapter_path).with_name(f"{Path(chapter_path).stem}_meta.json")
    meta_payload = {
        "episode_id": episode_id,
        "chapter_path": str(chapter_path),
        "word_count": word_count,
        "target_words": target_words,
        "scene_count": len(scenes),
        "interaction_count": len(interactions),
        "elapsed_seconds": {
            "total": total_elapsed,
            "distill": distill_elapsed,
            "prose": prose_elapsed,
        },
        "budget": budget,
        "generated_at": datetime.utcnow().isoformat(),
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("  Chapter: %s", chapter_path)
    logger.info("  Words: %d (target: %d)", word_count, target_words)
    logger.info("  Scenes: %d distilled from %d turns", len(scenes), len(interactions))
    if pattern_warnings:
        logger.info("  Pattern warnings: %d", len(pattern_warnings))
    logger.info("  Time: %.1fs (distill: %.1fs, prose: %.1fs)",
                total_elapsed, distill_elapsed, prose_elapsed)
    logger.info(
        "  Budget: $%.4f / $%.2f over %d LLM calls | tokens: %d in + %d out = %d total",
        spent_usd,
        budget_usd,
        call_count,
        budget.get("prompt_tokens", 0),
        budget.get("completion_tokens", 0),
        budget.get("total_tokens", 0),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
