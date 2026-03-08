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
import sys
from pathlib import Path
from datetime import datetime

from src.novel_writer.config_loader import load_episode, load_characters
from src.novel_writer.llm_client import LLMClient
from src.novel_writer.scene_distiller import SceneDistiller
from src.novel_writer.prose_generator import ProseGenerator
from src.novel_writer import database as db
from src.novel_writer.rl_policy import load_policy, tuned_scene_target, episode_runtime_policy
from src.novel_writer.env_loader import load_project_env
from src.novel_writer.review_feedback import load_reader_review, resolve_reader_review_path


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
    return p.parse_args()


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
    )
    if review_path:
        reader_feedback = load_reader_review(str(review_path))
        if reader_feedback:
            logger.info(
                "Loaded reader review feedback from %s (weak=%d, tips=%d)",
                review_path,
                len(reader_feedback.get("what_felt_boring_or_hard", []) or []),
                len(reader_feedback.get("style_tips", []) or []),
            )
        else:
            logger.warning("Reader review file parsed but yielded no actionable guidance: %s", review_path)

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
    target_scenes = tuned_scene_target(target_scenes, rl_policy)

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
        runtime_policy=rl_policy,
        reader_feedback=reader_feedback,
    )

    distill_start = datetime.utcnow()
    scenes = distiller.distill(
        episode_id=episode_id,
        protagonist_id=args.protagonist,
        target_scenes=target_scenes,
    )
    distill_elapsed = (datetime.utcnow() - distill_start).total_seconds()

    logger.info(
        "Distilled %d turns into %d scenes (%.1fs)",
        len(interactions), len(scenes), distill_elapsed,
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
        max_history_episodes=int(rl_policy.get("prose_history_max_episodes", 12) or 12),
        runtime_policy=rl_policy,
        reader_feedback=reader_feedback,
    )

    prose_start = datetime.utcnow()
    chapter_path = prose_gen.generate_chapter(
        scenes=scenes,
        protagonist_name=args.protagonist_name,
        style=args.style,
        target_words=target_words,
    )
    prose_elapsed = (datetime.utcnow() - prose_start).total_seconds()

    # === Report ===
    chapter_text = Path(chapter_path).read_text(encoding="utf-8")
    word_count = len(chapter_text.split())
    total_elapsed = distill_elapsed + prose_elapsed

    budget = llm.budget_summary()

    logger.info("=" * 60)
    logger.info("  Chapter: %s", chapter_path)
    logger.info("  Words: %d (target: %d)", word_count, target_words)
    logger.info("  Scenes: %d distilled from %d turns", len(scenes), len(interactions))
    logger.info("  Time: %.1fs (distill: %.1fs, prose: %.1fs)",
                total_elapsed, distill_elapsed, prose_elapsed)
    logger.info("  Budget: $%.4f / $%.2f over %d LLM calls",
                budget["spent_usd"], budget["budget_usd"], budget["call_count"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
