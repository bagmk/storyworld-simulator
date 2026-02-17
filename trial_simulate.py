#!/usr/bin/env python3
"""
trial_simulate.py — Run an AI Story Simulation episode with trial-and-learn.

Wraps the standard simulation in a retry loop that learns from failures:
  - Failed trials are analyzed by the Director
  - Per-agent steering prompts are generated and refined
  - Successful patterns are extracted and reused as exemplars

Usage:
    python trial_simulate.py \\
        --episode  config/episodes/ep01.yaml \\
        --characters config/characters.yaml \\
        --world    config/world_facts.yaml \\
        [--storyline config/storyline.yaml] \\
        [--model   gpt-4o-mini] \\
        [--premium gpt-5-mini] \\
        [--budget  15.00] \\
        [--max-trials 5] \\
        [--success-threshold 1.0] \\
        [--output  output/]

Outputs:
    output/<episode_id>_trial_summary.json    Overall trial results
    output/<episode_id>_trialN_simulation.json  Per-trial interaction logs
    output/<episode_id>_trialN_debug.log        Per-trial director events
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import yaml

from src.novel_writer.config_loader import load_episode, load_world_facts, load_storyline
from src.novel_writer.trial_runner import TrialRunner
from src.novel_writer import database as db


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
        description="AI Story Simulation Engine — trial-and-learn runner"
    )
    p.add_argument("--episode",    required=True, help="Path to episode YAML")
    p.add_argument("--characters", required=True, help="Path to characters YAML")
    p.add_argument("--world",      default="config/world_facts.yaml",
                   help="Path to world_facts YAML")
    p.add_argument("--storyline",  default="config/storyline.yaml",
                   help="Path to storyline YAML for long-arc guardrails")
    p.add_argument("--model",      default="gpt-4o-mini",
                   help="Default LLM model for agent turns")
    p.add_argument("--premium",    default="gpt-5-mini",
                   help="Premium model for Director AI")
    p.add_argument("--budget",     type=float, default=15.0,
                   help="TOTAL budget across all trials (default: $15.00)")
    p.add_argument("--max-trials", type=int, default=5,
                   help="Max trial attempts (default: 5)")
    p.add_argument("--success-threshold", type=float, default=1.0,
                   help="Score threshold for success (default: 1.0 = all clues)")
    p.add_argument("--output",     default="output",
                   help="Output directory (default: output/)")
    p.add_argument("--db",         default="data/simulation.db",
                   help="SQLite database path")
    p.add_argument("--debug",      action="store_true",
                   help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("trial_simulate")

    logger.info("=" * 60)
    logger.info("  AI Story Simulation Engine — Trial & Learn Mode")
    logger.info("=" * 60)

    # ── Override DB path ─────────────────────────────────────────────
    db.DB_PATH = args.db
    db.init_db()
    logger.info("Database initialised at %s", args.db)

    # ── Load configs ─────────────────────────────────────────────────
    logger.info("Loading episode: %s", args.episode)
    episode_config = load_episode(args.episode)
    episode_id = episode_config["id"]

    logger.info("Loading characters: %s", args.characters)
    with Path(args.characters).open("r", encoding="utf-8") as f:
        raw_chars = yaml.safe_load(f).get("characters", [])

    # Attach character invariants for Director checks
    episode_config["character_invariants"] = [
        {"id": c["id"], "invariants": c.get("invariants", [])}
        for c in raw_chars
    ]

    logger.info("Loading world facts: %s", args.world)
    world_facts = load_world_facts(args.world)

    storyline: dict = {}
    storyline_path = Path(args.storyline)
    if storyline_path.exists():
        logger.info("Loading storyline: %s", args.storyline)
        storyline = load_storyline(args.storyline)
    else:
        logger.warning(
            "Storyline file not found at %s. Running without long-arc guardrails.",
            args.storyline,
        )

    # ── Run trials ───────────────────────────────────────────────────
    logger.info(
        "Episode: %s | Max trials: %d | Budget: $%.2f | Threshold: %.0f%%",
        episode_id, args.max_trials, args.budget,
        args.success_threshold * 100,
    )

    runner = TrialRunner(
        episode_config=episode_config,
        characters_config=raw_chars,
        world_facts=world_facts,
        storyline=storyline,
        model=args.model,
        premium_model=args.premium,
        total_budget=args.budget,
        max_trials=args.max_trials,
        success_threshold=args.success_threshold,
        output_dir=args.output,
        db_path=args.db,
    )

    start = datetime.utcnow()
    summary = runner.run()
    elapsed = (datetime.utcnow() - start).total_seconds()

    # ── Write summary ────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{episode_id}_trial_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # ── Final report ─────────────────────────────────────────────────
    logger.info("=" * 60)
    if summary["success"]:
        logger.info(
            "  SUCCESS on trial %d/%d (%.1fs, $%.4f spent)",
            summary["winning_trial"], summary["total_trials"],
            elapsed, summary["budget_used"],
        )
    else:
        logger.info(
            "  FAILED after %d trials (%.1fs, $%.4f spent)",
            summary["total_trials"], elapsed, summary["budget_used"],
        )

    # Per-trial breakdown
    for trial in summary["trials"]:
        status = "OK" if trial["success"] else "FAIL"
        logger.info(
            "  Trial %d [%s] score=%.2f clues=%.0f%% plot=%.0f%% $%.4f",
            trial["trial_number"], status, trial["combined_score"],
            trial["clue_discovery_rate"] * 100,
            trial["plot_resolution_rate"] * 100,
            trial["budget_spent"],
        )

    logger.info("  Summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
