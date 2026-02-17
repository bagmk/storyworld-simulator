#!/usr/bin/env python3
"""
simulate.py — Run an AI Story Simulation episode.

Usage:
    python simulate.py \\
        --episode  config/episodes/ep01.yaml \\
        --characters config/characters.yaml \\
        --world    config/world_facts.yaml \\
        [--model   gpt-4o-mini] \\
        [--premium gpt-5-mini] \\
        [--budget  5.00] \\
        [--output  output/]

Outputs:
    output/<episode_id>_simulation.json   Full interaction log
    output/<episode_id>_debug.log         Director interventions
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.novel_writer.config_loader import (
    load_characters, load_episode, load_world_facts,
    load_storyline, build_world_state, build_clue_manager,
)
from src.novel_writer.llm_client import LLMClient
from src.novel_writer.director import DirectorAI
from src.novel_writer.orchestrator import SimulationOrchestrator
from src.novel_writer import database as db


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    # 외부 라이브러리 노이즈 억제 (--debug 없을 때)
    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Story Simulation Engine — episode runner"
    )
    p.add_argument("--episode",    required=True, help="Path to episode YAML")
    p.add_argument("--characters", required=True, help="Path to characters YAML")
    p.add_argument("--world",      default="config/world_facts.yaml",
                   help="Path to world_facts YAML (default: config/world_facts.yaml)")
    p.add_argument("--storyline",  default="config/storyline.yaml",
                   help="Path to storyline YAML for long-arc guardrails (default: config/storyline.yaml)")
    p.add_argument("--model",      default="gpt-4o-mini",
                   help="Default LLM model for agent turns")
    p.add_argument("--premium",    default="gpt-5-mini",
                   help="Premium model for Director AI and narrative generation")
    p.add_argument("--budget",     type=float, default=3.0,
                   help="USD budget cap for the episode (default: $3.00)")
    p.add_argument("--output",     default="output",
                   help="Output directory (default: output/)")
    p.add_argument("--db",         default="data/simulation.db",
                   help="SQLite database path (default: data/simulation.db)")
    p.add_argument("--debug",      action="store_true",
                   help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    logger = logging.getLogger("simulate")
    logger.info("═" * 60)
    logger.info("  AI Story Simulation Engine")
    logger.info("═" * 60)

    # ── Override DB path ────────────────────────────────────────────────
    db.DB_PATH = args.db

    # ── Init database ───────────────────────────────────────────────────
    db.init_db()
    logger.info("Database initialised at %s", args.db)

    # ── Load configs ────────────────────────────────────────────────────
    logger.info("Loading episode: %s", args.episode)
    episode_config = load_episode(args.episode)
    episode_id     = episode_config["id"]

    logger.info("Loading characters: %s", args.characters)
    agents = load_characters(args.characters)

    # Emotion continuity: seed each agent with the latest prior episode state.
    loaded_count = 0
    for agent in agents:
        prev_emotions = db.load_previous_episode_final_emotions(
            agent_id=agent.id,
            current_episode_id=episode_id,
        )
        if prev_emotions:
            agent.memory.emotional_state = prev_emotions
            loaded_count += 1
    if loaded_count:
        logger.info("Loaded previous emotion states for %d agents", loaded_count)
    else:
        logger.info("No prior emotion states found to preload")

    logger.info("Loading world facts: %s", args.world)
    world_facts = load_world_facts(args.world)

    storyline: dict = {}
    if args.storyline:
        storyline_path = Path(args.storyline)
        if storyline_path.exists():
            logger.info("Loading storyline: %s", args.storyline)
            storyline = load_storyline(args.storyline)
        else:
            logger.warning(
                "Storyline file not found at %s. Running without long-arc guardrails.",
                args.storyline,
            )

    # ── Build subsystems ────────────────────────────────────────────────
    world       = build_world_state(episode_config, world_facts, agents)
    clue_mgr    = build_clue_manager(episode_config, world_facts)

    llm = LLMClient(
        model=args.model,
        premium_model=args.premium,
        budget_usd=args.budget,
    )
    logger.info("Models | agent_turn=%s | director=%s", llm.model, llm.premium_model)

    # Attach character invariants to episode_config for Director checks.
    # Keep episode_config["characters"] intact for optional episode cast config.
    episode_config["character_invariants"] = [
        {"id": a.id, "invariants": a.invariants} for a in agents
    ]

    director = DirectorAI(
        episode_config=episode_config,
        world_facts=world_facts,
        clue_manager=clue_mgr,
        storyline=storyline,
        llm=llm,
    )
    storyline_ctx = director.storyline_context
    current_milestone = storyline_ctx.get("current", {})
    if storyline:
        if current_milestone:
            logger.info(
                "Storyline guardrail | source=%s | act=%s | milestone=%s",
                episode_config.get("storyline_source_id", "(none)"),
                current_milestone.get("act_id", "(unknown)"),
                current_milestone.get("id", "(unknown)"),
            )
            logger.info(
                "Storyline focus: %s",
                current_milestone.get("description", "(no description)"),
            )
        else:
            logger.warning(
                "Storyline guardrail enabled but no milestone matched episode '%s' "
                "(source=%s).",
                episode_id,
                episode_config.get("storyline_source_id", "(none)"),
            )
    else:
        logger.info("Storyline guardrail disabled (no storyline config loaded).")

    orchestrator = SimulationOrchestrator(
        agents=agents,
        director=director,
        world=world,
        llm=llm,
        episode_id=episode_id,
        episode_config=episode_config,
    )

    # ── Run simulation ──────────────────────────────────────────────────
    start = datetime.utcnow()
    logger.info("Starting episode '%s' with %d agents | budget $%.2f",
                episode_id, len(agents), args.budget)

    interactions = orchestrator.run_episode()

    elapsed = (datetime.utcnow() - start).total_seconds()
    logger.info("Episode complete in %.1fs | %d interactions", elapsed, len(interactions))

    # ── Write outputs ───────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulation JSON
    sim_path = output_dir / f"{episode_id}_simulation.json"
    with sim_path.open("w", encoding="utf-8") as f:
        json.dump({
            "episode_id":   episode_id,
            "agent_count":  len(agents),
            "total_turns":  orchestrator.turn,
            "interactions": interactions,
            "budget":       llm.budget_summary(),
            "generated_at": datetime.utcnow().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    logger.info("Simulation log → %s", sim_path)

    # Debug log
    debug_path = output_dir / f"{episode_id}_debug.log"
    with debug_path.open("w", encoding="utf-8") as f:
        for entry in director.debug_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Debug log → %s (%d director events)", debug_path, len(director.debug_log))

    # Budget summary
    budget = llm.budget_summary()
    logger.info(
        "Budget used: $%.4f / $%.2f (%.1f%%) over %d LLM calls",
        budget["spent_usd"],
        budget["budget_usd"],
        100 * budget["spent_usd"] / max(budget["budget_usd"], 0.001),
        budget["call_count"],
    )

    logger.info("═" * 60)
    logger.info("  Done! Run generate_chapter.py to create the chapter.")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
