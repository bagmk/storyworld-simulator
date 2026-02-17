"""
Trial-and-Learn runner for the AI Story Simulation Engine.

Wraps the SimulationOrchestrator in a retry loop:
  1. Run episode simulation
  2. Evaluate outcome (clue discovery, plot resolution, beat completion)
  3. On failure: Director analyzes why → updates per-agent steering prompts
  4. On success: extract successful interaction patterns as exemplars
  5. Persist trial results and steering history to DB
"""

from __future__ import annotations
import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from models import Agent, SteeringContext, ClueManager
from config_loader import build_world_state, build_clue_manager
from llm_client import LLMClient
from director import DirectorAI
from orchestrator import SimulationOrchestrator
import database as db

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Outcome of a single trial attempt."""
    trial_number: int
    success: bool
    clue_discovery_rate: float
    plot_resolution_rate: float
    beat_complete: bool
    combined_score: float
    failure_reasons: list[str]
    interactions: list[dict]
    budget_spent: float
    turn_count: int
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "trial_number": self.trial_number,
            "success": self.success,
            "clue_discovery_rate": self.clue_discovery_rate,
            "plot_resolution_rate": self.plot_resolution_rate,
            "beat_complete": self.beat_complete,
            "combined_score": self.combined_score,
            "failure_reasons": self.failure_reasons,
            "budget_spent": self.budget_spent,
            "turn_count": self.turn_count,
            "timestamp": self.timestamp,
        }


class TrialRunner:
    """
    Runs multiple trials of the same episode, learning from failures.

    Each trial creates fresh agents, world state, and clue manager from
    the raw configs. Only SteeringContext carries over between trials,
    accumulating lessons from previous failures.
    """

    def __init__(
        self,
        episode_config: dict,
        characters_config: list[dict],
        world_facts: dict,
        storyline: dict,
        model: str = "gpt-4o-mini",
        premium_model: str = "gpt-5-mini",
        total_budget: float = 15.0,
        max_trials: int = 5,
        success_threshold: float = 1.0,
        output_dir: str = "output",
        db_path: str = "data/simulation.db",
    ) -> None:
        self.episode_config = episode_config
        self.characters_config = characters_config
        self.world_facts = world_facts
        self.storyline = storyline
        self.model = model
        self.premium_model = premium_model
        self.total_budget = total_budget
        self.max_trials = max_trials
        self.success_threshold = success_threshold
        self.output_dir = Path(output_dir)
        self.db_path = db_path

        self.base_episode_id = episode_config["id"]
        self.budget_spent_total = 0.0
        self.trial_results: list[TrialResult] = []

    # ------------------------------------------------------------------ #
    # Public: Run Trial Loop
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """
        Execute the trial-and-learn loop.

        Returns summary dict with success status, trial details, and budget.
        """
        logger.info(
            "Trial-and-Learn | episode=%s | max_trials=%d | budget=$%.2f | threshold=%.0f%%",
            self.base_episode_id, self.max_trials, self.total_budget,
            self.success_threshold * 100,
        )

        # Initialize empty steering contexts for all agents
        steering_contexts: dict[str, SteeringContext] = {
            c["id"]: SteeringContext(agent_id=c["id"])
            for c in self.characters_config
        }

        winning_trial: Optional[int] = None
        best_interactions: Optional[list[dict]] = None

        for trial_num in range(1, self.max_trials + 1):
            remaining = self.total_budget - self.budget_spent_total
            if remaining <= 0.01:
                logger.warning("Budget exhausted after %d trials.", trial_num - 1)
                break

            trial_budget = self._allocate_trial_budget(trial_num, remaining)

            logger.info(
                "━━━ Trial %d/%d | budget=$%.2f | remaining=$%.2f ━━━",
                trial_num, self.max_trials, trial_budget, remaining,
            )

            result, clue_mgr, director, agents = self._run_single_trial(
                trial_num, trial_budget, steering_contexts,
            )

            self.budget_spent_total += result.budget_spent
            self.trial_results.append(result)

            # Persist trial result and steering state
            self._persist_trial(trial_num, result, steering_contexts)

            logger.info(
                "Trial %d result: score=%.2f | clues=%.0f%% | plot=%.0f%% | beat=%s | $%.4f",
                trial_num, result.combined_score,
                result.clue_discovery_rate * 100,
                result.plot_resolution_rate * 100,
                result.beat_complete, result.budget_spent,
            )

            if result.success:
                winning_trial = trial_num
                best_interactions = result.interactions
                logger.info("Trial %d SUCCEEDED!", trial_num)

                # Extract and persist success patterns
                patterns = director.extract_success_patterns(
                    result.interactions, clue_mgr,
                )
                for pattern in patterns:
                    db.save_trial_exemplar(
                        self.base_episode_id, trial_num, pattern,
                    )
                logger.info("Extracted %d success patterns.", len(patterns))

                # Write winning trial's simulation output
                self._write_trial_output(trial_num, result, director)
                break

            # On failure: analyze and update steering
            if trial_num < self.max_trials:
                logger.info("Analyzing failure for trial %d...", trial_num)
                steering_contexts = self._generate_steering_updates(
                    result, clue_mgr, director, agents, steering_contexts,
                )
                for ctx in steering_contexts.values():
                    ctx.attempt_number = trial_num + 1

            # Write failed trial output too
            self._write_trial_output(trial_num, result, director)

        summary = {
            "episode_id": self.base_episode_id,
            "success": winning_trial is not None,
            "winning_trial": winning_trial,
            "total_trials": len(self.trial_results),
            "max_trials": self.max_trials,
            "trials": [r.to_dict() for r in self.trial_results],
            "budget_total": self.total_budget,
            "budget_used": self.budget_spent_total,
            "best_interactions_count": len(best_interactions) if best_interactions else 0,
        }
        return summary

    # ------------------------------------------------------------------ #
    # Single Trial Execution
    # ------------------------------------------------------------------ #

    def _run_single_trial(
        self,
        trial_number: int,
        trial_budget: float,
        steering_contexts: dict[str, SteeringContext],
    ) -> tuple[TrialResult, ClueManager, DirectorAI, list[Agent]]:
        """
        Run one trial with fresh state but carried-over steering contexts.
        """
        # Fresh agents from raw config (clean memory each trial)
        agents = [Agent.from_config(c) for c in self.characters_config]

        # EMOTION CONTINUITY: Load final emotional state from previous episode
        for agent in agents:
            prev_emotions = db.load_previous_episode_final_emotions(
                agent_id=agent.id,
                current_episode_id=self.base_episode_id
            )
            if prev_emotions:
                agent.memory.emotional_state = prev_emotions
                logger.info(f"Loaded previous emotions for {agent.id}: {prev_emotions}")

        # Deep copy world_facts to prevent cross-trial mutation
        trial_world_facts = copy.deepcopy(self.world_facts)

        # Fresh world state and clue manager
        world = build_world_state(self.episode_config, trial_world_facts, agents)
        clue_mgr = build_clue_manager(self.episode_config, trial_world_facts)

        # Fresh LLM client with trial-scoped budget
        llm = LLMClient(
            model=self.model,
            premium_model=self.premium_model,
            budget_usd=trial_budget,
        )

        # Director with fresh clue manager
        director = DirectorAI(
            episode_config=self.episode_config,
            world_facts=trial_world_facts,
            clue_manager=clue_mgr,
            storyline=self.storyline,
            llm=llm,
        )

        # Trial-scoped episode ID to avoid DB collisions
        trial_episode_id = f"{self.base_episode_id}_trial{trial_number}"

        orchestrator = SimulationOrchestrator(
            agents=agents,
            director=director,
            world=world,
            llm=llm,
            episode_id=trial_episode_id,
            episode_config=self.episode_config,
            steering_contexts=steering_contexts,
        )

        interactions = orchestrator.run_episode()

        result = self._evaluate_trial(
            trial_number, interactions, clue_mgr, director, world, llm,
        )

        return result, clue_mgr, director, agents

    # ------------------------------------------------------------------ #
    # Trial Evaluation
    # ------------------------------------------------------------------ #

    def _evaluate_trial(
        self,
        trial_number: int,
        interactions: list[dict],
        clue_manager: ClueManager,
        director: DirectorAI,
        world,
        llm: LLMClient,
    ) -> TrialResult:
        """Score the trial outcome."""
        # Clue discovery rate
        total_required = len(clue_manager.required_clues)
        discovered = len(clue_manager.introduced)
        clue_rate = discovered / max(total_required, 1)

        # Plot thread resolution
        unresolved = director._find_unresolved_threads(
            world=world,
            recent_interactions=interactions[-12:],
        )
        total_threads = len(director.must_resolve)
        resolved_count = total_threads - len(unresolved)
        plot_rate = resolved_count / max(total_threads, 1) if total_threads > 0 else 1.0

        # Beat completion
        beat_ok, _ = director.validate_resolution(
            turn=world.turn, world=world,
            recent_interactions=interactions[-12:],
        )

        # Combined score (weighted)
        combined = (clue_rate * 0.5) + (plot_rate * 0.3) + (0.2 if beat_ok else 0.0)

        # Failure reasons
        reasons: list[str] = []
        if clue_rate < 1.0:
            missing = clue_manager.unintroduced_required()
            reasons.append(
                f"Missing clues: {[c.get('id') for c in missing]}"
            )
        if unresolved:
            reasons.append(f"Unresolved threads: {unresolved}")
        if not beat_ok:
            reasons.append("Beat not complete")

        success = combined >= self.success_threshold

        return TrialResult(
            trial_number=trial_number,
            success=success,
            clue_discovery_rate=clue_rate,
            plot_resolution_rate=plot_rate,
            beat_complete=beat_ok,
            combined_score=combined,
            failure_reasons=reasons,
            interactions=interactions,
            budget_spent=llm.spent_usd,
            turn_count=world.turn,
            timestamp=datetime.utcnow().isoformat(),
        )

    # ------------------------------------------------------------------ #
    # Steering Updates (Learning from Failure)
    # ------------------------------------------------------------------ #

    def _generate_steering_updates(
        self,
        result: TrialResult,
        clue_manager: ClueManager,
        director: DirectorAI,
        agents: list[Agent],
        current_steering: dict[str, SteeringContext],
    ) -> dict[str, SteeringContext]:
        """
        Call Director to analyze failure and update steering contexts.
        Also load any existing exemplars from DB for this episode.
        """
        # Build a WorldState-like object for the analysis
        # (director.analyze_failure needs world for thread resolution)
        trial_world_facts = copy.deepcopy(self.world_facts)
        world = build_world_state(self.episode_config, trial_world_facts, agents)

        agent_updates = director.analyze_failure(
            result.interactions, clue_manager, world, agents,
        )

        # Update steering contexts with Director's recommendations
        for agent_id, update in agent_updates.items():
            if not isinstance(update, dict):
                continue
            if agent_id not in current_steering:
                current_steering[agent_id] = SteeringContext(agent_id=agent_id)

            ctx = current_steering[agent_id]
            ctx.tactical_goals = update.get("tactical_goals", ctx.tactical_goals)
            ctx.steering_prompt = update.get("steering_prompt", ctx.steering_prompt)

            logger.info(
                "  Steering update for %s: %d tactical goals",
                agent_id, len(ctx.tactical_goals),
            )

        # Load any exemplars from prior successful trials of this episode
        exemplars = db.load_trial_exemplars(self.base_episode_id)
        if exemplars:
            for exemplar in exemplars:
                agent_id = exemplar.get("discovering_agent", "")
                if agent_id and agent_id in current_steering:
                    text = exemplar.get("exemplar_text", "")
                    if text and text not in current_steering[agent_id].exemplar_actions:
                        current_steering[agent_id].exemplar_actions.append(text)
                        # Cap at 3 exemplars per agent
                        current_steering[agent_id].exemplar_actions = \
                            current_steering[agent_id].exemplar_actions[-3:]

        return current_steering

    # ------------------------------------------------------------------ #
    # Budget Allocation
    # ------------------------------------------------------------------ #

    def _allocate_trial_budget(self, trial_number: int, remaining: float) -> float:
        """
        Allocate budget for a trial.

        Strategy: base = total / max_trials, with rollover from unspent trials.
        Later trials get a 1.2x multiplier since they have better steering.
        Always reserve at least base * 0.5 for the final trial.
        """
        base_per_trial = self.total_budget / max(self.max_trials, 1)
        trials_left = self.max_trials - trial_number + 1

        if trials_left <= 1:
            return remaining

        # Reserve minimum for final trial
        reserved = base_per_trial * 0.5
        available = remaining - reserved

        # Later trials get slight bonus
        multiplier = 1.0 + (trial_number - 1) * 0.05  # 1.0, 1.05, 1.10, ...
        allocation = min(base_per_trial * multiplier, available)

        return max(allocation, base_per_trial * 0.5)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _persist_trial(
        self,
        trial_number: int,
        result: TrialResult,
        steering_contexts: dict[str, SteeringContext],
    ) -> None:
        """Save trial result and steering history to DB."""
        # Build steering summary for the trial record
        steering_summary = {
            aid: ctx.to_dict() for aid, ctx in steering_contexts.items()
            if ctx.steering_prompt or ctx.tactical_goals
        }

        result_dict = result.to_dict()
        result_dict["steering_used"] = steering_summary

        db.save_trial(self.base_episode_id, trial_number, result_dict)

        # Save per-agent steering history
        for agent_id, ctx in steering_contexts.items():
            if ctx.steering_prompt or ctx.tactical_goals or ctx.exemplar_actions:
                db.save_steering_history(
                    self.base_episode_id, trial_number, agent_id, ctx.to_dict(),
                )

    # ------------------------------------------------------------------ #
    # Output Writing
    # ------------------------------------------------------------------ #

    def _write_trial_output(
        self,
        trial_number: int,
        result: TrialResult,
        director: DirectorAI,
    ) -> None:
        """Write simulation JSON and debug log for a trial."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        trial_id = f"{self.base_episode_id}_trial{trial_number}"

        # Simulation JSON
        sim_path = self.output_dir / f"{trial_id}_simulation.json"
        with sim_path.open("w", encoding="utf-8") as f:
            json.dump({
                "episode_id": self.base_episode_id,
                "trial_number": trial_number,
                "success": result.success,
                "combined_score": result.combined_score,
                "clue_discovery_rate": result.clue_discovery_rate,
                "plot_resolution_rate": result.plot_resolution_rate,
                "beat_complete": result.beat_complete,
                "failure_reasons": result.failure_reasons,
                "total_turns": result.turn_count,
                "interactions": result.interactions,
                "budget_spent": result.budget_spent,
                "generated_at": datetime.utcnow().isoformat(),
            }, f, indent=2, ensure_ascii=False)

        # Debug log
        debug_path = self.output_dir / f"{trial_id}_debug.log"
        with debug_path.open("w", encoding="utf-8") as f:
            for entry in director.debug_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        status = "SUCCESS" if result.success else "FAILED"
        logger.info(
            "Trial %d [%s] output → %s (%d events)",
            trial_number, status, sim_path.name, len(director.debug_log),
        )
