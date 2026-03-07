#!/usr/bin/env python3
"""
Bandit-style code policy trainer (prompt-independent).

Optimizes runtime behavior parameters consumed by code:
- scene target bias
- prose history cap
- director fallback cast size
- generation temperatures

Reward is computed from benchmark outputs (coherence + quality).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.novel_writer.rl_policy import load_policy, save_policy


ACTION_SPACE: dict[str, list[Any]] = {
    "scene_target_bias": [-1, 0, 1],
    # Fixed by design (user requested):
    # - prose_history_max_episodes pinned to max
    # - director_fallback_cast_size handled conditionally in DirectorAI
    "distiller_temperature": [0.2, 0.3, 0.4],
    "prose_scene_temperature": [0.65, 0.75, 0.85],
    "prose_polish_temperature": [0.25, 0.35, 0.45],
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def compute_reward(coh_report: dict[str, Any], quality_report: dict[str, Any]) -> float:
    base = float(coh_report.get("series_coherence_score_with_full_text", coh_report.get("series_coherence_score", 0.0)))
    full_text = float((coh_report.get("full_text_coherence_benchmark") or {}).get("score", 0.0))
    q_avg = float((quality_report.get("summary") or {}).get("avg_overall_score", 0.0))
    quality_eps = quality_report.get("episodes", []) or []
    low_tail = min((float(e.get("overall_score", 0.0)) for e in quality_eps), default=0.0)
    reward = (base * 0.45) + (full_text * 0.15) + (q_avg * 0.25) + (low_tail * 0.15)
    return round(reward, 4)


def mutate_policy(base: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    cand = dict(base)
    changed: dict[str, Any] = {}
    # mutate 1-3 knobs per round
    keys = list(ACTION_SPACE.keys())
    rng.shuffle(keys)
    num = rng.randint(1, 3)
    for key in keys[:num]:
        options = ACTION_SPACE[key]
        current = cand.get(key)
        choice = rng.choice(options)
        # avoid no-op if possible
        if len(options) > 1 and current in options:
            tries = 0
            while choice == current and tries < 5:
                choice = rng.choice(options)
                tries += 1
        cand[key] = choice
        changed[key] = choice
    return cand, changed


def run_cmd(cmd: str, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None})
    proc = subprocess.run(cmd, shell=True, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def write_status(
    path: Path,
    *,
    trainer_run_id: str,
    phase: str,
    total_rounds: int,
    current_round: int,
    best_reward: float | None,
    last_reward: float | None,
    history: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "trainer_run_id": trainer_run_id,
        "phase": phase,
        "total_rounds": total_rounds,
        "current_round": current_round,
        "best_reward": best_reward,
        "last_reward": last_reward,
        "updated_at": datetime.utcnow().isoformat(),
        "history_len": len(history),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Train code runtime policy from benchmark rewards")
    p.add_argument("--policy-path", default="data/rl_policy.json")
    p.add_argument("--coherence-report", default="reports/character_coherence_ep01_ep05_regen_20260221_v2.json")
    p.add_argument("--quality-report", default="reports/quality_ep01_ep05_regen_20260221.json")
    p.add_argument("--evaluate-cmd", default="", help="Optional shell command to regenerate/evaluate before scoring")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epsilon", type=float, default=0.35, help="Exploration probability")
    p.add_argument("--history-out", default="reports/rl_code_policy_history.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    policy_path = str(args.policy_path)
    current = load_policy(policy_path)
    trainer_run_id = f"rl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{args.seed}"
    status_path = (ROOT / args.history_out).with_suffix(".status.json")
    trainer_start = time.time()

    print("=" * 80)
    print("[trainer] RL code policy training start")
    print(f"[trainer] trainer_run_id={trainer_run_id}")
    print(f"[trainer] rounds={args.rounds} epsilon={args.epsilon} seed={args.seed}")
    print(f"[trainer] policy_path={args.policy_path}")
    print(f"[trainer] coherence_report={args.coherence_report}")
    print(f"[trainer] quality_report={args.quality_report}")
    print(f"[trainer] history_out={args.history_out}")
    print(f"[trainer] status_out={status_path.relative_to(ROOT)}")
    print(f"[trainer] evaluate_cmd={args.evaluate_cmd or '(none)'}")
    print("=" * 80)

    coh_path = ROOT / args.coherence_report
    q_path = ROOT / args.quality_report
    write_status(
        status_path,
        trainer_run_id=trainer_run_id,
        phase="starting",
        total_rounds=args.rounds,
        current_round=0,
        best_reward=None,
        last_reward=None,
        history=[],
        extra={"message": "before baseline evaluate"},
    )
    if args.evaluate_cmd:
        print(f"[trainer] baseline evaluate | run_id={trainer_run_id} iteration=0")
        run_cmd(args.evaluate_cmd, {
            "NOVEL_RUN_ID": trainer_run_id,
            "NOVEL_ITERATION": "0",
            "NOVEL_PHASE": "rl_eval",
            "RL_TRAINER_ROUND": "0",
        })
    best_reward = compute_reward(load_json(coh_path), load_json(q_path))
    best_policy = dict(current)
    history: list[dict[str, Any]] = [{
        "round": 0,
        "reward": best_reward,
        "policy": best_policy,
        "accepted": True,
        "changes": {},
    }]
    write_status(
        status_path,
        trainer_run_id=trainer_run_id,
        phase="baseline_complete",
        total_rounds=args.rounds,
        current_round=0,
        best_reward=best_reward,
        last_reward=best_reward,
        history=history,
        extra={"elapsed_seconds": round(time.time() - trainer_start, 1)},
    )
    print(f"[trainer] baseline reward={best_reward:.4f}")

    for i in range(1, args.rounds + 1):
        round_start = time.time()
        explore = rng.random() < args.epsilon
        if explore:
            cand, changes = mutate_policy(best_policy, rng)
        else:
            cand, changes = mutate_policy(current, rng)
        save_policy(cand, policy_path)
        write_status(
            status_path,
            trainer_run_id=trainer_run_id,
            phase="evaluating_round",
            total_rounds=args.rounds,
            current_round=i,
            best_reward=best_reward,
            last_reward=None,
            history=history,
            extra={
                "explore": explore,
                "candidate_changes": changes,
                "progress_ratio": round(i / max(args.rounds, 1), 4),
            },
        )

        if args.evaluate_cmd:
            print(f"[trainer] evaluate round={i} | run_id={trainer_run_id} explore={explore}")
            run_cmd(args.evaluate_cmd, {
                "NOVEL_RUN_ID": trainer_run_id,
                "NOVEL_ITERATION": str(i),
                "NOVEL_PHASE": "rl_eval",
                "RL_TRAINER_ROUND": str(i),
            })

        reward = compute_reward(load_json(coh_path), load_json(q_path))
        accepted = reward >= best_reward
        if accepted:
            best_reward = reward
            best_policy = dict(cand)
            current = dict(cand)
        else:
            # revert to best known policy
            save_policy(best_policy, policy_path)
            current = dict(best_policy)

        history.append({
            "round": i,
            "reward": reward,
            "best_reward": best_reward,
            "accepted": accepted,
            "explore": explore,
            "changes": changes,
            "candidate_policy": cand,
        })
        print(
            f"round={i} reward={reward:.4f} best={best_reward:.4f} "
            f"{'ACCEPT' if accepted else 'REJECT'} changes={changes} "
            f"elapsed={time.time() - round_start:.1f}s total={time.time() - trainer_start:.1f}s "
            f"progress={i}/{args.rounds}"
        )
        hist_path = ROOT / args.history_out
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(
            json.dumps(
                {
                    "trainer_run_id": trainer_run_id,
                    "history": history,
                    "best_policy": best_policy,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_status(
            status_path,
            trainer_run_id=trainer_run_id,
            phase="round_complete",
            total_rounds=args.rounds,
            current_round=i,
            best_reward=best_reward,
            last_reward=reward,
            history=history,
            extra={
                "accepted": accepted,
                "explore": explore,
                "round_elapsed_seconds": round(time.time() - round_start, 1),
                "total_elapsed_seconds": round(time.time() - trainer_start, 1),
                "progress_ratio": round(i / max(args.rounds, 1), 4),
            },
        )

    hist_path = ROOT / args.history_out
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(
        json.dumps(
            {"trainer_run_id": trainer_run_id, "history": history, "best_policy": best_policy},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_policy(best_policy, policy_path)
    write_status(
        status_path,
        trainer_run_id=trainer_run_id,
        phase="complete",
        total_rounds=args.rounds,
        current_round=args.rounds,
        best_reward=best_reward,
        last_reward=history[-1]["reward"] if history else None,
        history=history,
        extra={"total_elapsed_seconds": round(time.time() - trainer_start, 1)},
    )
    print(f"Saved policy: {policy_path}")
    print(f"History: {hist_path}")
    print(f"Status: {status_path}")
    print(f"Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
