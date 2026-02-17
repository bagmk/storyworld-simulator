#!/usr/bin/env python3
"""
Rebuild canonical episode interactions/emotions from trial simulation outputs.

Rules:
- Per episode base ID, pick one trial simulation file.
- Prefer winning_trial from trial_summary.json.
- If no winner, choose the trial with highest combined_score from summary.
- If summary is missing, choose the highest trial number found.

Then rewrite DB rows for that episode_id in:
- interactions
- emotions
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import uuid
from pathlib import Path


TRIAL_SIM_RE = re.compile(r"^(?P<base>.+)_trial(?P<trial>\d+)_simulation\.json$")


def choose_trial_file(output_dir: Path, episode_base: str) -> Path | None:
    summary_path = output_dir / f"{episode_base}_trial_summary.json"
    trial_to_use: int | None = None

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        winning = summary.get("winning_trial")
        if isinstance(winning, int) and winning > 0:
            trial_to_use = winning
        else:
            best_score = -1.0
            best_trial = None
            for t in summary.get("trials", []):
                if not isinstance(t, dict):
                    continue
                tn = t.get("trial_number")
                sc = t.get("combined_score", -1)
                if isinstance(tn, int) and isinstance(sc, (int, float)) and sc > best_score:
                    best_score = float(sc)
                    best_trial = tn
            if isinstance(best_trial, int):
                trial_to_use = best_trial

    if trial_to_use is not None:
        p = output_dir / f"{episode_base}_trial{trial_to_use}_simulation.json"
        if p.exists():
            return p

    # Fallback: highest trial number present
    cands: list[tuple[int, Path]] = []
    for p in output_dir.glob(f"{episode_base}_trial*_simulation.json"):
        m = TRIAL_SIM_RE.match(p.name)
        if not m:
            continue
        cands.append((int(m.group("trial")), p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def extract_episode_bases(output_dir: Path) -> list[str]:
    bases: set[str] = set()
    for p in output_dir.glob("*_trial*_simulation.json"):
        m = TRIAL_SIM_RE.match(p.name)
        if m:
            bases.add(m.group("base"))
    return sorted(bases)


def rebuild_episode(conn: sqlite3.Connection, episode_base: str, sim_path: Path) -> tuple[int, int]:
    data = json.loads(sim_path.read_text(encoding="utf-8"))
    interactions = data.get("interactions", [])
    if not isinstance(interactions, list):
        interactions = []

    cur = conn.cursor()
    cur.execute("DELETE FROM emotions WHERE episode_id = ?", (episode_base,))
    cur.execute("DELETE FROM interactions WHERE episode_id = ?", (episode_base,))

    inserted_ix = 0
    inserted_em = 0

    for i, ix in enumerate(interactions, start=1):
        if not isinstance(ix, dict):
            continue
        ix_id = str(ix.get("id") or uuid.uuid4())
        turn = int(ix.get("turn") or i)
        speaker_id = str(ix.get("speaker_id") or ix.get("agent_id") or "unknown")
        speaker_name = str(ix.get("speaker_name") or ix.get("agent_name") or speaker_id)
        content = str(ix.get("content") or ix.get("response") or "")
        action_type = str(ix.get("action_type") or "dialogue")
        target_id = ix.get("target_id")
        timestamp = str(ix.get("timestamp") or "")
        metadata = ix.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        cur.execute(
            """
            INSERT INTO interactions
            (id, episode_id, turn, speaker_id, speaker_name, content,
             action_type, target_id, timestamp, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ix_id,
                episode_base,
                turn,
                speaker_id,
                speaker_name,
                content,
                action_type,
                target_id,
                timestamp,
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        inserted_ix += 1

        emotions = metadata.get("emotions", {})
        if isinstance(emotions, dict):
            for emotion_type, intensity in emotions.items():
                if not isinstance(emotion_type, str):
                    continue
                try:
                    val = float(intensity)
                except (TypeError, ValueError):
                    continue
                if val <= 0:
                    continue
                val = max(0.0, min(1.0, val))
                cur.execute(
                    """
                    INSERT INTO emotions
                    (agent_id, interaction_id, episode_id, turn, emotion_type, intensity, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (speaker_id, ix_id, episode_base, turn, emotion_type, val, timestamp),
                )
                inserted_em += 1

    conn.commit()
    return inserted_ix, inserted_em


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild episode interactions/emotions from trial simulation JSONs.")
    p.add_argument("--db", default="data/simulation.db", help="SQLite DB path")
    p.add_argument("--output", default="output", help="Directory containing *_trial*_simulation.json")
    p.add_argument("--episodes", nargs="*", default=None, help="Optional episode base IDs to rebuild")
    args = p.parse_args()

    output_dir = Path(args.output)
    if not output_dir.exists():
        raise SystemExit(f"Output dir not found: {output_dir}")

    episodes = args.episodes or extract_episode_bases(output_dir)
    if not episodes:
        raise SystemExit("No trial simulation files found.")

    conn = sqlite3.connect(args.db)
    total_ix = 0
    total_em = 0
    done = 0

    for ep in episodes:
        sim = choose_trial_file(output_dir, ep)
        if sim is None:
            print(f"[SKIP] {ep}: no simulation file")
            continue
        ix, em = rebuild_episode(conn, ep, sim)
        total_ix += ix
        total_em += em
        done += 1
        print(f"[OK] {ep}: {sim.name} -> interactions={ix}, emotions={em}")

    conn.close()
    print(f"Done. episodes={done}, interactions={total_ix}, emotions={total_em}")


if __name__ == "__main__":
    main()
