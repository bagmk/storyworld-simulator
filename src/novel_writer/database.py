"""
Database layer for the AI Story Simulation Engine.
Uses SQLite by default; switchable to PostgreSQL via DB_URL env var.
Stores all interactions, agent states, emotions, relationships, and clues
with full persistence and no truncation.
"""

import sqlite3
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "data/simulation.db")


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            persona_json TEXT NOT NULL,
            role        TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id              TEXT PRIMARY KEY,
            beat_config_json TEXT NOT NULL,
            start_time      TEXT,
            end_time        TEXT,
            status          TEXT DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id          TEXT PRIMARY KEY,
            episode_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            speaker_id  TEXT NOT NULL,
            speaker_name TEXT NOT NULL,
            content     TEXT NOT NULL,
            action_type TEXT DEFAULT 'dialogue',
            target_id   TEXT,
            timestamp   TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            FOREIGN KEY (episode_id) REFERENCES episodes(id)
        );

        CREATE TABLE IF NOT EXISTS emotions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id        TEXT NOT NULL,
            interaction_id  TEXT,
            episode_id      TEXT NOT NULL,
            turn            INTEGER,
            emotion_type    TEXT NOT NULL,
            intensity       REAL NOT NULL,
            timestamp       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent1_id   TEXT NOT NULL,
            agent2_id   TEXT NOT NULL,
            value       REAL NOT NULL,
            episode_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            updated_at  TEXT NOT NULL,
            UNIQUE(agent1_id, agent2_id, episode_id)
        );

        CREATE TABLE IF NOT EXISTS world_states (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            state_json  TEXT NOT NULL,
            timestamp   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS persona_deltas (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id    TEXT NOT NULL,
            episode_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            changes_json TEXT NOT NULL,
            trigger     TEXT NOT NULL,
            timestamp   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS clues (
            id              TEXT NOT NULL,
            episode_id      TEXT NOT NULL,
            clue_content    TEXT NOT NULL,
            introduced_turn INTEGER,
            PRIMARY KEY(id, episode_id)
        );

        CREATE TABLE IF NOT EXISTS agent_knowledge (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id        TEXT NOT NULL,
            clue_id         TEXT NOT NULL,
            episode_id      TEXT NOT NULL,
            discovered_turn INTEGER NOT NULL,
            UNIQUE(agent_id, clue_id, episode_id)
        );

        CREATE TABLE IF NOT EXISTS trials (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id      TEXT NOT NULL,
            trial_number    INTEGER NOT NULL,
            success         INTEGER NOT NULL DEFAULT 0,
            clue_discovery_rate  REAL,
            plot_resolution_rate REAL,
            beat_complete   INTEGER DEFAULT 0,
            combined_score  REAL,
            failure_reasons_json TEXT DEFAULT '[]',
            steering_used_json   TEXT DEFAULT '{}',
            budget_spent    REAL,
            turn_count      INTEGER,
            timestamp       TEXT NOT NULL,
            UNIQUE(episode_id, trial_number)
        );

        CREATE TABLE IF NOT EXISTS trial_exemplars (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id      TEXT NOT NULL,
            trial_number    INTEGER NOT NULL,
            clue_id         TEXT NOT NULL,
            discovering_agent TEXT,
            discovery_turn  INTEGER,
            exemplar_text   TEXT NOT NULL,
            exemplar_json   TEXT NOT NULL,
            timestamp       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS steering_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id      TEXT NOT NULL,
            trial_number    INTEGER NOT NULL,
            agent_id        TEXT NOT NULL,
            tactical_goals_json TEXT DEFAULT '[]',
            steering_prompt TEXT DEFAULT '',
            exemplar_actions_json TEXT DEFAULT '[]',
            timestamp       TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_interactions_episode ON interactions(episode_id);
        CREATE INDEX IF NOT EXISTS idx_interactions_speaker ON interactions(speaker_id);
        CREATE INDEX IF NOT EXISTS idx_emotions_agent ON emotions(agent_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_agents ON relationships(agent1_id, agent2_id);
        CREATE INDEX IF NOT EXISTS idx_trials_episode ON trials(episode_id);
        CREATE INDEX IF NOT EXISTS idx_trial_exemplars_episode ON trial_exemplars(episode_id);
        CREATE INDEX IF NOT EXISTS idx_steering_history_episode ON steering_history(episode_id, agent_id);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Agent persistence
# ---------------------------------------------------------------------------

def upsert_agent(agent) -> None:
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO agents (id, name, persona_json, role, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (agent.id, agent.name, json.dumps(agent.persona), agent.role,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def load_agent_row(agent_id: str) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM agents WHERE id=?", (agent_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Episode persistence
# ---------------------------------------------------------------------------

def upsert_episode(episode_id: str, beat_config: dict,
                   status: str = "pending",
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None) -> None:
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO episodes (id, beat_config_json, start_time, end_time, status)
        VALUES (?, ?, ?, ?, ?)
    """, (episode_id, json.dumps(beat_config), start_time, end_time, status))
    conn.commit()
    conn.close()


def update_episode_status(episode_id: str, status: str,
                          end_time: Optional[str] = None) -> None:
    conn = _connect()
    if end_time:
        conn.execute("UPDATE episodes SET status=?, end_time=? WHERE id=?",
                     (status, end_time, episode_id))
    else:
        conn.execute("UPDATE episodes SET status=? WHERE id=?",
                     (status, episode_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Interaction persistence
# ---------------------------------------------------------------------------

def save_interaction(interaction) -> None:
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO interactions
        (id, episode_id, turn, speaker_id, speaker_name, content,
         action_type, target_id, timestamp, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        interaction.id, interaction.episode_id, interaction.turn,
        interaction.speaker_id, interaction.speaker_name, interaction.content,
        interaction.action_type, interaction.target_id,
        interaction.timestamp.isoformat(),
        json.dumps(interaction.metadata),
    ))
    conn.commit()
    conn.close()


def load_episode_interactions(episode_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM interactions WHERE episode_id=? ORDER BY turn, timestamp
    """, (episode_id,)).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        result.append(d)
    return result


def load_agent_interactions(agent_id: str, episode_id: Optional[str] = None,
                             limit: Optional[int] = None) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM interactions WHERE speaker_id=?"
    params: list = [agent_id]
    if episode_id:
        query += " AND episode_id=?"
        params.append(episode_id)
    query += " ORDER BY turn, timestamp"
    if limit:
        query += f" LIMIT {limit}"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Emotion persistence
# ---------------------------------------------------------------------------

def save_emotion(agent_id: str, episode_id: str, turn: int,
                 emotion_type: str, intensity: float,
                 interaction_id: Optional[str] = None) -> None:
    conn = _connect()
    conn.execute("""
        INSERT INTO emotions
        (agent_id, interaction_id, episode_id, turn, emotion_type, intensity, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (agent_id, interaction_id, episode_id, turn, emotion_type, intensity,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def load_agent_emotions(agent_id: str, episode_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM emotions WHERE agent_id=? AND episode_id=? ORDER BY turn
    """, (agent_id, episode_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_previous_episode_final_emotions(agent_id: str, current_episode_id: str) -> dict[str, float]:
    """
    Load the final emotional state from the previous episode to maintain continuity.

    Returns a dict of {emotion_type: intensity} from the last turn of the previous episode.
    If no previous episode exists, returns empty dict (fresh start).
    """
    conn = _connect()

    # Extract episode number from current_episode_id (e.g., "ep05" from "ep05_unexpected_visitors")
    # Assumes format: epXX_name or epXX_name_trialN
    try:
        parts = current_episode_id.split('_')
        ep_prefix = parts[0]  # e.g., "ep05"
        if not ep_prefix.startswith("ep"):
            conn.close()
            return {}  # Can't parse episode number

        current_num = int(ep_prefix[2:])  # e.g., 5
        if current_num <= 1:
            conn.close()
            return {}  # No previous episode

        # Find the latest prior episode number that actually has emotion rows
        # for this agent. This allows continuity even if episodes are skipped.
        rows = conn.execute("""
            SELECT DISTINCT episode_id
            FROM emotions
            WHERE agent_id = ?
        """, (agent_id,)).fetchall()

        prev_num: Optional[int] = None
        for row in rows:
            eid = str(row["episode_id"])
            m = re.match(r"^ep(\d+)_", eid)
            if not m:
                continue
            ep_num = int(m.group(1))
            if ep_num < current_num and (prev_num is None or ep_num > prev_num):
                prev_num = ep_num

        if prev_num is None:
            conn.close()
            return {}  # No prior episode rows found

        prev_pattern = f"ep{prev_num:02d}_%"

        # Pull descending by turn/timestamp and keep first row per emotion_type.
        # This preserves each emotion's most recent value instead of only
        # emotions present at the single final turn.
        rows = conn.execute("""
            SELECT emotion_type, intensity
            FROM emotions
            WHERE agent_id = ?
              AND episode_id LIKE ?
            ORDER BY turn DESC, timestamp DESC
        """, (agent_id, prev_pattern)).fetchall()
        conn.close()

        latest_by_emotion: dict[str, float] = {}
        for row in rows:
            emo = str(row["emotion_type"])
            if emo not in latest_by_emotion:
                latest_by_emotion[emo] = float(row["intensity"])

        return latest_by_emotion
    except (ValueError, IndexError):
        conn.close()
        return {}  # Malformed episode_id


# ---------------------------------------------------------------------------
# Relationship persistence
# ---------------------------------------------------------------------------

def save_relationship(agent1_id: str, agent2_id: str, value: float,
                      episode_id: str, turn: int) -> None:
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO relationships
        (agent1_id, agent2_id, value, episode_id, turn, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (agent1_id, agent2_id, value, episode_id, turn,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# World state persistence
# ---------------------------------------------------------------------------

def save_world_state(episode_id: str, turn: int, state: dict) -> None:
    conn = _connect()
    conn.execute("""
        INSERT INTO world_states (episode_id, turn, state_json, timestamp)
        VALUES (?, ?, ?, ?)
    """, (episode_id, turn, json.dumps(state), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def load_world_states(episode_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM world_states WHERE episode_id=? ORDER BY turn
    """, (episode_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_previous_episode_final_state(current_episode_id: str) -> Optional[dict]:
    """
    Load the last world state snapshot of the most recently completed episode
    *before* the current one (by episode ID lexicographic order).

    Returns the parsed state dict or None.
    """
    conn = _connect()
    # Find the most recent completed episode that comes before this one
    row = conn.execute("""
        SELECT ws.state_json
        FROM world_states ws
        JOIN episodes e ON e.id = ws.episode_id
        WHERE e.status = 'complete'
          AND e.id < ?
        ORDER BY e.id DESC, ws.turn DESC
        LIMIT 1
    """, (current_episode_id,)).fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row["state_json"])
        except (json.JSONDecodeError, KeyError):
            pass
    return None


# ---------------------------------------------------------------------------
# Persona delta persistence
# ---------------------------------------------------------------------------

def save_persona_delta(agent_id: str, episode_id: str, turn: int,
                       changes: dict, trigger: str) -> None:
    conn = _connect()
    conn.execute("""
        INSERT INTO persona_deltas
        (agent_id, episode_id, turn, changes_json, trigger, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (agent_id, episode_id, turn, json.dumps(changes), trigger,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Clue persistence
# ---------------------------------------------------------------------------

def upsert_clue(clue_id: str, episode_id: str, content: str,
                introduced_turn: Optional[int] = None) -> None:
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO clues (id, episode_id, clue_content, introduced_turn)
        VALUES (?, ?, ?, ?)
    """, (clue_id, episode_id, content, introduced_turn))
    conn.commit()
    conn.close()


def save_agent_knowledge(agent_id: str, clue_id: str,
                         episode_id: str, discovered_turn: int) -> None:
    conn = _connect()
    conn.execute("""
        INSERT OR IGNORE INTO agent_knowledge
        (agent_id, clue_id, episode_id, discovered_turn)
        VALUES (?, ?, ?, ?)
    """, (agent_id, clue_id, episode_id, discovered_turn))
    conn.commit()
    conn.close()


def load_episode_clues(episode_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM clues WHERE episode_id=?", (episode_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Trial persistence (Trial-and-Learn)
# ---------------------------------------------------------------------------

def save_trial(episode_id: str, trial_number: int, result_dict: dict) -> None:
    """Save a trial result record."""
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO trials
        (episode_id, trial_number, success, clue_discovery_rate,
         plot_resolution_rate, beat_complete, combined_score,
         failure_reasons_json, steering_used_json, budget_spent,
         turn_count, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        episode_id, trial_number,
        1 if result_dict.get("success") else 0,
        result_dict.get("clue_discovery_rate", 0.0),
        result_dict.get("plot_resolution_rate", 0.0),
        1 if result_dict.get("beat_complete") else 0,
        result_dict.get("combined_score", 0.0),
        json.dumps(result_dict.get("failure_reasons", [])),
        json.dumps(result_dict.get("steering_used", {})),
        result_dict.get("budget_spent", 0.0),
        result_dict.get("turn_count", 0),
        datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


def save_trial_exemplar(episode_id: str, trial_number: int,
                         exemplar: dict) -> None:
    """Save a single success pattern exemplar."""
    conn = _connect()
    conn.execute("""
        INSERT INTO trial_exemplars
        (episode_id, trial_number, clue_id, discovering_agent,
         discovery_turn, exemplar_text, exemplar_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        episode_id, trial_number,
        exemplar.get("clue_id", ""),
        exemplar.get("discovering_agent", ""),
        exemplar.get("discovery_turn", 0),
        exemplar.get("exemplar_text", ""),
        json.dumps(exemplar.get("exemplar_sequence", [])),
        datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


def save_steering_history(episode_id: str, trial_number: int,
                           agent_id: str, steering_dict: dict) -> None:
    """Save the steering context used for a specific agent in a trial."""
    conn = _connect()
    conn.execute("""
        INSERT INTO steering_history
        (episode_id, trial_number, agent_id, tactical_goals_json,
         steering_prompt, exemplar_actions_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        episode_id, trial_number, agent_id,
        json.dumps(steering_dict.get("tactical_goals", [])),
        steering_dict.get("steering_prompt", ""),
        json.dumps(steering_dict.get("exemplar_actions", [])),
        datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


def load_trial_exemplars(episode_id: str) -> list[dict]:
    """Load all exemplars for an episode (across all trials)."""
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM trial_exemplars WHERE episode_id=?
        ORDER BY trial_number, discovery_turn
    """, (episode_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_trials(episode_id: str) -> list[dict]:
    """Load all trial results for an episode."""
    conn = _connect()
    rows = conn.execute("""
        SELECT * FROM trials WHERE episode_id=? ORDER BY trial_number
    """, (episode_id,)).fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["failure_reasons"] = json.loads(d.pop("failure_reasons_json", "[]"))
        d["steering_used"] = json.loads(d.pop("steering_used_json", "{}"))
        result.append(d)
    return result
