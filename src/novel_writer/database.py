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
from typing import Any, Optional

DB_PATH = os.environ.get("DB_PATH", "data/simulation.db")
_TRACKING_CONTEXT: dict[str, Any] = {
    "run_id": None,
    "iteration": None,
    "phase": None,
    "episode_run_id": None,
}


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def set_tracking_context(
    *,
    run_id: Optional[str] = None,
    iteration: Optional[int] = None,
    phase: Optional[str] = None,
    episode_run_id: Optional[int] = None,
    reset: bool = False,
) -> None:
    """Set process-local tracking context used to tag/query rows."""
    global _TRACKING_CONTEXT
    if reset:
        _TRACKING_CONTEXT = {
            "run_id": None,
            "iteration": None,
            "phase": None,
            "episode_run_id": None,
        }
        return
    if run_id is not None:
        _TRACKING_CONTEXT["run_id"] = str(run_id)
    if iteration is not None:
        _TRACKING_CONTEXT["iteration"] = int(iteration)
    if phase is not None:
        _TRACKING_CONTEXT["phase"] = str(phase)
    if episode_run_id is not None:
        _TRACKING_CONTEXT["episode_run_id"] = int(episode_run_id)


def get_tracking_context() -> dict[str, Any]:
    return dict(_TRACKING_CONTEXT)


def configure_tracking_from_env(prefix: str = "NOVEL_") -> dict[str, Any]:
    """Populate tracking context from environment variables."""
    run_id = os.environ.get(f"{prefix}RUN_ID")
    iteration_raw = os.environ.get(f"{prefix}ITERATION")
    phase = os.environ.get(f"{prefix}PHASE")
    set_tracking_context(reset=True)
    if run_id:
        kwargs: dict[str, Any] = {"run_id": run_id}
        if iteration_raw not in (None, ""):
            try:
                kwargs["iteration"] = int(iteration_raw)
            except ValueError:
                pass
        if phase:
            kwargs["phase"] = phase
        set_tracking_context(**kwargs)
    return get_tracking_context()


def _tracking_values() -> tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    return (
        _TRACKING_CONTEXT.get("run_id"),
        _TRACKING_CONTEXT.get("iteration"),
        _TRACKING_CONTEXT.get("phase"),
        _TRACKING_CONTEXT.get("episode_run_id"),
    )


def _tracking_where(alias: str = "", include_episode_run: bool = False) -> tuple[str, list[Any]]:
    """Build WHERE clause fragment for current tracking context."""
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    run_id = _TRACKING_CONTEXT.get("run_id")
    iteration = _TRACKING_CONTEXT.get("iteration")
    episode_run_id = _TRACKING_CONTEXT.get("episode_run_id")
    if run_id:
        clauses.append(f"{prefix}run_id = ?")
        params.append(run_id)
    if iteration is not None:
        clauses.append(f"{prefix}iteration = ?")
        params.append(int(iteration))
    if include_episode_run and episode_run_id is not None:
        clauses.append(f"{prefix}episode_run_id = ?")
        params.append(int(episode_run_id))
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


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
            status          TEXT DEFAULT 'pending',
            run_id          TEXT,
            iteration       INTEGER,
            phase           TEXT
        );

        CREATE TABLE IF NOT EXISTS episode_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id       TEXT NOT NULL,
            run_id           TEXT,
            iteration        INTEGER,
            phase            TEXT,
            source           TEXT DEFAULT 'simulate',
            status           TEXT DEFAULT 'running',
            started_at       TEXT NOT NULL,
            ended_at         TEXT,
            metadata_json    TEXT DEFAULT '{}'
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
            run_id      TEXT,
            iteration   INTEGER,
            phase       TEXT,
            episode_run_id INTEGER,
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
            timestamp       TEXT NOT NULL,
            run_id          TEXT,
            iteration       INTEGER,
            phase           TEXT,
            episode_run_id  INTEGER
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent1_id   TEXT NOT NULL,
            agent2_id   TEXT NOT NULL,
            value       REAL NOT NULL,
            episode_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            updated_at  TEXT NOT NULL,
            run_id      TEXT,
            iteration   INTEGER,
            phase       TEXT,
            episode_run_id INTEGER,
            UNIQUE(agent1_id, agent2_id, episode_id, run_id, iteration)
        );

        CREATE TABLE IF NOT EXISTS world_states (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            state_json  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            run_id      TEXT,
            iteration   INTEGER,
            phase       TEXT,
            episode_run_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS persona_deltas (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id    TEXT NOT NULL,
            episode_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            changes_json TEXT NOT NULL,
            trigger     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            run_id      TEXT,
            iteration   INTEGER,
            phase       TEXT,
            episode_run_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS clues (
            id              TEXT NOT NULL,
            episode_id      TEXT NOT NULL,
            clue_content    TEXT NOT NULL,
            introduced_turn INTEGER,
            run_id          TEXT,
            iteration       INTEGER,
            phase           TEXT,
            episode_run_id  INTEGER,
            PRIMARY KEY(id, episode_id)
        );

        CREATE TABLE IF NOT EXISTS agent_knowledge (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id        TEXT NOT NULL,
            clue_id         TEXT NOT NULL,
            episode_id      TEXT NOT NULL,
            discovered_turn INTEGER NOT NULL,
            run_id          TEXT,
            iteration       INTEGER,
            phase           TEXT,
            episode_run_id  INTEGER,
            UNIQUE(agent_id, clue_id, episode_id, run_id, iteration)
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
        CREATE INDEX IF NOT EXISTS idx_interactions_tracking ON interactions(run_id, iteration, episode_id, turn);
        CREATE INDEX IF NOT EXISTS idx_interactions_speaker ON interactions(speaker_id);
        CREATE INDEX IF NOT EXISTS idx_emotions_agent ON emotions(agent_id);
        CREATE INDEX IF NOT EXISTS idx_emotions_tracking ON emotions(run_id, iteration, episode_id, agent_id, turn);
        CREATE INDEX IF NOT EXISTS idx_relationships_agents ON relationships(agent1_id, agent2_id);
        CREATE INDEX IF NOT EXISTS idx_episode_runs_tracking ON episode_runs(run_id, iteration, episode_id);
        CREATE INDEX IF NOT EXISTS idx_trials_episode ON trials(episode_id);
        CREATE INDEX IF NOT EXISTS idx_trial_exemplars_episode ON trial_exemplars(episode_id);
        CREATE INDEX IF NOT EXISTS idx_steering_history_episode ON steering_history(episode_id, agent_id);
    """)
    _ensure_tracking_schema(conn)
    conn.commit()
    conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_suffix: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_suffix}")


def _ensure_tracking_schema(conn: sqlite3.Connection) -> None:
    """Best-effort schema migration for existing DBs."""
    tracked_tables = {
        "episodes": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
        },
        "interactions": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
            "episode_run_id": "INTEGER",
        },
        "emotions": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
            "episode_run_id": "INTEGER",
        },
        "relationships": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
            "episode_run_id": "INTEGER",
        },
        "world_states": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
            "episode_run_id": "INTEGER",
        },
        "persona_deltas": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
            "episode_run_id": "INTEGER",
        },
        "clues": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
            "episode_run_id": "INTEGER",
        },
        "agent_knowledge": {
            "run_id": "TEXT",
            "iteration": "INTEGER",
            "phase": "TEXT",
            "episode_run_id": "INTEGER",
        },
    }
    for table, cols in tracked_tables.items():
        for name, ddl in cols.items():
            _ensure_column(conn, table, name, ddl)


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
    run_id, iteration, phase, _episode_run_id = _tracking_values()
    conn.execute("""
        INSERT OR REPLACE INTO episodes
        (id, beat_config_json, start_time, end_time, status, run_id, iteration, phase)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        episode_id, json.dumps(beat_config), start_time, end_time, status,
        run_id, iteration, phase,
    ))
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


def begin_episode_run(
    episode_id: str,
    *,
    source: str = "simulate",
    status: str = "running",
    metadata: Optional[dict[str, Any]] = None,
) -> int:
    conn = _connect()
    run_id, iteration, phase, _episode_run_id = _tracking_values()
    cur = conn.execute("""
        INSERT INTO episode_runs
        (episode_id, run_id, iteration, phase, source, status, started_at, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        episode_id, run_id, iteration, phase, source, status,
        datetime.utcnow().isoformat(), json.dumps(metadata or {}, ensure_ascii=False),
    ))
    conn.commit()
    run_row_id = int(cur.lastrowid)
    conn.close()
    set_tracking_context(episode_run_id=run_row_id)
    return run_row_id


def finish_episode_run(
    episode_run_id: int,
    *,
    status: str = "complete",
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    conn = _connect()
    if metadata is None:
        conn.execute(
            "UPDATE episode_runs SET status=?, ended_at=? WHERE id=?",
            (status, datetime.utcnow().isoformat(), episode_run_id),
        )
    else:
        conn.execute(
            "UPDATE episode_runs SET status=?, ended_at=?, metadata_json=? WHERE id=?",
            (
                status,
                datetime.utcnow().isoformat(),
                json.dumps(metadata, ensure_ascii=False),
                episode_run_id,
            ),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Interaction persistence
# ---------------------------------------------------------------------------

def save_interaction(interaction) -> None:
    conn = _connect()
    run_id, iteration, phase, episode_run_id = _tracking_values()
    conn.execute("""
        INSERT OR REPLACE INTO interactions
        (id, episode_id, turn, speaker_id, speaker_name, content,
         action_type, target_id, timestamp, metadata_json,
         run_id, iteration, phase, episode_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        interaction.id, interaction.episode_id, interaction.turn,
        interaction.speaker_id, interaction.speaker_name, interaction.content,
        interaction.action_type, interaction.target_id,
        interaction.timestamp.isoformat(),
        json.dumps(interaction.metadata),
        run_id, iteration, phase, episode_run_id,
    ))
    conn.commit()
    conn.close()


def load_episode_interactions(episode_id: str) -> list[dict]:
    conn = _connect()
    where_tracking, params_tracking = _tracking_where()
    rows = conn.execute(f"""
        SELECT * FROM interactions WHERE episode_id=?{where_tracking} ORDER BY turn, timestamp
    """, [episode_id, *params_tracking]).fetchall()
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
    where_tracking, params_tracking = _tracking_where()
    query += where_tracking
    params.extend(params_tracking)
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
    run_id, iteration, phase, episode_run_id = _tracking_values()
    conn.execute("""
        INSERT INTO emotions
        (agent_id, interaction_id, episode_id, turn, emotion_type, intensity, timestamp,
         run_id, iteration, phase, episode_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        agent_id, interaction_id, episode_id, turn, emotion_type, intensity,
        datetime.utcnow().isoformat(), run_id, iteration, phase, episode_run_id,
    ))
    conn.commit()
    conn.close()


def load_agent_emotions(agent_id: str, episode_id: str) -> list[dict]:
    conn = _connect()
    where_tracking, params_tracking = _tracking_where()
    rows = conn.execute(f"""
        SELECT * FROM emotions WHERE agent_id=? AND episode_id=?{where_tracking} ORDER BY turn
    """, [agent_id, episode_id, *params_tracking]).fetchall()
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
        where_tracking, params_tracking = _tracking_where()
        rows = conn.execute(f"""
            SELECT DISTINCT episode_id
            FROM emotions
            WHERE agent_id = ?{where_tracking}
        """, [agent_id, *params_tracking]).fetchall()

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
        where_tracking, params_tracking = _tracking_where()
        rows = conn.execute(f"""
            SELECT emotion_type, intensity
            FROM emotions
            WHERE agent_id = ?
              AND episode_id LIKE ?{where_tracking}
            ORDER BY turn DESC, timestamp DESC
        """, [agent_id, prev_pattern, *params_tracking]).fetchall()
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
# Cross-episode memory loaders
# ---------------------------------------------------------------------------

def _episode_number(episode_id: str) -> Optional[int]:
    match = re.match(r"^ep(\d+)(?:_|$)", str(episode_id))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_trial_episode_id(episode_id: str) -> bool:
    return bool(re.search(r"_trial\d+$", str(episode_id)))


def _completed_story_episode_ids(conn: sqlite3.Connection) -> list[str]:
    where_tracking, params_tracking = _tracking_where()
    rows = conn.execute(f"""
        SELECT id
        FROM episodes
        WHERE status = 'complete'
        {where_tracking}
    """, params_tracking).fetchall()
    return [str(r["id"]) for r in rows]


def _prior_story_episode_ids(conn: sqlite3.Connection, current_episode_id: str) -> list[str]:
    current_num = _episode_number(current_episode_id)
    if current_num is None:
        return []

    prior: list[tuple[int, str]] = []
    for eid in _completed_story_episode_ids(conn):
        if _is_trial_episode_id(eid):
            continue
        num = _episode_number(eid)
        if num is None or num >= current_num:
            continue
        prior.append((num, eid))

    prior.sort(key=lambda x: (x[0], x[1]))
    return [eid for _, eid in prior]


def load_previous_episode_relationships(agent_id: str, current_episode_id: str) -> dict[str, float]:
    """
    Load latest known relationship values for agent_id from all prior episodes.
    Returns {other_agent_id: value}.
    """
    conn = _connect()
    try:
        prior_ids = _prior_story_episode_ids(conn, current_episode_id)
        if not prior_ids:
            return {}

        placeholders = ",".join("?" for _ in prior_ids)
        where_tracking, params_tracking = _tracking_where()
        rows = conn.execute(f"""
            SELECT episode_id, agent2_id, value
            FROM relationships
            WHERE agent1_id = ?
              AND episode_id IN ({placeholders})
              {where_tracking}
        """, [agent_id, *prior_ids, *params_tracking]).fetchall()
    finally:
        conn.close()

    episode_order = {eid: idx for idx, eid in enumerate(prior_ids)}
    rel_rows = sorted(
        [dict(r) for r in rows],
        key=lambda r: episode_order.get(str(r.get("episode_id", "")), -1),
    )
    latest: dict[str, float] = {}
    for row in rel_rows:
        other = str(row.get("agent2_id", "")).strip()
        if not other:
            continue
        latest[other] = float(row.get("value", 0.0))
    return latest


def load_previous_episode_known_clues(agent_id: str, current_episode_id: str) -> set[str]:
    """
    Load all clue IDs previously discovered by this agent in prior episodes.
    """
    conn = _connect()
    try:
        prior_ids = _prior_story_episode_ids(conn, current_episode_id)
        if not prior_ids:
            return set()

        placeholders = ",".join("?" for _ in prior_ids)
        where_tracking, params_tracking = _tracking_where()
        rows = conn.execute(f"""
            SELECT clue_id
            FROM agent_knowledge
            WHERE agent_id = ?
              AND episode_id IN ({placeholders})
              {where_tracking}
        """, [agent_id, *prior_ids, *params_tracking]).fetchall()
    finally:
        conn.close()

    return {
        str(r["clue_id"]).strip()
        for r in rows
        if str(r["clue_id"]).strip()
    }


def load_previous_episode_persona_deltas(
    agent_id: str,
    current_episode_id: str,
    max_entries: int = 40,
) -> list[dict[str, Any]]:
    """
    Load persona deltas for this agent across prior episodes (oldest->newest).
    Each item contains: episode_id, turn, trigger, changes.
    """
    conn = _connect()
    try:
        prior_ids = _prior_story_episode_ids(conn, current_episode_id)
        if not prior_ids:
            return []

        placeholders = ",".join("?" for _ in prior_ids)
        where_tracking, params_tracking = _tracking_where()
        rows = conn.execute(f"""
            SELECT episode_id, turn, trigger, changes_json
            FROM persona_deltas
            WHERE agent_id = ?
              AND episode_id IN ({placeholders})
              {where_tracking}
            ORDER BY turn ASC, timestamp ASC
        """, [agent_id, *prior_ids, *params_tracking]).fetchall()
    finally:
        conn.close()

    episode_order = {eid: idx for idx, eid in enumerate(prior_ids)}
    sorted_rows = sorted(
        [dict(r) for r in rows],
        key=lambda r: (
            episode_order.get(str(r.get("episode_id", "")), -1),
            int(r.get("turn", 0)),
        ),
    )

    output: list[dict[str, Any]] = []
    for row in sorted_rows:
        try:
            changes = json.loads(str(row.get("changes_json", "{}")))
        except json.JSONDecodeError:
            changes = {}
        if not isinstance(changes, dict):
            continue
        output.append({
            "episode_id": str(row.get("episode_id", "")),
            "turn": int(row.get("turn", 0)),
            "trigger": str(row.get("trigger", "")),
            "changes": changes,
        })

    if max_entries > 0 and len(output) > max_entries:
        return output[-max_entries:]
    return output


def load_episode_history_context(
    current_episode_id: str,
    max_episodes: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Load compact, ordered story context from all prior completed episodes.
    Returns list of dicts:
      {id, number, date, location, summary, clue_ids}
    """
    conn = _connect()
    try:
        prior_ids = _prior_story_episode_ids(conn, current_episode_id)
        if not prior_ids:
            return []

        if max_episodes is not None and max_episodes > 0:
            prior_ids = prior_ids[-max_episodes:]

        placeholders = ",".join("?" for _ in prior_ids)
        where_tracking, params_tracking = _tracking_where()
        rows = conn.execute(f"""
            SELECT id, beat_config_json
            FROM episodes
            WHERE id IN ({placeholders})
              {where_tracking}
        """, [*prior_ids, *params_tracking]).fetchall()
    finally:
        conn.close()

    payload_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        eid = str(row["id"])
        try:
            payload = json.loads(str(row["beat_config_json"]))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload_by_id[eid] = payload

    result: list[dict[str, Any]] = []
    for eid in prior_ids:
        cfg = payload_by_id.get(eid, {})
        clues = cfg.get("introduced_clues", [])
        clue_ids: list[str] = []
        if isinstance(clues, list):
            for clue in clues:
                if isinstance(clue, dict):
                    cid = str(clue.get("id", "")).strip()
                    if cid:
                        clue_ids.append(cid)
        result.append({
            "id": eid,
            "number": _episode_number(eid) or 0,
            "date": str(cfg.get("date", "")).strip(),
            "location": str(cfg.get("location", "")).strip(),
            "summary": str(cfg.get("summary", "")).strip(),
            "clue_ids": clue_ids,
        })
    return result


# ---------------------------------------------------------------------------
# Relationship persistence
# ---------------------------------------------------------------------------

def save_relationship(agent1_id: str, agent2_id: str, value: float,
                      episode_id: str, turn: int) -> None:
    conn = _connect()
    run_id, iteration, phase, episode_run_id = _tracking_values()
    conn.execute("""
        INSERT OR REPLACE INTO relationships
        (agent1_id, agent2_id, value, episode_id, turn, updated_at,
         run_id, iteration, phase, episode_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        agent1_id, agent2_id, value, episode_id, turn,
        datetime.utcnow().isoformat(), run_id, iteration, phase, episode_run_id,
    ))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# World state persistence
# ---------------------------------------------------------------------------

def save_world_state(episode_id: str, turn: int, state: dict) -> None:
    conn = _connect()
    run_id, iteration, phase, episode_run_id = _tracking_values()
    conn.execute("""
        INSERT INTO world_states
        (episode_id, turn, state_json, timestamp, run_id, iteration, phase, episode_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        episode_id, turn, json.dumps(state), datetime.utcnow().isoformat(),
        run_id, iteration, phase, episode_run_id,
    ))
    conn.commit()
    conn.close()


def load_world_states(episode_id: str) -> list[dict]:
    conn = _connect()
    where_tracking, params_tracking = _tracking_where()
    rows = conn.execute(f"""
        SELECT * FROM world_states WHERE episode_id=?{where_tracking} ORDER BY turn
    """, [episode_id, *params_tracking]).fetchall()
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
    where_tracking, params_tracking = _tracking_where("ws")
    row = conn.execute(f"""
        SELECT ws.state_json
        FROM world_states ws
        JOIN episodes e ON e.id = ws.episode_id
        WHERE e.status = 'complete'
          AND e.id < ?
          {where_tracking}
        ORDER BY e.id DESC, ws.turn DESC
        LIMIT 1
    """, [current_episode_id, *params_tracking]).fetchone()
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
    run_id, iteration, phase, episode_run_id = _tracking_values()
    conn.execute("""
        INSERT INTO persona_deltas
        (agent_id, episode_id, turn, changes_json, trigger, timestamp,
         run_id, iteration, phase, episode_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        agent_id, episode_id, turn, json.dumps(changes), trigger,
        datetime.utcnow().isoformat(), run_id, iteration, phase, episode_run_id,
    ))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Clue persistence
# ---------------------------------------------------------------------------

def upsert_clue(clue_id: str, episode_id: str, content: str,
                introduced_turn: Optional[int] = None) -> None:
    conn = _connect()
    run_id, iteration, phase, episode_run_id = _tracking_values()
    conn.execute("""
        INSERT OR REPLACE INTO clues
        (id, episode_id, clue_content, introduced_turn, run_id, iteration, phase, episode_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (clue_id, episode_id, content, introduced_turn, run_id, iteration, phase, episode_run_id))
    conn.commit()
    conn.close()


def save_agent_knowledge(agent_id: str, clue_id: str,
                         episode_id: str, discovered_turn: int) -> None:
    conn = _connect()
    run_id, iteration, phase, episode_run_id = _tracking_values()
    conn.execute("""
        INSERT OR IGNORE INTO agent_knowledge
        (agent_id, clue_id, episode_id, discovered_turn, run_id, iteration, phase, episode_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (agent_id, clue_id, episode_id, discovered_turn, run_id, iteration, phase, episode_run_id))
    conn.commit()
    conn.close()


def load_episode_clues(episode_id: str) -> list[dict]:
    conn = _connect()
    where_tracking, params_tracking = _tracking_where()
    rows = conn.execute(
        f"SELECT * FROM clues WHERE episode_id=?{where_tracking}", [episode_id, *params_tracking]
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
