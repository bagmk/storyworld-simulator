#!/usr/bin/env python3
"""Ingest benchmark/report JSON files into a SQLite database for analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "benchmark_metrics.db"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"


@dataclass
class ReportMeta:
    path: Path
    report_type: str
    file_size: int
    mtime_utc: str
    sha256: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_report_type(path: Path, payload: Any) -> str | None:
    name = path.name
    if isinstance(payload, dict) and "transitions" in payload and "series_coherence_score" in payload:
        return "character_coherence"
    if isinstance(payload, dict) and "summary" in payload and "episodes" in payload:
        return "quality"
    if isinstance(payload, dict) and "episode" in payload and "overall_score" in payload:
        return "quality_single"
    if isinstance(payload, dict) and "history" in payload and "best_policy" in payload:
        return "rl_policy_history"
    if isinstance(payload, dict) and "metric" in payload and "rows" in payload and "total" in payload:
        return "comparison_benchmark"
    if "character_coherence" in name:
        return "character_coherence"
    if "quality" in name:
        return "quality"
    if "rl_code_policy_history" in name:
        return "rl_policy_history"
    return None


def compute_meta(path: Path, report_type: str) -> ReportMeta:
    stat = path.stat()
    data = path.read_bytes()
    return ReportMeta(
        path=path.resolve(),
        report_type=report_type,
        file_size=stat.st_size,
        mtime_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS report_files (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            report_type TEXT NOT NULL,
            label TEXT,
            file_size INTEGER NOT NULL,
            file_mtime_utc TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            ingested_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS coherence_runs (
            id INTEGER PRIMARY KEY,
            report_file_id INTEGER NOT NULL UNIQUE REFERENCES report_files(id) ON DELETE CASCADE,
            label TEXT,
            ep_start INTEGER,
            ep_end INTEGER,
            episode_count INTEGER,
            series_coherence_score REAL,
            full_text_coherence_score REAL,
            series_coherence_score_with_full_text REAL,
            protagonist_presence_rate REAL,
            global_character_voice_score REAL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS coherence_episodes (
            id INTEGER PRIMARY KEY,
            coherence_run_id INTEGER NOT NULL REFERENCES coherence_runs(id) ON DELETE CASCADE,
            episode_id TEXT NOT NULL,
            episode_number INTEGER,
            chapter_found INTEGER,
            chapter_path TEXT,
            character_mentions_count INTEGER,
            protagonist_present INTEGER
        );

        CREATE TABLE IF NOT EXISTS coherence_transitions (
            id INTEGER PRIMARY KEY,
            coherence_run_id INTEGER NOT NULL REFERENCES coherence_runs(id) ON DELETE CASCADE,
            from_episode TEXT NOT NULL,
            to_episode TEXT NOT NULL,
            cast_overlap_score REAL,
            clue_memory_score REAL,
            emotion_continuity_score REAL,
            relationship_continuity_score REAL,
            transition_score REAL
        );

        CREATE TABLE IF NOT EXISTS coherence_voice_details (
            id INTEGER PRIMARY KEY,
            coherence_run_id INTEGER NOT NULL REFERENCES coherence_runs(id) ON DELETE CASCADE,
            character_id TEXT NOT NULL,
            episodes_covered INTEGER,
            voice_consistency_score REAL
        );

        CREATE TABLE IF NOT EXISTS quality_runs (
            id INTEGER PRIMARY KEY,
            report_file_id INTEGER NOT NULL UNIQUE REFERENCES report_files(id) ON DELETE CASCADE,
            label TEXT,
            ep_start INTEGER,
            ep_end INTEGER,
            total_episodes INTEGER,
            avg_overall_score REAL,
            passed INTEGER,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quality_episodes (
            id INTEGER PRIMARY KEY,
            quality_run_id INTEGER NOT NULL REFERENCES quality_runs(id) ON DELETE CASCADE,
            episode_key TEXT NOT NULL,
            episode_num INTEGER,
            overall_score REAL,
            word_count INTEGER,
            sentence_count INTEGER,
            paragraph_count INTEGER,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quality_episode_metrics (
            id INTEGER PRIMARY KEY,
            quality_episode_id INTEGER NOT NULL REFERENCES quality_episodes(id) ON DELETE CASCADE,
            metric_name TEXT NOT NULL,
            score REAL,
            pass INTEGER,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rl_runs (
            id INTEGER PRIMARY KEY,
            report_file_id INTEGER NOT NULL UNIQUE REFERENCES report_files(id) ON DELETE CASCADE,
            label TEXT,
            rounds INTEGER,
            best_reward REAL,
            best_policy_json TEXT,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rl_iterations (
            id INTEGER PRIMARY KEY,
            rl_run_id INTEGER NOT NULL REFERENCES rl_runs(id) ON DELETE CASCADE,
            round INTEGER NOT NULL,
            reward REAL,
            best_reward REAL,
            accepted INTEGER,
            explore INTEGER,
            changes_json TEXT,
            policy_json TEXT,
            candidate_policy_json TEXT
        );

        CREATE TABLE IF NOT EXISTS comparison_runs (
            id INTEGER PRIMARY KEY,
            report_file_id INTEGER NOT NULL UNIQUE REFERENCES report_files(id) ON DELETE CASCADE,
            label TEXT,
            metric TEXT,
            total INTEGER,
            output_closer_count INTEGER,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS comparison_rows (
            id INTEGER PRIMARY KEY,
            comparison_run_id INTEGER NOT NULL REFERENCES comparison_runs(id) ON DELETE CASCADE,
            ep TEXT,
            cfg_out REAL,
            cfg_good REAL,
            closer TEXT,
            raw_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_report_files_type ON report_files(report_type);
        CREATE INDEX IF NOT EXISTS idx_coherence_transitions_run ON coherence_transitions(coherence_run_id);
        CREATE INDEX IF NOT EXISTS idx_quality_episodes_run ON quality_episodes(quality_run_id);
        CREATE INDEX IF NOT EXISTS idx_rl_iterations_run ON rl_iterations(rl_run_id);
        CREATE INDEX IF NOT EXISTS idx_comparison_rows_run ON comparison_rows(comparison_run_id);
        """
    )
    conn.commit()


def parse_ep_range_from_payload(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    nums: list[int] = []
    if isinstance(payload.get("episode_ids"), list):
        for eid in payload["episode_ids"]:
            if not isinstance(eid, str):
                continue
            m = re.match(r"ep(\d+)_", eid)
            if m:
                nums.append(int(m.group(1)))
    if not nums and isinstance(payload.get("episodes"), list):
        for ep in payload["episodes"]:
            if not isinstance(ep, dict):
                continue
            if isinstance(ep.get("episode_number"), int):
                nums.append(ep["episode_number"])
            elif isinstance(ep.get("episode"), str):
                m = re.search(r"ep(\d+)", ep["episode"])
                if m:
                    nums.append(int(m.group(1)))
    if not nums:
        return None, None
    return min(nums), max(nums)


def parse_label(path: Path) -> str:
    return path.stem


def upsert_report_file(conn: sqlite3.Connection, meta: ReportMeta) -> int:
    label = parse_label(meta.path)
    conn.execute(
        """
        INSERT INTO report_files(file_path, report_type, label, file_size, file_mtime_utc, sha256, ingested_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            report_type=excluded.report_type,
            label=excluded.label,
            file_size=excluded.file_size,
            file_mtime_utc=excluded.file_mtime_utc,
            sha256=excluded.sha256,
            ingested_at_utc=excluded.ingested_at_utc
        """,
        (
            str(meta.path),
            meta.report_type,
            label,
            meta.file_size,
            meta.mtime_utc,
            meta.sha256,
            utc_now_iso(),
        ),
    )
    row = conn.execute("SELECT id FROM report_files WHERE file_path = ?", (str(meta.path),)).fetchone()
    assert row is not None
    return int(row["id"])


def bool_to_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def ingest_coherence(conn: sqlite3.Connection, report_file_id: int, path: Path, payload: dict[str, Any]) -> None:
    conn.execute("DELETE FROM coherence_runs WHERE report_file_id = ?", (report_file_id,))
    ep_start, ep_end = parse_ep_range_from_payload(payload)
    gv = payload.get("global_character_voice") or {}
    ft = payload.get("full_text_coherence_benchmark") or {}
    cur = conn.execute(
        """
        INSERT INTO coherence_runs(
            report_file_id, label, ep_start, ep_end, episode_count,
            series_coherence_score, full_text_coherence_score, series_coherence_score_with_full_text,
            protagonist_presence_rate, global_character_voice_score, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report_file_id,
            path.stem,
            ep_start,
            ep_end,
            len(payload.get("episodes", []) or []),
            payload.get("series_coherence_score"),
            ft.get("score"),
            payload.get("series_coherence_score_with_full_text"),
            payload.get("protagonist_presence_rate"),
            gv.get("score") if isinstance(gv, dict) else None,
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    run_id = int(cur.lastrowid)

    for ep in payload.get("episodes", []) or []:
        if not isinstance(ep, dict):
            continue
        conn.execute(
            """
            INSERT INTO coherence_episodes(
                coherence_run_id, episode_id, episode_number, chapter_found, chapter_path,
                character_mentions_count, protagonist_present
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                ep.get("episode_id"),
                ep.get("episode_number"),
                bool_to_int(ep.get("chapter_found")),
                ep.get("chapter_path"),
                ep.get("character_mentions_count"),
                bool_to_int(ep.get("protagonist_present")),
            ),
        )

    for tr in payload.get("transitions", []) or []:
        if not isinstance(tr, dict):
            continue
        conn.execute(
            """
            INSERT INTO coherence_transitions(
                coherence_run_id, from_episode, to_episode, cast_overlap_score, clue_memory_score,
                emotion_continuity_score, relationship_continuity_score, transition_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                tr.get("from_episode"),
                tr.get("to_episode"),
                tr.get("cast_overlap_score"),
                tr.get("clue_memory_score"),
                tr.get("emotion_continuity_score"),
                tr.get("relationship_continuity_score"),
                tr.get("transition_score"),
            ),
        )

    details = gv.get("details") if isinstance(gv, dict) else None
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            conn.execute(
                """
                INSERT INTO coherence_voice_details(
                    coherence_run_id, character_id, episodes_covered, voice_consistency_score
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    item.get("character_id"),
                    item.get("episodes_covered"),
                    item.get("voice_consistency_score"),
                ),
            )


def normalize_quality_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "summary" in payload and "episodes" in payload:
        return payload
    if "episode" in payload and "overall_score" in payload:
        score = payload.get("overall_score")
        return {
            "episodes": [payload],
            "summary": {
                "total_episodes": 1,
                "avg_overall_score": score,
                "passed": 1 if isinstance(score, (int, float)) and score >= 0.7 else 0,
            },
        }
    return payload


def ingest_quality(conn: sqlite3.Connection, report_file_id: int, path: Path, payload: dict[str, Any]) -> None:
    conn.execute("DELETE FROM quality_runs WHERE report_file_id = ?", (report_file_id,))
    norm = normalize_quality_payload(payload)
    ep_start, ep_end = parse_ep_range_from_payload(norm)
    summary = norm.get("summary") or {}
    cur = conn.execute(
        """
        INSERT INTO quality_runs(
            report_file_id, label, ep_start, ep_end, total_episodes, avg_overall_score, passed, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report_file_id,
            path.stem,
            ep_start,
            ep_end,
            summary.get("total_episodes"),
            summary.get("avg_overall_score"),
            summary.get("passed"),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    run_id = int(cur.lastrowid)

    for ep in norm.get("episodes", []) or []:
        if not isinstance(ep, dict):
            continue
        basic = ep.get("basic_stats") or {}
        episode_key = ep.get("episode")
        episode_num = None
        if isinstance(episode_key, str):
            m = re.search(r"ep(\d+)", episode_key)
            if m:
                episode_num = int(m.group(1))
        cur_ep = conn.execute(
            """
            INSERT INTO quality_episodes(
                quality_run_id, episode_key, episode_num, overall_score, word_count, sentence_count, paragraph_count, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                episode_key,
                episode_num,
                ep.get("overall_score"),
                basic.get("word_count"),
                basic.get("sentence_count"),
                basic.get("paragraph_count"),
                json.dumps(ep, ensure_ascii=False),
            ),
        )
        q_ep_id = int(cur_ep.lastrowid)

        for key, value in ep.items():
            if key in {"episode", "overall_score"}:
                continue
            if not isinstance(value, dict):
                continue
            conn.execute(
                """
                INSERT INTO quality_episode_metrics(quality_episode_id, metric_name, score, pass, raw_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    q_ep_id,
                    key,
                    value.get("score") if isinstance(value, dict) else None,
                    bool_to_int(value.get("pass")) if isinstance(value, dict) else None,
                    json.dumps(value, ensure_ascii=False),
                ),
            )


def ingest_comparison_benchmark(
    conn: sqlite3.Connection, report_file_id: int, path: Path, payload: dict[str, Any]
) -> None:
    conn.execute("DELETE FROM comparison_runs WHERE report_file_id = ?", (report_file_id,))
    cur = conn.execute(
        """
        INSERT INTO comparison_runs(report_file_id, label, metric, total, output_closer_count, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            report_file_id,
            path.stem,
            payload.get("metric"),
            payload.get("total"),
            payload.get("output_closer_count"),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    comparison_run_id = int(cur.lastrowid)
    for row in payload.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        conn.execute(
            """
            INSERT INTO comparison_rows(comparison_run_id, ep, cfg_out, cfg_good, closer, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                comparison_run_id,
                row.get("ep"),
                row.get("cfg_out"),
                row.get("cfg_good"),
                row.get("closer"),
                json.dumps(row, ensure_ascii=False),
            ),
        )


def ingest_rl(conn: sqlite3.Connection, report_file_id: int, path: Path, payload: dict[str, Any]) -> None:
    conn.execute("DELETE FROM rl_runs WHERE report_file_id = ?", (report_file_id,))
    history = payload.get("history") or []
    best_policy = payload.get("best_policy")
    best_reward = None
    if isinstance(history, list):
        rewards = [h.get("reward") for h in history if isinstance(h, dict) and isinstance(h.get("reward"), (int, float))]
        if rewards:
            best_reward = max(rewards)
    cur = conn.execute(
        """
        INSERT INTO rl_runs(report_file_id, label, rounds, best_reward, best_policy_json, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            report_file_id,
            path.stem,
            len(history) - 1 if isinstance(history, list) and history else None,
            best_reward,
            json.dumps(best_policy, ensure_ascii=False) if best_policy is not None else None,
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    rl_run_id = int(cur.lastrowid)

    for item in history:
        if not isinstance(item, dict):
            continue
        conn.execute(
            """
            INSERT INTO rl_iterations(
                rl_run_id, round, reward, best_reward, accepted, explore, changes_json, policy_json, candidate_policy_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rl_run_id,
                item.get("round"),
                item.get("reward"),
                item.get("best_reward"),
                bool_to_int(item.get("accepted")),
                bool_to_int(item.get("explore")),
                json.dumps(item.get("changes"), ensure_ascii=False) if "changes" in item else None,
                json.dumps(item.get("policy"), ensure_ascii=False) if "policy" in item else None,
                json.dumps(item.get("candidate_policy"), ensure_ascii=False) if "candidate_policy" in item else None,
            ),
        )


def ingest_one(conn: sqlite3.Connection, path: Path, *, verbose: bool = False) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report_type = detect_report_type(path, payload)
    if not report_type:
        return "skipped"
    meta = compute_meta(path, report_type)
    with conn:
        report_file_id = upsert_report_file(conn, meta)
        if report_type == "character_coherence":
            ingest_coherence(conn, report_file_id, path, payload)
        elif report_type in {"quality", "quality_single"}:
            ingest_quality(conn, report_file_id, path, payload)
        elif report_type == "rl_policy_history":
            ingest_rl(conn, report_file_id, path, payload)
        elif report_type == "comparison_benchmark":
            ingest_comparison_benchmark(conn, report_file_id, path, payload)
    if verbose:
        print(f"[ingest] {report_type}: {path}")
    return report_type


def report_files(reports_dir: Path) -> Iterable[Path]:
    yield from sorted(reports_dir.glob("*.json"))


def cmd_init(args: argparse.Namespace) -> int:
    conn = connect_db(Path(args.db))
    init_schema(conn)
    print(f"Initialized DB schema at {args.db}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = connect_db(Path(args.db))
    init_schema(conn)
    ok = 0
    skipped = 0
    errors = 0
    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        try:
            result = ingest_one(conn, path, verbose=args.verbose)
            if result == "skipped":
                skipped += 1
            else:
                ok += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[error] {path}: {exc}")
            if not args.skip_errors:
                return 1
    print(f"Ingest complete | ok={ok} skipped={skipped} errors={errors} | db={args.db}")
    return 0


def cmd_ingest_all(args: argparse.Namespace) -> int:
    conn = connect_db(Path(args.db))
    init_schema(conn)
    ok = 0
    skipped = 0
    errors = 0
    for path in report_files(Path(args.reports_dir)):
        try:
            result = ingest_one(conn, path, verbose=args.verbose)
            if result == "skipped":
                skipped += 1
            else:
                ok += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[error] {path}: {exc}")
            if not args.skip_errors:
                return 1
    print(f"Ingest-all complete | ok={ok} skipped={skipped} errors={errors} | db={args.db}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    conn = connect_db(Path(args.db))
    init_schema(conn)
    rows = conn.execute(
        """
        SELECT report_type, COUNT(*) AS cnt, MAX(file_mtime_utc) AS latest_file_mtime
        FROM report_files
        GROUP BY report_type
        ORDER BY report_type
        """
    ).fetchall()
    print(f"DB: {args.db}")
    for row in rows:
        print(f"- {row['report_type']}: {row['cnt']} files (latest file mtime: {row['latest_file_mtime']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark report SQLite ingester")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create schema")
    p_init.set_defaults(func=cmd_init)

    p_ingest = sub.add_parser("ingest", help="Ingest specific report JSON files")
    p_ingest.add_argument("paths", nargs="+", help="Report JSON paths")
    p_ingest.add_argument("--skip-errors", action="store_true")
    p_ingest.add_argument("--verbose", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_all = sub.add_parser("ingest-all", help="Ingest all reports/*.json")
    p_all.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    p_all.add_argument("--skip-errors", action="store_true", help="Skip invalid/partial JSON during active runs")
    p_all.add_argument("--verbose", action="store_true")
    p_all.set_defaults(func=cmd_ingest_all)

    p_summary = sub.add_parser("summary", help="Show DB counts")
    p_summary.set_defaults(func=cmd_summary)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
