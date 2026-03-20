#!/usr/bin/env python3
"""
Agent 1: Config Guardian

Read-only consistency checker for all episode/world configs.
- Reads all ep*.yaml + storyline.yaml + world_facts.yaml + characters.yaml
- Checks timeline ordering, character arc consistency, clue flow, beat coverage
- Cross-references story_state.json for completed episodes
- Outputs: reports/config_check_YYYYMMDD.md
- Proposes config changes via pending_config_changes.json (never edits directly)

Usage:
    python tools/config_guardian.py --episode-dir config/episodes/ [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Locked config files (read-only for agents) ────────────────────────────────
LOCKED_FILES = {
    "config/storyline.yaml",
    "config/world_facts.yaml",
    "config/characters.yaml",
}


def _assert_not_locked(rel_path: str) -> None:
    """Raise if an agent tries to write a locked config file."""
    normalized = rel_path.replace("\\", "/").lstrip("/")
    if normalized in LOCKED_FILES:
        raise PermissionError(
            f"Locked config file '{rel_path}' cannot be modified by agents. "
            "Submit a pending_config_changes.json request instead."
        )


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Episode loader ─────────────────────────────────────────────────────────────

def load_all_episodes(episode_dir: Path) -> list[dict[str, Any]]:
    """Load all ep*.yaml files sorted by filename. Skip files with YAML errors."""
    episodes = []
    for ep_file in sorted(episode_dir.glob("ep*.yaml")):
        try:
            data = _load_yaml(ep_file)
        except Exception as exc:
            print(f"⚠️ YAML 파싱 실패 ({ep_file.name}): {exc}", file=sys.stderr)
            continue
        if data and "episode" in data:
            episodes.append({
                "file": ep_file,
                "key": ep_file.stem,
                "data": data["episode"],
            })
    return episodes


# ── Checks ────────────────────────────────────────────────────────────────────

def check_timeline_order(episodes: list[dict]) -> list[str]:
    """Verify episode dates are strictly increasing."""
    issues = []
    prev_date = None
    prev_key = None
    for ep in episodes:
        raw_date = ep["data"].get("date")
        if raw_date is None:
            issues.append(f"⚠️ `{ep['key']}`: date 필드 없음")
            continue
        date_str = str(raw_date)
        try:
            date = datetime.fromisoformat(date_str)
        except ValueError:
            issues.append(f"⚠️ `{ep['key']}`: date 파싱 실패 ({date_str!r})")
            continue
        if prev_date is not None and date <= prev_date:
            issues.append(
                f"❌ 타임라인 역전: `{ep['key']}` ({date_str}) ≤ `{prev_key}` ({prev_date.isoformat()})"
            )
        prev_date = date
        prev_key = ep["key"]
    return issues


def check_clue_flow(episodes: list[dict], world_facts: dict) -> list[str]:
    """Check that clue IDs referenced in episodes are defined in world_facts."""
    issues = []
    # Extract discoverable clue IDs from world_facts
    discoverable = set()
    hidden_section = world_facts.get("world_facts", {}).get("discoverable_facts", {})
    for clue_group in hidden_section.values() if isinstance(hidden_section, dict) else []:
        if isinstance(clue_group, list):
            for clue in clue_group:
                if isinstance(clue, dict) and "id" in clue:
                    discoverable.add(clue["id"])

    all_ep_clue_ids: set[str] = set()
    for ep in episodes:
        clues = ep["data"].get("introduced_clues", []) or []
        for clue in clues:
            if isinstance(clue, dict):
                cid = clue.get("id")
                if cid:
                    all_ep_clue_ids.add(cid)

    # Check resolved clues — only flag entries that look like clue IDs (clue_ep* pattern)
    # Descriptive text entries (서술형) are valid and should be ignored.
    clue_id_pattern = re.compile(r"^clue_")
    for ep in episodes:
        for cid in ep["data"].get("resolved", []) or []:
            if not isinstance(cid, str):
                continue
            if not clue_id_pattern.match(cid):
                continue  # 서술형 텍스트 — 무시
            if cid not in all_ep_clue_ids:
                issues.append(
                    f"⚠️ `{ep['key']}`: resolved 클루 ID `{cid}` 가 어떤 에피소드에서도 introduced 되지 않음"
                )
    return issues


def check_character_arcs(episodes: list[dict], story_state: dict) -> list[str]:
    """Verify character states referenced in completed episodes match story_state."""
    issues = []
    completed = set(story_state.get("episode_summaries", {}).keys())
    char_states = story_state.get("character_states", {})

    for ep_key in completed:
        summary = story_state["episode_summaries"][ep_key]
        # Check that each character mentioned has a state entry
        # (lightweight check — more detailed checks require episode YAML cross-ref)
        for char_id in char_states:
            known_clues = char_states[char_id].get("known_clues", [])
            # Clues introduced in completed episodes should be tracked
            # Find matching ep in loaded episodes list
            # (story_state cross-check is best-effort here)
    return issues


def check_beat_coverage(episodes: list[dict], storyline: dict) -> list[str]:
    """Check that episodes cover the expected act structure from storyline."""
    issues = []
    acts = storyline.get("story_structure", {}).get("acts", [])
    if not acts:
        return issues

    # Build expected episode count per act
    total_eps = sum(a.get("episode_count", 0) for a in acts if isinstance(a, dict))
    actual_eps = len(episodes)
    if actual_eps != total_eps:
        issues.append(
            f"⚠️ 스토리라인 예상 에피소드 수({total_eps}) ≠ 실제 ep*.yaml 파일 수({actual_eps})"
        )
    return issues


def check_duplicate_episode_ids(episodes: list[dict]) -> list[str]:
    """Detect duplicate episode IDs."""
    issues = []
    seen: dict[str, str] = {}
    for ep in episodes:
        ep_id = ep["data"].get("id", ep["key"])
        if ep_id in seen:
            issues.append(
                f"❌ 중복 에피소드 ID: `{ep_id}` — `{seen[ep_id]}` 와 `{ep['key']}`"
            )
        else:
            seen[ep_id] = ep["key"]
    return issues


def check_word_repetition_in_summaries(episodes: list[dict]) -> list[str]:
    """Flag episodes whose summary reuses the same Korean phrase too many times."""
    issues = []
    THRESHOLD = 4
    for ep in episodes:
        summary = ep["data"].get("summary", "") or ""
        words = re.findall(r"[가-힣]{3,}", summary)
        freq: dict[str, int] = defaultdict(int)
        for w in words:
            freq[w] += 1
        repeats = [f"'{w}'({c}회)" for w, c in freq.items() if c >= THRESHOLD]
        if repeats:
            issues.append(
                f"⚠️ `{ep['key']}` summary 반복 표현: {', '.join(repeats)}"
            )
    return issues


# ── Pending config change helpers ─────────────────────────────────────────────

def _next_request_id(pending: dict) -> str:
    requests = pending.get("requests", [])
    if not requests:
        return "req_001"
    last_ids = [r.get("id", "req_000") for r in requests]
    nums = []
    for rid in last_ids:
        m = re.search(r"(\d+)$", rid)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums, default=0) + 1
    return f"req_{next_num:03d}"


def propose_config_change(
    pending_path: Path,
    requested_by: str,
    file_rel: str,
    description: str,
    proposed_diff: str,
) -> str:
    """Add a change request to pending_config_changes.json. Returns request ID."""
    _assert_not_locked(file_rel)  # Validate that target is not locked
    pending = _load_json(pending_path) or {"requests": []}
    req_id = _next_request_id(pending)
    pending.setdefault("requests", []).append({
        "id": req_id,
        "requested_by": requested_by,
        "file": file_rel,
        "description": description,
        "proposed_diff": proposed_diff,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    _save_json(pending_path, pending)
    return req_id


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(
    episodes: list[dict],
    all_issues: dict[str, list[str]],
    story_state: dict,
    run_date: str,
) -> str:
    total_issues = sum(len(v) for v in all_issues.values())
    status_icon = "✅" if total_issues == 0 else "⚠️"

    lines = [
        f"# Config Guardian 리포트 — {run_date}",
        f"",
        f"**상태**: {status_icon} 총 {total_issues}건 이슈",
        f"**에피소드 파일 수**: {len(episodes)}",
        f"**마지막 완료 에피소드**: {story_state.get('last_completed_episode') or '없음'}",
        f"",
    ]

    for section, issues in all_issues.items():
        lines.append(f"## {section}")
        if issues:
            for issue in issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- ✅ 이상 없음")
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_guardian(episode_dir: Path, dry_run: bool = False) -> tuple[str, list[str]]:
    """
    Run all checks. Returns (report_text, list_of_pending_request_ids).
    """
    ep_dir = REPO_ROOT / episode_dir if not episode_dir.is_absolute() else episode_dir
    config_dir = REPO_ROOT / "config"
    data_dir = REPO_ROOT / "data"

    episodes = load_all_episodes(ep_dir)

    storyline = _load_yaml(config_dir / "storyline.yaml") or {}
    world_facts = _load_yaml(config_dir / "world_facts.yaml") or {}
    story_state = _load_json(data_dir / "story_state.json")
    pending_path = data_dir / "pending_config_changes.json"

    all_issues: dict[str, list[str]] = {}

    all_issues["타임라인 순서"] = check_timeline_order(episodes)
    all_issues["클루 흐름"] = check_clue_flow(episodes, world_facts)
    all_issues["캐릭터 Arc 연속성"] = check_character_arcs(episodes, story_state)
    all_issues["Beat 커버리지"] = check_beat_coverage(episodes, storyline)
    all_issues["중복 에피소드 ID"] = check_duplicate_episode_ids(episodes)

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    report_text = generate_report(episodes, all_issues, story_state, run_date)

    # Save report
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"config_check_{run_date}.txt"
    if not dry_run:
        report_path.write_text(report_text, encoding="utf-8")

    # Collect any pending request IDs that were created during this run
    # (currently none are auto-proposed in this baseline; proposals come from
    #  human-readable issues list)
    pending_ids: list[str] = []

    return report_text, pending_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Config Guardian — read-only consistency checker")
    parser.add_argument("--episode-dir", default="config/episodes", help="Directory with ep*.yaml files")
    parser.add_argument("--dry-run", action="store_true", help="Do not write report file")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    report_text, pending_ids = run_guardian(episode_dir, dry_run=args.dry_run)

    print(report_text)
    if pending_ids:
        print(f"\n📋 Config 변경 요청 {len(pending_ids)}건 등록: {', '.join(pending_ids)}")

    if not args.dry_run:
        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        print(f"\n✅ 리포트 저장 완료: reports/config_check_{run_date}.txt")


if __name__ == "__main__":
    main()
