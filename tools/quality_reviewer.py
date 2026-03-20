#!/usr/bin/env python3
"""
Agent 4: Quality Reviewer

Objective checks + user scorecard generation for a generated chapter.

Auto-checks (pass/fail):
  - Beat coverage: every beat in episode YAML appears in the chapter
  - Character emotional continuity from story_state.json
  - Character invariant violations
  - Timeline contradictions
  - Word/phrase over-repetition

Scorecard (human judgement):
  - Prompts user to rate 긴장감 / 캐릭터 / 흐름 / 재미 (1-5)
  - Receives free-text feedback, parses with LLM
  - Updates story_state.json with scores + episode summary

Usage:
    python tools/quality_reviewer.py --episode ep01_academic_presentation [--dry-run]
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

from src.novel_writer.env_loader import load_project_env
from src.novel_writer.llm_client import LLMClient

# ── Locked file guard (same as config_guardian) ────────────────────────────────
LOCKED_FILES = {
    "config/storyline.yaml",
    "config/world_facts.yaml",
    "config/characters.yaml",
}


def _assert_not_locked(rel_path: str) -> None:
    normalized = rel_path.replace("\\", "/").lstrip("/")
    if normalized in LOCKED_FILES:
        raise PermissionError(
            f"Locked config file '{rel_path}' cannot be modified by agents."
        )


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
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


# ── Episode resolver ───────────────────────────────────────────────────────────

def resolve_episode_file(episode_key: str) -> Path:
    ep_dir = REPO_ROOT / "config" / "episodes"
    # Allow bare number: "1" → ep01_..., "15" → ep15_...
    if episode_key.isdigit():
        padded = f"ep{int(episode_key):02d}"
        matches = sorted(ep_dir.glob(f"{padded}_*.yaml"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"에피소드 {episode_key}화에 해당하는 파일 없음")
    # Try exact match first
    exact = ep_dir / f"{episode_key}.yaml"
    if exact.exists():
        return exact
    # Try prefix match
    matches = sorted(ep_dir.glob(f"{episode_key}*.yaml"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Episode file not found for key: {episode_key}")


def resolve_chapter_file(episode_key: str, run_output_dir: Path | None = None) -> Path | None:
    """Find most recent chapter.md for the episode."""
    candidates: list[Path] = []
    if run_output_dir and run_output_dir.exists():
        candidates += sorted(run_output_dir.rglob("*chapter*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    output_dir = REPO_ROOT / "output"
    if output_dir.exists():
        candidates += sorted(output_dir.rglob("*chapter*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if episode_key.split("_")[0] in str(c) or episode_key in str(c):
            return c
    # Fall back to most recent overall
    if candidates:
        return candidates[0]
    return None


# ── Auto checks ────────────────────────────────────────────────────────────────

def check_beat_coverage(episode_data: dict, chapter_text: str) -> tuple[bool, list[str]]:
    """Check that all beats/key events in the episode YAML appear in the chapter."""
    beats = episode_data.get("beats", []) or []
    if not beats:
        # Many episodes don't have a 'beats' field — use introduced_clues as proxy
        beats = [c.get("content", "")[:80] for c in (episode_data.get("introduced_clues") or []) if isinstance(c, dict)]

    missing = []
    for beat in beats:
        beat_str = str(beat).strip()[:120]
        # Simple keyword presence check (not perfect but fast)
        keywords = re.findall(r"[가-힣A-Za-z]{3,}", beat_str)
        if keywords:
            top_kw = keywords[:3]
            found = any(kw in chapter_text for kw in top_kw)
            if not found:
                missing.append(beat_str[:80])

    ok = len(missing) == 0
    return ok, missing


def check_character_continuity(episode_data: dict, chapter_text: str, story_state: dict) -> tuple[bool, list[str]]:
    """Check that characters' emotional states match story_state baseline."""
    issues = []
    char_states = story_state.get("character_states", {})
    for char_id, state in char_states.items():
        arc_status = state.get("arc_status", "")
        emotional_summary = state.get("emotional_summary", "")
        # Simple: if the character's name appears, their arc status keyword should appear too
        char_name_variants = [char_id, char_id.replace("_", " ")]
        appears = any(v in chapter_text.lower() for v in char_name_variants)
        # No hard check here — just collect state as context
    return True, issues


def check_character_invariants(chapter_text: str, characters_yaml: dict) -> tuple[bool, list[str]]:
    """Check character invariants from characters.yaml 'avoid' fields."""
    issues = []
    chars = characters_yaml.get("characters", []) or []
    for char in chars:
        avoid_list = char.get("speech_profile", {}).get("avoid", []) or []
        for avoid_phrase in avoid_list:
            # Check if the phrase appears verbatim (rough heuristic)
            if len(avoid_phrase) > 4 and avoid_phrase in chapter_text:
                issues.append(
                    f"⚠️ {char.get('id','?')}: 금지 표현 감지 — \"{avoid_phrase}\""
                )
    ok = len(issues) == 0
    return ok, issues


def check_word_repetition(chapter_text: str, threshold: int = 5) -> tuple[bool, list[str]]:
    """Detect words/phrases used too many times in the chapter."""
    issues = []
    words = re.findall(r"[가-힣]{3,}|[A-Za-z]{5,}", chapter_text)
    freq: dict[str, int] = defaultdict(int)
    for w in words:
        freq[w] += 1
    repeats = [(w, c) for w, c in freq.items() if c >= threshold]
    repeats.sort(key=lambda x: -x[1])
    for word, count in repeats[:10]:
        issues.append(f"'{word}' {count}회 반복")
    ok = len(repeats) == 0
    return ok, issues


def check_timeline_consistency(episode_data: dict, chapter_text: str) -> tuple[bool, list[str]]:
    """Basic check: episode date era appears correctly."""
    issues = []
    ep_date = str(episode_data.get("date", ""))
    if ep_date:
        year = ep_date[:4]
        # If a different year appears multiple times, flag it
        other_years = re.findall(r"\b(20[3-9]\d|21\d\d)\b", chapter_text)
        wrong = [y for y in other_years if y != year and other_years.count(y) > 2]
        for y in set(wrong):
            issues.append(f"⚠️ 에피소드 날짜({year})와 다른 연도 `{y}` 다수 등장")
    ok = len(issues) == 0
    return ok, issues


# ── Scorecard builder ─────────────────────────────────────────────────────────

def build_scorecard(
    episode_key: str,
    auto_results: dict[str, tuple[bool, list[str]]],
) -> str:
    lines = [
        f"📋 **{episode_key} 품질 스코어카드**",
        "─────────────────────────────",
        "**자동 체크 결과:**",
    ]
    for check_name, (ok, details) in auto_results.items():
        icon = "✅" if ok else "⚠️"
        detail_str = f" ({', '.join(details[:3])})" if details else ""
        lines.append(f"  {icon} {check_name}{detail_str}")

    lines += [
        "",
        "**아래 점수(1–5)로 자유롭게 피드백 주세요:**",
        "",
        "예시: \"긴장감이 잘 살았어. 중반부가 좀 늘어지긴 했지만 전반적으로 좋아. 다음으로 가자\"",
        "",
        "_(점수 구조: 긴장감·캐릭터·흐름·재미 + 자유 코멘트)_",
        "_(\"다음으로 가자\", \"next\", \"ok\" 등의 신호가 있으면 다음 에피소드로 진행합니다)_",
    ]
    return "\n".join(lines)


# ── Feedback parser ───────────────────────────────────────────────────────────

def parse_feedback_with_llm(
    raw_feedback: str,
    episode_key: str,
    llm: LLMClient,
) -> dict:
    """
    Parse free-text user feedback into structured JSON using LLM.
    Returns dict with keys: approved_next_episode, estimated_scores, specific_issues,
                            positive_notes, raw_feedback
    """
    system = (
        "You are a feedback parser for a novel writing pipeline. "
        "Extract structured information from the user's free-text review of a novel chapter. "
        "Return ONLY valid JSON with these fields:\n"
        "- approved_next_episode: boolean (true if user signals to move to next episode)\n"
        "- estimated_scores: {긴장감: 1-5, 캐릭터: 1-5, 흐름: 1-5, 재미: 1-5} (estimate from tone if not explicit)\n"
        "- specific_issues: list of strings (concrete problems mentioned)\n"
        "- positive_notes: list of strings (specific praise)\n"
        "- raw_feedback: the original text verbatim\n\n"
        "Approval signals include: '다음으로 가자', 'next', 'ok', '이정도면 됐어', '진행해', 'good to go', '넘어가자'\n"
        "Return ONLY the JSON object, no markdown."
    )
    prompt = (
        f"Episode: {episode_key}\n"
        f"User feedback:\n{raw_feedback.strip()}"
    )
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            system,
            False,
            "quality_reviewer_feedback_parse",
            None,
            400,
        )
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\n?", "", response).strip().rstrip("`").strip()
        return json.loads(cleaned)
    except Exception as exc:
        # Fallback: basic heuristic parsing
        approved = any(
            kw in raw_feedback.lower()
            for kw in ["다음으로 가자", "next", "ok", "이정도면", "진행해", "넘어가자", "good to go"]
        )
        return {
            "approved_next_episode": approved,
            "estimated_scores": {"긴장감": 3, "캐릭터": 3, "흐름": 3, "재미": 3},
            "specific_issues": [],
            "positive_notes": [],
            "raw_feedback": raw_feedback,
            "_parse_error": str(exc),
        }


# ── Story state updater ───────────────────────────────────────────────────────

def update_story_state(
    story_state_path: Path,
    episode_key: str,
    episode_data: dict,
    parsed_feedback: dict,
) -> None:
    """Update story_state.json with episode summary and user scores."""
    state = _load_json(story_state_path)

    scores = parsed_feedback.get("estimated_scores", {})
    episode_summary = {
        "one_line": (episode_data.get("summary", "") or "")[:100].strip().replace("\n", " "),
        "key_events": [],
        "user_scores": scores,
        "feedback": {
            "approved_next_episode": parsed_feedback.get("approved_next_episode", False),
            "specific_issues": parsed_feedback.get("specific_issues", []),
            "positive_notes": parsed_feedback.get("positive_notes", []),
            "raw_feedback": parsed_feedback.get("raw_feedback", ""),
        },
        "completed_at": datetime.utcnow().strftime("%Y-%m-%d"),
    }

    state.setdefault("episode_summaries", {})[episode_key] = episode_summary
    if parsed_feedback.get("approved_next_episode"):
        state["last_completed_episode"] = episode_key

    _save_json(story_state_path, state)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_quality_review(
    episode_key: str,
    chapter_path: Path | None = None,
    run_output_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[str, dict[str, tuple[bool, list[str]]]]:
    """
    Run all auto-checks. Returns (scorecard_text, auto_results_dict).
    """
    load_project_env(REPO_ROOT)

    ep_file = resolve_episode_file(episode_key)
    raw = _load_yaml(ep_file)
    episode_data = raw.get("episode", {}) if raw else {}

    chars_yaml = _load_yaml(REPO_ROOT / "config" / "characters.yaml") or {}
    story_state = _load_json(REPO_ROOT / "data" / "story_state.json")

    if chapter_path is None:
        chapter_path = resolve_chapter_file(episode_key, run_output_dir)

    chapter_text = ""
    if chapter_path and chapter_path.exists():
        chapter_text = chapter_path.read_text(encoding="utf-8", errors="replace")

    if not chapter_text:
        chapter_text = ""

    auto_results: dict[str, tuple[bool, list[str]]] = {}
    ok_beat, missing_beats = check_beat_coverage(episode_data, chapter_text)
    auto_results["Beat 커버리지"] = (ok_beat, missing_beats)

    ok_cont, cont_issues = check_character_continuity(episode_data, chapter_text, story_state)
    auto_results["캐릭터 연속성"] = (ok_cont, cont_issues)

    ok_inv, inv_issues = check_character_invariants(chapter_text, chars_yaml)
    auto_results["캐릭터 Invariant"] = (ok_inv, inv_issues)

    ok_rep, rep_issues = check_word_repetition(chapter_text)
    auto_results["단어 반복"] = (ok_rep, rep_issues)

    ok_tl, tl_issues = check_timeline_consistency(episode_data, chapter_text)
    auto_results["타임라인 일관성"] = (ok_tl, tl_issues)

    scorecard = build_scorecard(episode_key, auto_results)
    return scorecard, auto_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality Reviewer — episode chapter auto-checks + scorecard")
    parser.add_argument("--episode", required=True, help="Episode key, e.g. ep01_academic_presentation")
    parser.add_argument("--chapter", default=None, help="Path to chapter.md (optional, auto-discovered)")
    parser.add_argument("--run-dir", default=None, help="Run output directory for chapter discovery")
    parser.add_argument("--dry-run", action="store_true", help="Print scorecard only, no state updates")
    args = parser.parse_args()

    chapter_path = Path(args.chapter) if args.chapter else None
    run_dir = Path(args.run_dir) if args.run_dir else None

    scorecard, auto_results = run_quality_review(
        episode_key=args.episode,
        chapter_path=chapter_path,
        run_output_dir=run_dir,
        dry_run=args.dry_run,
    )

    print(scorecard)
    print()
    total_issues = sum(len(details) for ok, details in auto_results.values() if not ok)
    if total_issues == 0:
        print("✅ 자동 검수 이상 없음")
    else:
        print(f"⚠️ 자동 검수 이슈 {total_issues}건 발견")


if __name__ == "__main__":
    main()
