#!/usr/bin/env python3
"""
Discord-driven 3-agent loop bot for the Novel Writer project.

Flow:
1) Simulator agent runs simulation + chapter generation and posts result.
2) Reviewer agent reads chapter and produces:
   - story-content critique
   - writing-style critique
3) Fixer agent maps critiques to concrete simulator config edits and applies patch.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import sys
import ssl
import urllib.parse
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import aiohttp
import discord
import yaml
import certifi

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()

from src.novel_writer.env_loader import load_project_env
from src.novel_writer.llm_client import LLMClient


ROOT_OUTPUT_DIR = REPO_ROOT / "output"
CONFIG_EP_DIR = REPO_ROOT / "config" / "episodes"
STATE_FILE = REPO_ROOT / "data" / "discord_loop_state.json"

SIM_DONE_TAG = "[NOVEL_LOOP][SIM_DONE]"
REVIEW_DONE_TAG = "[NOVEL_LOOP][REVIEW_DONE]"
FIX_DONE_TAG = "[NOVEL_LOOP][FIX_DONE]"
RUN_END_TAG = "[NOVEL_LOOP][RUN_END]"

CMD_START = "!novel-loop"
CMD_RESET = "!novel-loop-reset"
CMD_REVIEW = "!novel-review"
CMD_FIX = "!novel-fix"
CMD_STOP = "!novel-stop"

PROTECTED_FIXER_RULE_FILES: dict[str, list[str]] = {
    "src/novel_writer/prose_generator.py": [
        "COLON_DIALOGUE_LABEL_BAN = (",
        "f\"{COLON_DIALOGUE_LABEL_BAN}\\n\"",
        "`이름: \\\"대사\\\"` 형식",
    ],
}

STOP_REQUESTED_CHANNELS: set[int] = set()
FIXER_APPLY_LOCK = asyncio.Lock()


def _request_stop(channel_id: int) -> None:
    STOP_REQUESTED_CHANNELS.add(int(channel_id))


def _clear_stop(channel_id: int) -> None:
    STOP_REQUESTED_CHANNELS.discard(int(channel_id))


def _is_stop_requested(channel_id: int) -> bool:
    return int(channel_id) in STOP_REQUESTED_CHANNELS


@dataclass
class JobConfig:
    channel_id: int
    episode_key: str
    max_cycles: int
    target_words: int
    scenes: int
    budget: float
    protagonist: str
    seed_message_id: int
    run_date: str
    run_id: str
    run_output_dir: str
    reviewer_bot_token: str
    fixer_bot_token: str
    manager_bot_token: str
    parallel_branches: int
    manager_period: int


def _run_cmd(
    cmd: list[str],
    timeout_sec: int = 3600,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _list_worktree_change_candidates() -> set[str]:
    """
    Return modified/deleted/untracked worktree paths relative to repo root.
    """
    rc, out, _ = _run_cmd(
        ["git", "ls-files", "-m", "-d", "-o", "--exclude-standard", "--", "."],
        timeout_sec=30,
    )
    if rc != 0:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _sha256_for_relpath(rel_path: str) -> str | None:
    p = (REPO_ROOT / rel_path).resolve()
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        return None
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


async def _run_cmd_stream(
    cmd: list[str],
    timeout_sec: int = 3600,
    extra_env: dict[str, str] | None = None,
    on_line: Any = None,
    on_heartbeat: Any = None,
    heartbeat_sec: int = 0,
    max_silence_sec: int = 0,
    should_stop: Any = None,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None

    lines: list[str] = []
    idle_timeout_triggered = False
    silence_elapsed = 0

    async def _drain() -> int:
        nonlocal idle_timeout_triggered, silence_elapsed
        while True:
            if should_stop is not None and bool(should_stop()):
                proc.kill()
                await proc.wait()
                return 130
            if heartbeat_sec > 0:
                try:
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout=heartbeat_sec)
                except asyncio.TimeoutError:
                    if proc.returncode is not None:
                        break
                    # Reduce false "still running" heartbeats right after process exit.
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=0.05)
                        break
                    except asyncio.TimeoutError:
                        pass
                    silence_elapsed += heartbeat_sec
                    if max_silence_sec > 0 and silence_elapsed >= max_silence_sec:
                        idle_timeout_triggered = True
                        proc.kill()
                        await proc.wait()
                        return 124
                    if should_stop is not None and bool(should_stop()):
                        proc.kill()
                        await proc.wait()
                        return 130
                    if on_heartbeat is not None:
                        await on_heartbeat()
                    continue
            else:
                raw = await proc.stdout.readline()
            if not raw:
                break
            silence_elapsed = 0
            line = raw.decode("utf-8", errors="replace")
            lines.append(line)
            if on_line is not None:
                await on_line(line.rstrip("\n"))
            if should_stop is not None and bool(should_stop()):
                proc.kill()
                await proc.wait()
                return 130
        return await proc.wait()

    try:
        rc = await asyncio.wait_for(_drain(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "".join(lines), "timeout"
    if idle_timeout_triggered:
        return 124, "".join(lines), f"idle-timeout({max_silence_sec}s)"
    return rc, "".join(lines), ""


def _env_value(*keys: str) -> str:
    for key in keys:
        val = os.environ.get(key, "")
        if val and val.strip():
            return val.strip()
    return ""


def _force_load_env_keys(keys: list[str]) -> None:
    """Force-load selected keys from .env, overriding inherited shell vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    wanted = set(keys)
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if key not in wanted:
            continue
        value = v.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _resolve_openai_api_key() -> str:
    key = _env_value("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in .env")
    if not key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY format invalid (must start with 'sk-').")
    return key


def _resolve_stage_bot_tokens() -> tuple[str, str, str]:
    reviewer_bot = _env_value("DISCORD_BOT_TOKEN2", "TOKEN2", "token2")
    fixer_bot = _env_value("DISCORD_BOT_TOKEN3", "TOKEN3", "token3")
    manager_bot = _env_value("DISCORD_BOT_TOKEN4", "TOKEN4", "token4")
    return reviewer_bot, fixer_bot, manager_bot


def _extract_episode_meta(episode_file: Path) -> tuple[str, str]:
    data = yaml.safe_load(episode_file.read_text(encoding="utf-8")) or {}
    ep_data = data["episode"] if isinstance(data.get("episode"), dict) else data

    episode_id = str(ep_data.get("id", "")).strip() or episode_file.stem
    protagonist = str(ep_data.get("protagonist", "")).strip()
    return episode_id, protagonist


def _resolve_episode_file(episode_key: str) -> Path:
    candidate = Path(episode_key)
    if candidate.exists():
        return candidate.resolve()

    exact = CONFIG_EP_DIR / f"{episode_key}.yaml"
    if exact.exists():
        return exact

    pref = sorted(CONFIG_EP_DIR.glob(f"{episode_key}*.yaml"))
    if pref:
        return pref[0]

    for yml in sorted(CONFIG_EP_DIR.glob("*.yaml")):
        try:
            eid, _ = _extract_episode_meta(yml)
        except Exception:
            continue
        if eid == episode_key or eid.startswith(episode_key):
            return yml
    raise FileNotFoundError(f"Cannot resolve episode: {episode_key}")


def _find_latest(base_dir: Path, path_pattern: str) -> Path | None:
    files = sorted(base_dir.glob(path_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _allocate_run_output_dir() -> tuple[str, str, Path]:
    date_key = datetime.now().strftime("%Y%m%d")
    date_dir = ROOT_OUTPUT_DIR / date_key
    date_dir.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for child in date_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            idx = int(child.name)
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    run_id = f"{max_idx + 1:03d}"
    run_dir = date_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return date_key, run_id, run_dir


def _ensure_state_dir() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> dict[str, Any]:
    _ensure_state_dir()
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, Any]) -> None:
    _ensure_state_dir()
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_reviewer_prompt(chapter_text: str) -> str:
    return (
        "You are a regular high-school student reader.\n"
        "You enjoy good novels, and you are now reviewing this chapter only as a casual reader.\n"
        "This novel is series thriller science fiction novel"
        "Do NOT evaluate plot logic or story structure.\n"
        "Focus only on readability, sentence flow, immersion, and whether it feels fun to read.\n"
        "Write in Korean.\n"
        "Return strict JSON with keys:\n"
        "{\n"
        '  "thrill_score_10": int,\n'
        '  "style_score_10": int,\n'
        '  "one_line_verdict": string,\n'
        '  "what_felt_good": [string, ...],\n'
        '  "what_felt_boring_or_hard": [string, ...],\n'
        '  "style_tips": [string, ...],\n'
        '  "reader_comment": string\n'
        "}\n"
        "Rules:\n"
        "- Do not talk about story logic correctness.\n"
        "- Talk like a normal student reader, not an expert critic.\n"
        "- Be concrete and honest.\n"
        "- Each list should have at least 3 items.\n"
        "- reader_comment should be 4~6 sentences.\n\n"
        f"Chapter text:\n{chapter_text[:16000]}"
    )


def _build_manager_prompt(
    cycle: int,
    branch_scores: list[dict[str, Any]],
    cycle_review_snippets: list[str],
    periodic_review_snippets: list[str],
) -> str:
    return (
        "너는 매니저 에이전트다. 목표는 리뷰 점수를 장기적으로 올리는 것이다.\n"
        "아래 데이터를 보고 Fixer에게 줄 강한 코드수정 지시를 작성하라.\n"
        "반드시 JSON만 출력하라.\n"
        "{\n"
        '  "cycle_summary": [string, ...],\n'
        '  "cross_branch_issues": [string, ...],\n'
        '  "fixer_priority_actions": [string, ...],\n'
        '  "score_strategy": [string, ...],\n'
        '  "periodic_diagnosis": [string, ...]\n'
        "}\n"
        "규칙:\n"
        "- 코드 수준 개선 지시만 작성(프롬프트/후처리/안전장치/리듬/가독성).\n"
        "- config/episodes 수정 지시는 금지.\n"
        "- 지시는 짧고 명령형으로 작성.\n"
        "- 한국어로 작성.\n\n"
        f"[현재 사이클] {cycle}\n"
        f"[브랜치 점수 요약]\n{json.dumps(branch_scores, ensure_ascii=False)}\n\n"
        f"[이번 사이클 리뷰 발췌(브랜치 5개)]\n" + "\n---\n".join(cycle_review_snippets[:5]) + "\n\n"
        f"[장기 리뷰 발췌(주기 집계)]\n" + ("\n---\n".join(periodic_review_snippets[:25]) if periodic_review_snippets else "(없음)")
    )


def _format_manager_md(
    episode_id: str,
    cycle: int,
    manager_data: dict[str, Any],
    branch_scores: list[dict[str, Any]],
) -> str:
    def _items(key: str) -> list[str]:
        vals = manager_data.get(key, [])
        if not isinstance(vals, list):
            return []
        return [str(v).strip() for v in vals if str(v).strip()]

    lines = [
        f"# Manager Report: {episode_id} (cycle {cycle})",
        "",
        "## Branch Score Snapshot",
    ]
    for row in branch_scores:
        b = row.get("branch", "?")
        thrill = row.get("thrill_score_10", "n/a")
        style = row.get("style_score_10", "n/a")
        lines.append(f"- {b}: thrill={thrill}, style={style}")

    mapping = [
        ("cycle_summary", "Cycle Summary"),
        ("cross_branch_issues", "Cross-Branch Issues"),
        ("fixer_priority_actions", "Fixer Priority Actions"),
        ("score_strategy", "Score Strategy"),
        ("periodic_diagnosis", "Periodic Diagnosis (every manager period)"),
    ]
    for key, title in mapping:
        lines.extend(["", f"## {title}"])
        vals = _items(key)
        if vals:
            for v in vals:
                lines.append(f"- {v}")
        else:
            lines.append("- (none)")
    return "\n".join(lines).strip() + "\n"


def _parse_json_safe(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}
    return {}


def _format_review_md(
    episode_id: str,
    cycle: int,
    review: dict[str, Any],
) -> str:
    good = review.get("what_felt_good", []) or []
    weak = review.get("what_felt_boring_or_hard", []) or []
    tips = review.get("style_tips", []) or []
    verdict = review.get("one_line_verdict", "")
    comment = review.get("reader_comment", "")
    fun_score = review.get("thrill_score_10", "n/a")
    style_score = review.get("style_score_10", "n/a")

    lines = [
        f"# Reader Review: {episode_id} (cycle {cycle})",
        "",
        "## Verdict",
        f"- 스릴 점수(10점): {fun_score}",
        f"- 문체 점수(10점): {style_score}",
        f"- 한 줄 평: {verdict}",
        "",
        "## 좋았던 점(독자 체감)",
    ]
    for item in good:
        lines.append(f"- {item}")

    lines.extend(["", "## 지루하거나 읽기 어려웠던 점"])
    for item in weak:
        lines.append(f"- {item}")

    lines.extend(["", "## 문체 개선 팁"])
    for item in tips:
        lines.append(f"- {item}")

    lines.extend(["", "## 독자 코멘트", comment or "-"])
    return "\n".join(lines).strip() + "\n"


def _build_fixer_prompt(review_md: str, code_context: str) -> str:
    return (
        "너는 Fixer Agent다.\n"
        "독자 리뷰와 현재 코드 스니펫을 바탕으로, 문제의 원인을 분석하고 실제 코드 수정안을 제시하라.\n"
        "아래 스키마의 JSON만 엄격히 반환하라:\n"
        "{\n"
        '  "root_cause_analysis": [string, ...],\n'
        '  "change_summary": [string, ...],\n'
        '  "edits": [\n'
        "    {\n"
        '      "path": "relative/path.py",\n'
        '      "find": "exact old snippet",\n'
        '      "replace": "exact new snippet",\n'
        '      "reason": "why"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "규칙:\n"
        "- 코드만 수정 대상으로 삼아라. config/episodes 파일은 절대 수정하지 마라.\n"
        "- 독자 리뷰에서 지적된 가독성/문체 문제 해결에 집중하라.\n"
        "- find/replace는 실제 파일에 존재하는 정확한 코드 조각을 사용하라.\n"
        "- JSON 바깥의 설명 문장/코드블록/주석은 출력하지 마라.\n"
        "- root_cause_analysis, change_summary, edits.reason은 한국어로 작성하라.\n\n"
        f"독자 리뷰:\n{review_md[:9000]}\n\n"
        f"코드 컨텍스트:\n{code_context[:18000]}"
    )


def _build_code_context_for_fixer() -> str:
    target_files = [
        "simulate.py",
        "generate_chapter.py",
        "src/novel_writer/prose_generator.py",
        "src/novel_writer/scene_distiller.py",
        "src/novel_writer/orchestrator.py",
        "src/novel_writer/director.py",
    ]
    blocks: list[str] = []
    for rel in target_files:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        blocks.append(f"\n### FILE: {rel}\n{text[:5000]}")
    return "\n".join(blocks)


def _looks_english_heavy(text: str) -> bool:
    if not text:
        return False
    ascii_letters = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    hangul_letters = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
    return ascii_letters > (hangul_letters * 2 + 20)


def _needs_korean_fixer_data(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    parts: list[str] = []
    for key in ("root_cause_analysis", "change_summary"):
        vals = data.get(key, [])
        if isinstance(vals, list):
            parts.extend(str(v) for v in vals)
    edits = data.get("edits", [])
    if isinstance(edits, list):
        for e in edits:
            if isinstance(e, dict):
                parts.append(str(e.get("reason", "")))
    return _looks_english_heavy("\n".join(parts))


async def _translate_fixer_data_to_korean(llm: LLMClient, fixer_data: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "다음 JSON의 설명 텍스트만 한국어로 번역해서 같은 스키마의 JSON으로 반환하라.\n"
        "중요:\n"
        "- edits.path, edits.find, edits.replace는 절대 변경하지 마라.\n"
        "- root_cause_analysis, change_summary, edits.reason만 자연스러운 한국어로 변환하라.\n"
        "- JSON 외 텍스트를 출력하지 마라.\n\n"
        f"JSON:\n{json.dumps(fixer_data, ensure_ascii=False)}"
    )
    raw = await asyncio.to_thread(
        llm.chat,
        [{"role": "user", "content": prompt}],
        None,
        True,
        "discord_fixer_translate_ko",
        None,
        1600,
    )
    parsed = _parse_json_safe(raw)
    return parsed if isinstance(parsed, dict) else fixer_data


def _apply_code_edits(edits: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    applied: list[str] = []
    failed: list[str] = []
    changed_paths: list[str] = []
    for i, edit in enumerate(edits, start=1):
        path = str(edit.get("path", "")).strip()
        find = str(edit.get("find", ""))
        replace = str(edit.get("replace", ""))
        reason = str(edit.get("reason", "")).strip()
        if not path or not find:
            failed.append(f"{i}. invalid edit shape")
            continue
        abs_path = (REPO_ROOT / path).resolve()
        try:
            abs_path.relative_to(REPO_ROOT)
        except ValueError:
            failed.append(f"{i}. path outside repo: {path}")
            continue
        rel = str(abs_path.relative_to(REPO_ROOT))
        if rel.startswith("config/episodes/"):
            failed.append(f"{i}. blocked episode config edit: {rel}")
            continue
        if not abs_path.exists():
            failed.append(f"{i}. file not found: {rel}")
            continue
        src = abs_path.read_text(encoding="utf-8", errors="replace")
        if find not in src:
            failed.append(f"{i}. find snippet not found in {rel}")
            continue
        new_src = src.replace(find, replace, 1)
        abs_path.write_text(new_src, encoding="utf-8")
        applied.append(f"{i}. {rel} ({reason or 'updated'})")
        changed_paths.append(rel)
    return applied, failed, changed_paths


def _run_codex_fix(review_md_path: Path, summary_out_path: Path) -> tuple[int, str, str]:
    prompt = (
        "다음 리뷰 파일을 읽고 코드베이스를 직접 수정하라.\n"
        f"리뷰 파일: {review_md_path}\n\n"
        "요구사항:\n"
        "1) 왜 이런 문제가 생겼는지 원인 분석\n"
        "2) 시뮬레이터/챕터 생성 관련 코드를 실제로 수정\n"
        "3) config/episodes/* 는 절대 수정 금지\n"
        "4) 수정 후 문법 체크(변경 파일 대상) 수행\n"
        "5) 마지막 답변은 한국어로, 아래 형식으로 간단히:\n"
        "- Fixer 고민(원인 분석)\n"
        "- 코드 수정 요약\n"
        "- 적용된 코드 변경 파일 목록\n"
        "- 남은 리스크\n"
        "6) 중요: 사용자가 더티 워크트리 상태에서 진행을 명시적으로 승인했다.\n"
        "   기존 변경이 있어도 질문하지 말고, 해당 변경을 보존한 채 최소 diff로 계속 진행하라.\n"
        "   '어떻게 진행할지 선택해달라' 같은 확인 질문을 절대 출력하지 마라.\n"
        "7) 금지 규칙 보호: src/novel_writer/prose_generator.py 의 콜론 대사 금지 규칙은\n"
        "   수정/삭제/우회하지 마라. 해당 규칙은 고정 정책이다.\n"
    )
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(REPO_ROOT),
        "-o",
        str(summary_out_path),
        prompt,
    ]
    return _run_cmd(cmd, timeout_sec=1800)


def _build_cmd_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=CMD_START, add_help=False)
    p.add_argument("episode_key")
    p.add_argument("--max-cycles", type=int, default=3)
    p.add_argument("--target-words", type=int, default=0)
    p.add_argument("--scenes", type=int, default=0)
    p.add_argument("--budget", type=float, default=4.0)
    p.add_argument("--protagonist", default="")
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--manager-period", type=int, default=5)
    return p


def _build_review_cmd_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=CMD_REVIEW, add_help=False)
    p.add_argument("target", help="chapter.md path or run directory")
    p.add_argument("--episode-id", default="", help="episode id override")
    p.add_argument("--cycle", type=int, default=1, help="review cycle number label")
    return p


def _build_fix_cmd_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=CMD_FIX, add_help=False)
    p.add_argument("target", help="review.md path or run directory")
    p.add_argument("--episode-id", default="", help="episode id override")
    p.add_argument("--cycle", type=int, default=1, help="fix cycle number label")
    p.add_argument("--review-md", default="", help="explicit review md path")
    return p


def _infer_episode_id_from_chapter_name(chapter_name: str) -> str:
    name = chapter_name
    if name.endswith("_chapter.md"):
        return name[:-len("_chapter.md")]
    if name.endswith(".md"):
        return name[:-3]
    return name


def _resolve_review_target(target: str, episode_id_override: str = "") -> tuple[Path, Path, str]:
    p = Path(target).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()

    if p.is_file():
        chapter_path = p
        run_dir = p.parent
    elif p.is_dir():
        if episode_id_override:
            candidate = p / f"{episode_id_override}_chapter.md"
            if candidate.exists():
                chapter_path = candidate
            else:
                chapter_path = _find_latest(p, "*_chapter.md")
        else:
            chapter_path = _find_latest(p, "*_chapter.md")
        if chapter_path is None:
            raise FileNotFoundError(f"No chapter md found in {p}")
        run_dir = p
    else:
        raise FileNotFoundError(f"Target not found: {p}")

    episode_id = (episode_id_override or _infer_episode_id_from_chapter_name(chapter_path.name)).strip()
    if not episode_id:
        raise RuntimeError("Could not infer episode_id. Pass --episode-id.")
    return chapter_path, run_dir, episode_id


def _resolve_fix_target(
    target: str,
    episode_id_override: str = "",
    cycle: int = 1,
    review_md_override: str = "",
) -> tuple[Path, Path, str]:
    p = Path(target).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()

    if review_md_override:
        review_md = Path(review_md_override).expanduser()
        if not review_md.is_absolute():
            review_md = (REPO_ROOT / review_md).resolve()
        if not review_md.exists():
            raise FileNotFoundError(f"review md not found: {review_md}")
    else:
        review_md = None

    if p.is_file():
        run_dir = p.parent
        if review_md is None:
            review_md = p
    elif p.is_dir():
        run_dir = p
        if review_md is None:
            review_md = _find_latest(run_dir, f"*_cycle{cycle}_review.md")
            if review_md is None:
                review_md = _find_latest(run_dir, "*_review.md")
    else:
        raise FileNotFoundError(f"Target not found: {p}")

    if review_md is None or not review_md.exists():
        raise FileNotFoundError("No review markdown found. Pass --review-md.")

    episode_id = (episode_id_override or _infer_episode_id_from_chapter_name(review_md.name.split("_cycle")[0] + ".md")).strip()
    if not episode_id:
        raise RuntimeError("Could not infer episode_id. Pass --episode-id.")
    return review_md, run_dir, episode_id


async def _send_file(channel: discord.abc.Messageable, path: Path, note: str) -> None:
    if not path.exists():
        return
    await channel.send(content=note, file=discord.File(str(path), filename=path.name))


async def _send_text(channel: discord.abc.Messageable, text: str) -> None:
    content = (text or "").strip()
    if not content:
        return
    limit = 1900
    for i in range(0, len(content), limit):
        await channel.send(content[i:i + limit])


async def _rest_send_text(channel_id: int, text: str, bot_token: str) -> None:
    content = (text or "").strip()
    if not content:
        return
    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }
    limit = 1900
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(content), limit):
            chunk = content[i:i + limit]
            async with session.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                headers=headers,
                json={"content": chunk},
            ) as resp:
                if resp.status >= 300:
                    body = await resp.text()
                    raise RuntimeError(f"REST text send failed: {resp.status} {body[:240]}")


async def _rest_send_text_return_message_id(channel_id: int, text: str, bot_token: str) -> int | None:
    content = str(text).strip()
    if not content:
        return None
    headers = {"Authorization": f"Bot {bot_token}"}
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers=headers,
            json={"content": content[:1900]},
        ) as resp:
            body = await resp.text()
            if resp.status >= 300:
                raise RuntimeError(f"REST text send failed: {resp.status} {body[:240]}")
            try:
                data = json.loads(body)
                mid = data.get("id")
                return int(mid) if mid else None
            except Exception:
                return None


async def _rest_add_reaction(channel_id: int, message_id: int, emoji: str, bot_token: str) -> None:
    headers = {"Authorization": f"Bot {bot_token}"}
    encoded = urllib.parse.quote(emoji, safe="")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.put(
            f"https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me",
            headers=headers,
        ) as resp:
            if resp.status >= 300:
                body = await resp.text()
                raise RuntimeError(f"REST reaction failed: {resp.status} {body[:240]}")


async def _rest_send_file(channel_id: int, path: Path, note: str, bot_token: str) -> None:
    if not path.exists():
        return
    headers = {"Authorization": f"Bot {bot_token}"}
    form = aiohttp.FormData()
    form.add_field("payload_json", json.dumps({"content": note}, ensure_ascii=False))
    form.add_field(
        "files[0]",
        path.read_bytes(),
        filename=path.name,
        content_type="application/octet-stream",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers=headers,
            data=form,
        ) as resp:
            if resp.status >= 300:
                body = await resp.text()
                raise RuntimeError(f"REST file send failed: {resp.status} {body[:240]}")


async def _send_text_with_token(
    channel: discord.abc.Messageable,
    channel_id: int,
    text: str,
    bot_token: str,
    required: bool = False,
) -> None:
    if bot_token:
        try:
            await _rest_send_text(channel_id, text, bot_token)
            return
        except Exception as exc:
            if required:
                raise RuntimeError(f"stage bot send failed: {exc}") from exc
    elif required:
        raise RuntimeError("required stage bot token is missing")
    await _send_text(channel, text)


async def _send_file_with_token(
    channel: discord.abc.Messageable,
    channel_id: int,
    path: Path,
    note: str,
    bot_token: str,
    required: bool = False,
) -> None:
    if bot_token:
        try:
            await _rest_send_file(channel_id, path, note, bot_token)
            return
        except Exception as exc:
            if required:
                raise RuntimeError(f"stage bot file send failed: {exc}") from exc
    elif required:
        raise RuntimeError("required stage bot token is missing")
    await _send_file(channel, path, note)


async def run_simulator_agent(channel: discord.abc.Messageable, job: JobConfig, cycle: int) -> tuple[bool, dict[str, Any]]:
    run_dir = Path(job.run_output_dir)
    episode_file = _resolve_episode_file(job.episode_key)
    episode_id, protagonist_from_yaml = _extract_episode_meta(episode_file)
    protagonist = job.protagonist or protagonist_from_yaml or "kim_sumin"
    branch_label = run_dir.name if run_dir.name.startswith("branch_") else "main"
    sim_tag = f"[{branch_label}]"

    if _is_stop_requested(job.channel_id):
        await channel.send(f"{sim_tag} {RUN_END_TAG} 중지됨: 사용자 요청({CMD_STOP})")
        return False, {"stopped": True}

    await channel.send(
        f"{sim_tag} 1) Simulator Agent 시작 (cycle {cycle})\n"
        f"- episode: `{episode_id}`\n"
        f"- run_dir: `{run_dir}`"
    )

    sim_cmd = [
        "python3", "simulate.py",
        "--episode", str(episode_file),
        "--characters", "config/characters.yaml",
        "--world", "config/world_facts.yaml",
        "--storyline", "config/storyline.yaml",
        "--budget", str(job.budget),
        "--output", str(run_dir),
    ]
    previous_review_md: Path | None = None
    if cycle > 1:
        previous_review_md = _find_latest(run_dir, f"{episode_id}_cycle{cycle - 1}_review.md")
        if previous_review_md is None:
            previous_review_md = _find_latest(run_dir, f"{episode_id}_cycle*_review.md")
    if previous_review_md is not None:
        sim_cmd.extend(["--reader-review-md", str(previous_review_md)])
        await channel.send(f"{sim_tag} 리뷰 피드백 적용: `{previous_review_md.name}`")
    turn_re = re.compile(r"Turn\s+(\d+)\s*/\s*(\d+)")
    last_turn = 0
    checkpoints_sent: set[int] = set()

    async def _on_sim_line(line: str) -> None:
        nonlocal last_turn
        m = turn_re.search(line)
        if not m:
            return
        turn = int(m.group(1))
        total = int(m.group(2))
        if turn <= last_turn:
            return
        last_turn = turn

        # Announce only 5 times: 1%, 25%, 50%, 75%, 100%
        progress = (turn / max(total, 1)) * 100.0
        for checkpoint in (1, 25, 50, 75, 100):
            if progress >= checkpoint and checkpoint not in checkpoints_sent:
                checkpoints_sent.add(checkpoint)
                await channel.send(
                    f"{sim_tag} 시뮬레이션 진행: {checkpoint}% (Turn {turn}/{total})"
                )

    rc, _, err = await _run_cmd_stream(
        sim_cmd,
        3600,
        {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
        _on_sim_line,
        should_stop=lambda: _is_stop_requested(job.channel_id),
    )
    if rc == 130:
        await channel.send(f"{sim_tag} {RUN_END_TAG} 중지됨: 사용자 요청({CMD_STOP})")
        return False, {"stopped": True}
    if rc != 0:
        await _send_text(channel, f"{sim_tag} 시뮬레이션 실패\n```{err[-1500:]}```")
        return False, {}
    await channel.send(f"{sim_tag} 시뮬레이션 완료. 챕터 생성을 시작합니다.")

    gen_cmd = [
        "python3", "generate_chapter.py",
        "--episode", episode_id,
        "--episode-config", str(episode_file),
        "--protagonist", protagonist,
        "--output", str(run_dir),
        "--budget", str(job.budget),
    ]
    if previous_review_md is not None:
        gen_cmd.extend(["--reader-review-md", str(previous_review_md)])
    if job.target_words > 0:
        gen_cmd.extend(["--words", str(job.target_words)])
    if job.scenes > 0:
        gen_cmd.extend(["--scenes", str(job.scenes)])
    await channel.send(f"{sim_tag} 챕터 생성 진행 중...")

    gen_flags = {
        "stage1": False,
        "stage2": False,
        "distilled": False,
    }

    async def _on_gen_line(line: str) -> None:
        low = line.lower()
        if ("stage 1" in low or "scene distillation" in low) and not gen_flags["stage1"]:
            gen_flags["stage1"] = True
            await channel.send("챕터 생성: 장면 압축 단계 진행 중...")
        if ("stage 2" in low or "prose generation" in low) and not gen_flags["stage2"]:
            gen_flags["stage2"] = True
            await channel.send("챕터 생성: 본문 생성 단계 진행 중...")
        if ("distilled" in low and "scene" in low) and not gen_flags["distilled"]:
            gen_flags["distilled"] = True
            await channel.send("챕터 생성: 장면 압축 완료, 본문 작성으로 넘어갑니다.")

    async def _on_gen_heartbeat() -> None:
        heartbeat_count[0] += 1
        if heartbeat_count[0] == 1 or heartbeat_count[0] % 4 == 0:
            mins = int((heartbeat_count[0] * 45) / 60)
            await channel.send(
                f"{sim_tag} 챕터 생성 진행 중... ({mins}분째 출력 없음, 작업 계속 시도 중)"
            )

    heartbeat_count = [0]
    rc, _, err2 = await _run_cmd_stream(
        gen_cmd,
        3600,
        {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
        _on_gen_line,
        _on_gen_heartbeat,
        45,
        1200,
        lambda: _is_stop_requested(job.channel_id),
    )
    if rc == 130:
        await channel.send(f"{sim_tag} {RUN_END_TAG} 중지됨: 사용자 요청({CMD_STOP})")
        return False, {"stopped": True}
    if rc != 0:
        await _send_text(channel, f"{sim_tag} 챕터 생성 실패\n```{err2[-1500:]}```")
        return False, {}

    chapter = run_dir / f"{episode_id}_chapter.md"
    if not chapter.exists():
        chapter = _find_latest(run_dir, f"{episode_id}*chapter.md")
    if chapter is None:
        await channel.send(f"{sim_tag} 챕터 파일을 찾지 못했습니다.")
        return False, {}

    # Keep a per-cycle chapter snapshot so chapters are not overwritten in-place.
    cycle_chapter = run_dir / f"{episode_id}_cycle{cycle}_chapter.md"
    try:
        shutil.copyfile(chapter, cycle_chapter)
    except Exception:
        await channel.send(f"{sim_tag} 사이클 챕터 스냅샷 저장 실패: `{cycle_chapter}`")
        return False, {}

    # Keep chapter markdown local only; notify completion in channel.
    await channel.send(f"{sim_tag} 챕터 생성 완료 (로컬 저장): `{cycle_chapter}`")
    await channel.send(f"{sim_tag} {SIM_DONE_TAG} cycle={cycle} episode={episode_id} chapter={chapter.name}")

    return True, {
        "episode_id": episode_id,
        "episode_file": str(episode_file),
        "chapter": str(cycle_chapter),
        "run_output_dir": str(run_dir),
        "channel_id": job.channel_id,
        "reviewer_bot_token": job.reviewer_bot_token,
        "fixer_bot_token": job.fixer_bot_token,
    }


async def run_reviewer_agent(channel: discord.abc.Messageable, cycle: int, ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    run_dir = Path(ctx["run_output_dir"])
    episode_id = ctx["episode_id"]
    chapter_path = Path(ctx["chapter"])
    review_json_path = run_dir / f"{episode_id}_cycle{cycle}_reader_review.json"

    chapter_text = chapter_path.read_text(encoding="utf-8")
    llm = LLMClient(
        model="gpt-4o-mini",
        premium_model="gpt-5-mini",
        budget_usd=3.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    review_raw = await asyncio.to_thread(
        llm.chat,
        [{"role": "user", "content": _build_reviewer_prompt(chapter_text)}],
        None,
        True,
        "discord_reviewer",
        None,
        1800,
    )
    review_data = _parse_json_safe(review_raw)
    review_json_path.write_text(
        json.dumps(review_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reader_comment = (
        review_data.get("reader_comment", "") if isinstance(review_data, dict) else ""
    ).strip()
    one_line = (
        review_data.get("one_line_verdict", "") if isinstance(review_data, dict) else ""
    ).strip()

    review_md = _format_review_md(episode_id, cycle, review_data)
    review_md_path = run_dir / f"{episode_id}_cycle{cycle}_review.md"
    review_md_path.write_text(review_md, encoding="utf-8")

    channel_id = int(ctx["channel_id"])
    reviewer_bot_token = ctx.get("reviewer_bot_token", "")
    await _send_file_with_token(
        channel, channel_id, review_json_path, f"{episode_id} reader review json", reviewer_bot_token, required=True
    )
    # Keep markdown review report local only.
    await _send_text_with_token(
        channel,
        channel_id,
        "2) Reviewer Agent 감상평\n"
        f"{one_line}\n\n{reader_comment if reader_comment else '(리뷰 생성 실패: report 파일 확인)'}",
        reviewer_bot_token,
        required=True,
    )
    await _send_text_with_token(
        channel,
        channel_id,
        f"{REVIEW_DONE_TAG} cycle={cycle} episode={episode_id}",
        reviewer_bot_token,
        required=True,
    )

    return True, {
        **ctx,
        "quality_json": str(review_json_path),
        "review_md": str(review_md_path),
    }


async def run_manager_agent(
    channel: discord.abc.Messageable,
    cycle: int,
    job: JobConfig,
    branch_contexts: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    episode_id = str(branch_contexts[0].get("episode_id", "unknown")).strip() if branch_contexts else "unknown"
    run_dir = Path(job.run_output_dir)
    manager_md_path = run_dir / f"{episode_id}_cycle{cycle}_manager.md"

    branch_scores: list[dict[str, Any]] = []
    cycle_review_snippets: list[str] = []
    for ix, ctx in enumerate(branch_contexts, start=1):
        qpath = Path(str(ctx.get("quality_json", "")))
        rpath = Path(str(ctx.get("review_md", "")))
        qdata: dict[str, Any] = {}
        if qpath.exists():
            try:
                qdata = json.loads(qpath.read_text(encoding="utf-8"))
            except Exception:
                qdata = {}
        branch_scores.append(
            {
                "branch": f"B{ix:02d}",
                "thrill_score_10": qdata.get("thrill_score_10", "n/a"),
                "style_score_10": qdata.get("style_score_10", "n/a"),
                "one_line_verdict": qdata.get("one_line_verdict", ""),
            }
        )
        if rpath.exists():
            cycle_review_snippets.append(rpath.read_text(encoding="utf-8", errors="replace")[:2500])

    periodic_review_snippets: list[str] = []
    if job.manager_period > 0 and cycle % job.manager_period == 0:
        branch_dirs = sorted((Path(job.run_output_dir)).glob("branch_*"))
        for bdir in branch_dirs:
            for c in range(max(1, cycle - job.manager_period + 1), cycle + 1):
                p = bdir / f"{episode_id}_cycle{c}_review.md"
                if p.exists():
                    periodic_review_snippets.append(p.read_text(encoding="utf-8", errors="replace")[:1800])

    llm = LLMClient(
        model="gpt-4o-mini",
        premium_model="gpt-5-mini",
        budget_usd=4.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    manager_raw = await asyncio.to_thread(
        llm.chat,
        [{"role": "user", "content": _build_manager_prompt(cycle, branch_scores, cycle_review_snippets, periodic_review_snippets)}],
        None,
        True,
        "discord_manager",
        None,
        2200,
    )
    manager_data = _parse_json_safe(manager_raw)
    manager_md_path.write_text(
        _format_manager_md(episode_id, cycle, manager_data, branch_scores),
        encoding="utf-8",
    )

    manager_bot_token = job.manager_bot_token
    channel_id = int(job.channel_id)
    avg_thrill = 0.0
    avg_style = 0.0
    score_count = 0
    for row in branch_scores:
        try:
            avg_thrill += float(row.get("thrill_score_10"))
            avg_style += float(row.get("style_score_10"))
            score_count += 1
        except Exception:
            pass
    if score_count > 0:
        avg_thrill /= score_count
        avg_style /= score_count

    periodic_note = (
        f"\n주기 진단 실행: 최근 {job.manager_period}사이클 리뷰 종합 완료"
        if periodic_review_snippets
        else ""
    )
    manager_msg = (
        "M) Manager Agent 종합 분석 완료\n"
        f"- cycle: {cycle}\n"
        f"- branches: {len(branch_contexts)}\n"
        f"- avg thrill/style: {avg_thrill:.2f}/{avg_style:.2f}"
        f"{periodic_note}"
    )
    manager_message_id: int | None = None
    if manager_bot_token:
        try:
            manager_message_id = await _rest_send_text_return_message_id(channel_id, manager_msg, manager_bot_token)
        except Exception:
            manager_message_id = None
    if manager_message_id is None:
        sent = await channel.send(manager_msg)
        manager_message_id = int(sent.id)

    if max(avg_thrill, avg_style) >= 8.0 and manager_message_id is not None:
        if manager_bot_token:
            try:
                await _rest_add_reaction(channel_id, manager_message_id, "❤️", manager_bot_token)
            except Exception:
                try:
                    msg_obj = await channel.fetch_message(manager_message_id)
                    await msg_obj.add_reaction("❤️")
                except Exception:
                    pass
        else:
            try:
                msg_obj = await channel.fetch_message(manager_message_id)
                await msg_obj.add_reaction("❤️")
            except Exception:
                pass

    return True, {
        "episode_id": episode_id,
        "run_output_dir": str(run_dir),
        "review_md": str(manager_md_path),
        "channel_id": job.channel_id,
        "fixer_bot_token": job.fixer_bot_token,
        "reviewer_bot_token": job.reviewer_bot_token,
        "manager_bot_token": job.manager_bot_token,
    }


async def run_fixer_agent(channel: discord.abc.Messageable, cycle: int, ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    run_dir = Path(ctx["run_output_dir"])
    episode_id = ctx["episode_id"]
    review_md_path = Path(ctx["review_md"])
    channel_id = int(ctx["channel_id"])
    fixer_bot_token = ctx.get("fixer_bot_token", "")
    if FIXER_APPLY_LOCK.locked():
        await _send_text_with_token(
            channel,
            channel_id,
            "3) Fixer Agent 대기 중... (다른 fixer가 코드 수정 중)",
            fixer_bot_token,
            required=True,
        )

    async with FIXER_APPLY_LOCK:
        await _send_text_with_token(
            channel, channel_id, f"3) Fixer Agent 시작 (cycle {cycle})", fixer_bot_token, required=True
        )

        summary_out_path = run_dir / f"{episode_id}_cycle{cycle}_fixer_summary.md"
        before_candidates = await asyncio.to_thread(_list_worktree_change_candidates)
        before_hashes = {
            path: digest
            for path in before_candidates
            if (digest := _sha256_for_relpath(path)) is not None
        }
        protected_before_text: dict[str, str] = {}
        for rel in PROTECTED_FIXER_RULE_FILES:
            p = REPO_ROOT / rel
            if p.exists():
                protected_before_text[rel] = p.read_text(encoding="utf-8", errors="replace")
        # Hard snapshot: episode configs are immutable for Fixer.
        episode_cfg_before: dict[str, str] = {}
        for p in sorted((REPO_ROOT / "config" / "episodes").glob("*.yaml")):
            rel = str(p.relative_to(REPO_ROOT))
            episode_cfg_before[rel] = p.read_text(encoding="utf-8", errors="replace")

        rc, out, err = await asyncio.to_thread(_run_codex_fix, review_md_path, summary_out_path)
        if rc != 0:
            tail = (err or out)[-1500:]
            await _send_text_with_token(
                channel,
                channel_id,
                f"{RUN_END_TAG} fixer 실행 실패\n```{tail}```",
                fixer_bot_token,
                required=True,
            )
            return False, ctx

        protected_restore_notes: list[str] = []
        for rel, markers in PROTECTED_FIXER_RULE_FILES.items():
            p = REPO_ROOT / rel
            if not p.exists():
                continue
            current = p.read_text(encoding="utf-8", errors="replace")
            if all(marker in current for marker in markers):
                continue
            before_text = protected_before_text.get(rel)
            if before_text is not None:
                p.write_text(before_text, encoding="utf-8")
                protected_restore_notes.append(f"- {rel}: 금지 규칙 변경 감지 -> 자동 원복")
            else:
                protected_restore_notes.append(f"- {rel}: 금지 규칙 변경 감지 (원본 스냅샷 없음)")

        # Hard-restore any config/episodes changes (modify/create/delete).
        episode_cfg_notes: list[str] = []
        episode_cfg_after_paths = {
            str(p.relative_to(REPO_ROOT))
            for p in sorted((REPO_ROOT / "config" / "episodes").glob("*.yaml"))
        }
        before_paths = set(episode_cfg_before.keys())
        for rel in sorted(before_paths | episode_cfg_after_paths):
            abs_path = REPO_ROOT / rel
            if rel in episode_cfg_before and abs_path.exists():
                current = abs_path.read_text(encoding="utf-8", errors="replace")
                if current != episode_cfg_before[rel]:
                    abs_path.write_text(episode_cfg_before[rel], encoding="utf-8")
                    episode_cfg_notes.append(f"- {rel}: 변경 감지 -> 자동 원복")
            elif rel in episode_cfg_before and not abs_path.exists():
                abs_path.write_text(episode_cfg_before[rel], encoding="utf-8")
                episode_cfg_notes.append(f"- {rel}: 삭제 감지 -> 자동 복원")
            elif rel not in episode_cfg_before and abs_path.exists():
                abs_path.unlink()
                episode_cfg_notes.append(f"- {rel}: 신규 생성 감지 -> 자동 삭제")

        after_candidates = await asyncio.to_thread(_list_worktree_change_candidates)
        all_candidates = sorted(before_candidates | after_candidates)
        changed_entries: list[dict[str, Any]] = []
        changed_paths: list[str] = []
        for path in all_candidates:
            before_sha = before_hashes.get(path)
            after_sha = _sha256_for_relpath(path)
            if before_sha == after_sha:
                continue
            if before_sha is None and after_sha is not None:
                change_type = "created_or_untracked"
            elif before_sha is not None and after_sha is None:
                change_type = "deleted"
            else:
                change_type = "modified"
            changed_entries.append(
                {
                    "path": path,
                    "change_type": change_type,
                    "before_sha256": before_sha,
                    "after_sha256": after_sha,
                }
            )
            changed_paths.append(path)

        changed_paths = sorted(changed_paths)
        changed_files_json_path = run_dir / f"{episode_id}_cycle{cycle}_changed_files.json"
        changed_files_json_path.write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "cycle": cycle,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "changed_files": changed_entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        diff_path = run_dir / f"{episode_id}_cycle{cycle}_fix.diff"
        diff_cmd = ["git", "diff", "--", *changed_paths] if changed_paths else ["git", "diff", "--", "."]
        _, diff_text, _ = await asyncio.to_thread(_run_cmd, diff_cmd, 30)
        diff_path.write_text(diff_text or "# No diff\n", encoding="utf-8")
        summary_text = (
            summary_out_path.read_text(encoding="utf-8", errors="replace").strip()
            if summary_out_path.exists()
            else "(fixer 요약 파일 없음)"
        )
        if protected_restore_notes:
            summary_text += (
                "\n\n[보호 규칙 적용 결과]\n"
                + "\n".join(protected_restore_notes)
            )
        if episode_cfg_notes:
            summary_text += (
                "\n\n[에피소드 설정 보호 결과]\n"
                + "\n".join(episode_cfg_notes)
            )
        changed_text = "\n".join(f"- {p}" for p in changed_paths) if changed_paths else "- 없음"
        await _send_file_with_token(
            channel, channel_id, diff_path, f"{episode_id} fix diff", fixer_bot_token, required=True
        )
        await _send_file_with_token(
            channel,
            channel_id,
            changed_files_json_path,
            f"{episode_id} cycle changed files json",
            fixer_bot_token,
            required=True,
        )
        await _send_text_with_token(
            channel,
            channel_id,
            f"{FIX_DONE_TAG} cycle={cycle} episode={episode_id}\n"
            f"{summary_text}\n\n"
            "실제 변경 파일:\n"
            f"{changed_text}",
            fixer_bot_token,
            required=True,
        )
        return True, ctx


async def run_full_cycle(channel: discord.abc.Messageable, job: JobConfig) -> None:
    state = _load_state()
    run_key = f"{job.channel_id}:{job.seed_message_id}:{job.episode_key}"
    current = state.get(run_key, {})
    cycle_start = int(current.get("cycle", 0)) + 1

    cycle = cycle_start
    while True:
        if _is_stop_requested(job.channel_id):
            await channel.send(f"{RUN_END_TAG} 중지 완료: 사용자 요청({CMD_STOP})")
            _clear_stop(job.channel_id)
            return
        if job.max_cycles > 0 and cycle > job.max_cycles:
            break
        try:
            branch_count = max(1, int(job.parallel_branches))
            branch_dirs = [f"branch_{i:02d}" for i in range(1, branch_count + 1)]
            await channel.send(
                f"사이클 {cycle} 시작: 병렬 브랜치 {branch_count}개 실행\n"
                f"- run_root: `{job.run_output_dir}`\n"
                f"- branches: {', '.join(branch_dirs)}"
            )

            async def _run_branch(branch_idx: int) -> tuple[bool, dict[str, Any]]:
                branch_dir = Path(job.run_output_dir) / f"branch_{branch_idx:02d}"
                branch_dir.mkdir(parents=True, exist_ok=True)
                branch_job = replace(job, run_output_dir=str(branch_dir))
                ok1, ctx1 = await run_simulator_agent(channel, branch_job, cycle)
                if not ok1:
                    return ok1, ctx1
                ok2, ctx2 = await run_reviewer_agent(channel, cycle, ctx1)
                return ok2, ctx2

            branch_results = await asyncio.gather(*[_run_branch(i) for i in range(1, branch_count + 1)])
            failed = [(i + 1, ctx) for i, (ok, ctx) in enumerate(branch_results) if not ok]
            if failed:
                if any(isinstance(ctx, dict) and ctx.get("stopped") for _, ctx in failed):
                    _clear_stop(job.channel_id)
                    return
                await channel.send(
                    f"{RUN_END_TAG} 실패: 병렬 브랜치 {len(failed)}/{branch_count}개 실패"
                )
                return
            branch_ctxs = [ctx for ok, ctx in branch_results if ok]

            okm, manager_ctx = await run_manager_agent(channel, cycle, job, branch_ctxs)
            if not okm:
                await channel.send(f"{RUN_END_TAG} 실패: manager 단계")
                return

            ok3, _ = await run_fixer_agent(channel, cycle, manager_ctx)
            if not ok3:
                await channel.send(f"{RUN_END_TAG} 실패: fixer 단계")
                return

            state[run_key] = {"cycle": cycle, "updated_at": int(time.time())}
            _save_state(state)
            cycle += 1
        except Exception as exc:
            await _send_text(channel, f"{RUN_END_TAG} 예외 발생: {type(exc).__name__}: {exc}")
            return

    await channel.send(f"{RUN_END_TAG} 완료: max_cycles={job.max_cycles}")
    _clear_stop(job.channel_id)


async def async_main() -> None:
    os.chdir(REPO_ROOT)
    load_project_env(REPO_ROOT)
    _force_load_env_keys(
        ["OPENAI_API_KEY", "DISCORD_BOT_TOKEN", "DISCORD_BOT_TOKEN2", "DISCORD_BOT_TOKEN3", "DISCORD_BOT_TOKEN4"]
    )
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set DISCORD_BOT_TOKEN in .env")
    _resolve_openai_api_key()
    reviewer_bot_token, fixer_bot_token, manager_bot_token = _resolve_stage_bot_tokens()

    intents = discord.Intents.default()
    intents.message_content = True
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    client = discord.Client(intents=intents, connector=connector)
    cmd_parser = _build_cmd_parser()
    review_cmd_parser = _build_review_cmd_parser()
    fix_cmd_parser = _build_fix_cmd_parser()

    @client.event
    async def on_ready():
        print(f"Discord bot connected as {client.user}")

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return

        content = (message.content or "").strip()

        if content.startswith(CMD_RESET):
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            await message.channel.send("상태 파일 삭제 완료: `data/discord_loop_state.json`")
            return

        if content.startswith(CMD_STOP):
            _request_stop(message.channel.id)
            await message.channel.send(
                f"중지 요청 수신: {CMD_STOP}\n"
                "현재 실행 중인 단계가 정리되는 즉시 루프를 멈춥니다."
            )
            return

        if content.startswith(CMD_REVIEW):
            argv = shlex.split(content[len(CMD_REVIEW):].strip())
            try:
                args = review_cmd_parser.parse_args(argv)
            except SystemExit:
                await message.channel.send(
                    "사용법: !novel-review <chapter.md|run_dir> [--episode-id ep01_conference_shadow] [--cycle 1]"
                )
                return

            try:
                chapter_path, run_dir, episode_id = _resolve_review_target(args.target, args.episode_id)
            except Exception as exc:
                await message.channel.send(f"리뷰 타겟 해석 실패: {exc}")
                return

            try:
                await _send_text_with_token(
                    message.channel,
                    message.channel.id,
                    "리뷰 단독 실행 시작\n"
                    f"- episode_id: `{episode_id}`\n"
                    f"- chapter: `{chapter_path}`\n"
                    f"- run_dir: `{run_dir}`\n"
                    f"- cycle: `{args.cycle}`",
                    reviewer_bot_token,
                    required=True,
                )

                reviewer_ctx = {
                    "episode_id": episode_id,
                    "chapter": str(chapter_path),
                    "run_output_dir": str(run_dir),
                    "channel_id": message.channel.id,
                    "reviewer_bot_token": reviewer_bot_token,
                    "fixer_bot_token": fixer_bot_token,
                }
                ok, _ = await run_reviewer_agent(message.channel, int(args.cycle), reviewer_ctx)
                if not ok:
                    await message.channel.send(f"{RUN_END_TAG} 실패: reviewer 단계(단독)")
            except Exception as exc:
                await message.channel.send(f"{RUN_END_TAG} reviewer 단독 예외: {type(exc).__name__}: {exc}")
            return

        if content.startswith(CMD_FIX):
            argv = shlex.split(content[len(CMD_FIX):].strip())
            try:
                args = fix_cmd_parser.parse_args(argv)
            except SystemExit:
                await message.channel.send(
                    "사용법: !novel-fix <review.md|run_dir> [--episode-id ep01_conference_shadow] [--cycle 1] [--review-md path]"
                )
                return

            try:
                review_md_path, run_dir, episode_id = _resolve_fix_target(
                    args.target, args.episode_id, int(args.cycle), args.review_md
                )
            except Exception as exc:
                await message.channel.send(f"fix 타겟 해석 실패: {exc}")
                return

            try:
                await _send_text_with_token(
                    message.channel,
                    message.channel.id,
                    "fix 단독 실행 시작\n"
                    f"- episode_id: `{episode_id}`\n"
                    f"- review_md: `{review_md_path}`\n"
                    f"- run_dir: `{run_dir}`\n"
                    f"- cycle: `{args.cycle}`",
                    fixer_bot_token,
                    required=True,
                )

                fixer_ctx = {
                    "episode_id": episode_id,
                    "review_md": str(review_md_path),
                    "run_output_dir": str(run_dir),
                    "channel_id": message.channel.id,
                    "fixer_bot_token": fixer_bot_token,
                    "reviewer_bot_token": reviewer_bot_token,
                }
                ok, _ = await run_fixer_agent(message.channel, int(args.cycle), fixer_ctx)
                if not ok:
                    await message.channel.send(f"{RUN_END_TAG} 실패: fixer 단계(단독)")
            except Exception as exc:
                await message.channel.send(f"{RUN_END_TAG} fixer 단독 예외: {type(exc).__name__}: {exc}")
            return

        if not content.startswith(CMD_START):
            return

        argv = shlex.split(content[len(CMD_START):].strip())
        try:
            args = cmd_parser.parse_args(argv)
        except SystemExit:
            await message.channel.send(
                "사용법: !novel-loop <episode_key> [--max-cycles 3] [--target-words 2200] "
                "[--scenes 6] [--budget 4.0] [--protagonist kim_sumin] "
                "[--parallel 5] [--manager-period 5]\n"
                "참고: --max-cycles 0 이면 중지할 때까지 무한 반복"
            )
            return

        await message.channel.send("명령 수신 완료. 작업을 시작합니다.")
        _clear_stop(message.channel.id)
        run_date, run_id, run_dir = _allocate_run_output_dir()
        await message.channel.send(
            f"- run: `{run_date}/{run_id}`\n"
        )
        job = JobConfig(
            channel_id=message.channel.id,
            episode_key=args.episode_key,
            max_cycles=args.max_cycles,
            target_words=args.target_words,
            scenes=args.scenes,
            budget=args.budget,
            protagonist=args.protagonist,
            seed_message_id=message.id,
            run_date=run_date,
            run_id=run_id,
            run_output_dir=str(run_dir),
            reviewer_bot_token=reviewer_bot_token,
            fixer_bot_token=fixer_bot_token,
            manager_bot_token=manager_bot_token,
            parallel_branches=max(1, int(args.parallel)),
            manager_period=max(1, int(args.manager_period)),
        )
        asyncio.create_task(run_full_cycle(message.channel, job))

    await client.start(token)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
