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
import json
import os
import re
import shlex
import subprocess
import time
import sys
import ssl
from datetime import datetime
from dataclasses import dataclass
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


async def _run_cmd_stream(
    cmd: list[str],
    timeout_sec: int = 3600,
    extra_env: dict[str, str] | None = None,
    on_line: Any = None,
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

    async def _drain() -> int:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
            lines.append(line)
            if on_line is not None:
                await on_line(line.rstrip("\n"))
        return await proc.wait()

    try:
        rc = await asyncio.wait_for(_drain(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "".join(lines), "timeout"
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


def _resolve_stage_bot_tokens() -> tuple[str, str]:
    reviewer_bot = _env_value("DISCORD_BOT_TOKEN2", "TOKEN2", "token2")
    fixer_bot = _env_value("DISCORD_BOT_TOKEN3", "TOKEN3", "token3")
    return reviewer_bot, fixer_bot


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
        "You are Fixer Agent.\n"
        "Given reader review and current code snippets, identify root cause and propose real code edits.\n"
        "Return strict JSON:\n"
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
        "Rules:\n"
        "- Edit code only. Do NOT edit config/episodes files.\n"
        "- Focus on readability/style issues from the reader review.\n"
        "- Use exact find/replace snippets that exist in files.\n"
        "- Keep edits small and safe.\n\n"
        f"Reader review:\n{review_md[:9000]}\n\n"
        f"Code context:\n{code_context[:18000]}"
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


def _build_cmd_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=CMD_START, add_help=False)
    p.add_argument("episode_key")
    p.add_argument("--max-cycles", type=int, default=3)
    p.add_argument("--target-words", type=int, default=0)
    p.add_argument("--scenes", type=int, default=0)
    p.add_argument("--budget", type=float, default=4.0)
    p.add_argument("--protagonist", default="")
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

    await channel.send(
        f"1) Simulator Agent 시작 (cycle {cycle})\n- episode: `{episode_id}`"
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
                    f"시뮬레이션 진행: {checkpoint}% (Turn {turn}/{total})"
                )

    rc, _, err = await _run_cmd_stream(
        sim_cmd,
        3600,
        {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
        _on_sim_line,
    )
    if rc != 0:
        await _send_text(channel, f"시뮬레이션 실패\n```{err[-1500:]}```")
        return False, {}
    await channel.send("시뮬레이션 완료. 챕터 생성을 시작합니다.")

    gen_cmd = [
        "python3", "generate_chapter.py",
        "--episode", episode_id,
        "--episode-config", str(episode_file),
        "--protagonist", protagonist,
        "--output", str(run_dir),
        "--budget", str(job.budget),
    ]
    if job.target_words > 0:
        gen_cmd.extend(["--words", str(job.target_words)])
    if job.scenes > 0:
        gen_cmd.extend(["--scenes", str(job.scenes)])
    await channel.send("챕터 생성 진행 중...")
    rc, _, err2 = await asyncio.to_thread(
        _run_cmd,
        gen_cmd,
        3600,
        {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
    )
    if rc != 0:
        await _send_text(channel, f"챕터 생성 실패\n```{err2[-1500:]}```")
        return False, {}

    chapter = run_dir / f"{episode_id}_chapter.md"
    if not chapter.exists():
        chapter = _find_latest(run_dir, f"{episode_id}*chapter.md")
    if chapter is None:
        await channel.send("챕터 파일을 찾지 못했습니다.")
        return False, {}

    # Keep chapter markdown local only; notify completion in channel.
    await channel.send(f"챕터 생성 완료 (로컬 저장): `{chapter}`")
    await channel.send(f"{SIM_DONE_TAG} cycle={cycle} episode={episode_id} chapter={chapter.name}")

    return True, {
        "episode_id": episode_id,
        "episode_file": str(episode_file),
        "chapter": str(chapter),
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


async def run_fixer_agent(channel: discord.abc.Messageable, cycle: int, ctx: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    run_dir = Path(ctx["run_output_dir"])
    episode_id = ctx["episode_id"]
    review_md = Path(ctx["review_md"]).read_text(encoding="utf-8")
    code_context = _build_code_context_for_fixer()

    llm = LLMClient(
        model="gpt-4o-mini",
        premium_model="gpt-5-mini",
        budget_usd=3.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    channel_id = int(ctx["channel_id"])
    fixer_bot_token = ctx.get("fixer_bot_token", "")
    await _send_text_with_token(
        channel, channel_id, f"3) Fixer Agent 시작 (cycle {cycle})", fixer_bot_token, required=True
    )
    fixer_raw = await asyncio.to_thread(
        llm.chat,
        [{"role": "user", "content": _build_fixer_prompt(review_md, code_context)}],
        None,
        True,
        "discord_fixer",
        None,
        2200,
    )
    fixer_data = _parse_json_safe(fixer_raw)
    summary = fixer_data.get("change_summary", []) if isinstance(fixer_data, dict) else []
    analysis = fixer_data.get("root_cause_analysis", []) if isinstance(fixer_data, dict) else []
    edits = fixer_data.get("edits", []) if isinstance(fixer_data, dict) else []
    edits = edits if isinstance(edits, list) else []

    applied, failed, changed_paths = _apply_code_edits(edits)
    diff_path = run_dir / f"{episode_id}_cycle{cycle}_fix.diff"
    diff_cmd = ["git", "diff", "--", *changed_paths] if changed_paths else ["git", "diff", "--", "."]
    _, diff_text, _ = await asyncio.to_thread(_run_cmd, diff_cmd, 30)
    diff_path.write_text(diff_text or "# No diff\n", encoding="utf-8")

    analysis_text = "\n".join([f"- {s}" for s in analysis]) if analysis else "- (분석 생성 없음)"
    summary_text = "\n".join([f"- {s}" for s in summary]) if summary else "- (수정 요약 없음)"
    applied_text = "\n".join([f"- {s}" for s in applied]) if applied else "- (적용된 코드 변경 없음)"
    failed_text = "\n".join([f"- {s}" for s in failed]) if failed else "- 없음"
    await _send_file_with_token(
        channel, channel_id, diff_path, f"{episode_id} fix diff", fixer_bot_token, required=True
    )
    await _send_text_with_token(
        channel,
        channel_id,
        f"{FIX_DONE_TAG} cycle={cycle} episode={episode_id}\n"
        "Fixer 고민(원인 분석):\n"
        f"{analysis_text}\n\n"
        "코드 수정 요약:\n"
        f"{summary_text}\n\n"
        "적용된 코드 변경:\n"
        f"{applied_text}\n\n"
        "적용 실패 항목:\n"
        f"{failed_text}",
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
        if job.max_cycles > 0 and cycle > job.max_cycles:
            break
        try:
            ok1, ctx = await run_simulator_agent(channel, job, cycle)
            if not ok1:
                await channel.send(f"{RUN_END_TAG} 실패: simulator 단계")
                return

            ok2, ctx = await run_reviewer_agent(channel, cycle, ctx)
            if not ok2:
                await channel.send(f"{RUN_END_TAG} 실패: reviewer 단계")
                return

            ok3, _ = await run_fixer_agent(channel, cycle, ctx)
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


async def async_main() -> None:
    os.chdir(REPO_ROOT)
    load_project_env(REPO_ROOT)
    _force_load_env_keys(
        ["OPENAI_API_KEY", "DISCORD_BOT_TOKEN", "DISCORD_BOT_TOKEN2", "DISCORD_BOT_TOKEN3"]
    )
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set DISCORD_BOT_TOKEN in .env")
    _resolve_openai_api_key()
    reviewer_bot_token, fixer_bot_token = _resolve_stage_bot_tokens()

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
                "[--scenes 6] [--budget 4.0] [--protagonist kim_sumin]\n"
                "참고: --max-cycles 0 이면 중지할 때까지 무한 반복"
            )
            return

        await message.channel.send("명령 수신 완료. 작업을 시작합니다.")
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
        )
        asyncio.create_task(run_full_cycle(message.channel, job))

    await client.start(token)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
