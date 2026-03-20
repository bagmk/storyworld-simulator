#!/usr/bin/env python3
"""
Discord bot for the Novel Writer project.

Supports: !novel-daily, !meitner, !status, !stop, !chapter, !approve, !reject
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import ssl
import urllib.parse
from datetime import datetime
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
from tools.daily_pipeline import run_daily_pipeline
from tools.config_guardian import _assert_not_locked


ROOT_OUTPUT_DIR = REPO_ROOT / "output"

DAILY_TAG = ""

CMD_MEITNER = "!meitner"
CMD_DAILY = "!novel-daily"
CMD_APPROVE = "!approve"
CMD_REJECT = "!reject"
CMD_STATUS = "!status"
CMD_PIPELINE_STOP = "!stop"
CMD_CHAPTER = "!chapter"

# Daily pipeline: per-channel state
DAILY_FEEDBACK_QUEUES: dict[int, asyncio.Queue] = {}
DAILY_STOP_EVENTS: dict[int, asyncio.Event] = {}
DAILY_STATUS: dict[int, str] = {}           # channel_id → current status string
DAILY_WAITING_FEEDBACK: set[int] = set()   # 스코어카드 이후 피드백 대기 중인 채널만 등록
DAILY_PROCESS_INFO: dict[int, dict[str, Any]] = {}
DAILY_SESSION_METRICS: dict[int, dict[str, Any]] = {}
DAILY_CHAPTER_PATHS: dict[int, Path] = {}  # channel_id → 최근 생성된 챕터 파일 경로
DAILY_PENDING_REVIEW_TIER: dict[int, dict[str, Any]] = {}
DAILY_START_TIMES: dict[int, float] = {}   # channel_id → pipeline start time (monotonic)
DAILY_EPISODE_KEYS: dict[int, str] = {}    # channel_id → episode key (kept after stop for chart)


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _parse_review_tier_choice(text: str) -> str | None:
    t = (text or "").strip().lower()
    if t in {"1", "mini", "min", "저렴", "빠르게", "가볍게"}:
        return "mini"
    if t in {"2", "premium", "prem", "프리미엄", "정밀", "고품질"}:
        return "premium"
    if t in {"3", "claude", "클로드", "코덱스", "codex", "무료", "free"}:
        return "codex"
    return None


def _top_level_repo_snapshot() -> str:
    lines: list[str] = []
    for path in sorted(REPO_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if path.name.startswith("."):
            continue
        kind = "dir" if path.is_dir() else "file"
        lines.append(f"- {path.name} ({kind})")
    return "\n".join(lines[:40])


def _repo_file_inventory(limit: int = 220) -> str:
    rc, out, _ = _run_cmd(
        ["rg", "--files", "README.md", "src", "tools", "config", "tests"],
        timeout_sec=20,
    )
    if rc != 0:
        return ""
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return "\n".join(files[:limit])


def _extract_repo_search_terms(question: str, limit: int = 6) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z0-9_./-]{2,}|[가-힣]{2,}", question or "")
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "what", "where", "when", "how",
        "can", "does", "repo", "repository", "code", "file", "files", "function", "functions",
        "please", "about", "into", "have", "show", "tell", "there", "them", "they",
        "지금", "이거", "그거", "저거", "어디", "무엇", "뭐", "어떻게", "가능", "있니", "있어",
        "파일", "코드", "저장소", "리포", "함수", "구조", "설명", "연결", "대화",
    }
    seen: set[str] = set()
    terms: list[str] = []
    for term in raw_terms:
        normalized = term.strip().lower()
        if len(normalized) < 2 or normalized in stopwords:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term.strip())
        if len(terms) >= limit:
            break
    return terms


def _repo_search_excerpt(question: str, max_lines: int = 120) -> str:
    terms = _extract_repo_search_terms(question)
    if not terms:
        return ""
    cmd = ["rg", "-n", "-S"]
    for term in terms:
        cmd.extend(["-e", term])
    cmd.extend(["README.md", "src", "tools", "config", "tests"])
    rc, out, err = _run_cmd(cmd, timeout_sec=25)
    if rc not in (0, 1):
        return f"(search failed: {err[:240]})"
    matches = [line for line in out.splitlines() if line.strip()]
    return "\n".join(matches[:max_lines])


def _read_repo_reference_snippets() -> str:
    snippets: list[str] = []
    references = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "tools" / "discord_loop_bot.py",
        REPO_ROOT / "src" / "novel_writer" / "scene_distiller.py",
    ]
    for path in references:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[:3000].strip()
        if not text:
            continue
        rel = path.relative_to(REPO_ROOT)
        snippets.append(f"## {rel}\n{text}")
    return "\n\n".join(snippets)


def _build_meitner_prompt(question: str) -> str:
    repo_map = _top_level_repo_snapshot()
    inventory = _repo_file_inventory()
    search_excerpt = _repo_search_excerpt(question)
    references = _read_repo_reference_snippets()
    return (
        "User question:\n"
        f"{question.strip()}\n\n"
        "Top-level repo snapshot:\n"
        f"{repo_map or '(unavailable)'}\n\n"
        "File inventory:\n"
        f"{inventory or '(unavailable)'}\n\n"
        "Search results:\n"
        f"{search_excerpt or '(no targeted matches)'}\n\n"
        "Reference snippets:\n"
        f"{references or '(unavailable)'}\n"
    )


async def run_meitner_agent(
    channel: discord.abc.Messageable,
    question: str,
    channel_id: int,
    bot_token: str,
) -> None:
    prompt = _build_meitner_prompt(question)
    llm = LLMClient(
        model="gpt-4o-mini",
        premium_model="gpt-5-mini",
        budget_usd=2.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    system = (
        "You are Meitner, a repository explorer for this Novel Writer codebase. "
        "Answer only from the provided repository context. If context is insufficient, say so clearly. "
        "Keep answers concise, practical, and cite concrete repo paths in backticks."
    )
    answer = await asyncio.to_thread(
        llm.chat,
        [{"role": "user", "content": prompt}],
        system,
        True,
        "discord_meitner",
        None,
        1000,
    )
    await _send_text_with_token(channel, channel_id, answer, bot_token, required=False)


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


def _find_latest(base_dir: Path, path_pattern: str) -> Path | None:
    files = sorted(base_dir.glob(path_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _make_stop_chart(episode_key: str) -> Path | None:
    """
    !stop 시 현재까지의 사이클별 5개 점수를 matplotlib 차트로 생성.
    가장 최근 run_dir을 자동 탐색. PNG 임시 파일 경로 반환.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import tempfile
    except ImportError:
        return None

    # 가장 최근 run_dir 탐색
    daily_dir = REPO_ROOT / "output" / "daily"
    if not daily_dir.exists():
        return None
    run_dirs = sorted(daily_dir.glob(f"*_{episode_key}/*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        # episode_key 없이 최신 폴더로 대체
        run_dirs = sorted(daily_dir.glob("*/*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        return None
    run_dir = run_dirs[0]

    SCORE_DEFS = [
        ("thrill_score_10",        "긴장감",  "#e74c3c"),
        ("style_score_10",         "문체",    "#3498db"),
        ("causality_score_10",     "인과성",  "#9b59b6"),
        ("character_score_10",     "캐릭터",  "#f39c12"),
        ("scene_function_score_10","씬기능",  "#2ecc71"),
    ]

    review_files = sorted(run_dir.glob("auto_review_cycle*.json"))
    cycles: list[int] = []
    score_data: dict[str, list[float]] = {k: [] for k, _, _ in SCORE_DEFS}
    avg_scores: list[float] = []

    for rf in review_files:
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
            m = re.search(r"cycle(\d+)", rf.stem)
            if not m:
                continue
            cycles.append(int(m.group(1)))
            row_vals = []
            for k, _, _ in SCORE_DEFS:
                v = float(data.get(k, 0))
                score_data[k].append(v)
                row_vals.append(v)
            avg_scores.append(sum(row_vals) / len(row_vals))
        except Exception:
            pass

    if not cycles:
        return None

    # 한글 폰트
    _available = {f.name for f in fm.fontManager.ttflist}
    for _kf in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
        if _kf in _available:
            plt.rcParams["font.family"] = _kf
            plt.rcParams["axes.unicode_minus"] = False
            break

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(cycles)))

    for k, label, color in SCORE_DEFS:
        ax.plot(x, score_data[k], "o-", color=color, label=label, linewidth=2, markersize=6)
    ax.plot(x, avg_scores, "^--", color="#555555", label="평균", linewidth=1.5, markersize=5, alpha=0.8)
    ax.axhline(y=8.5, color="#e67e22", linestyle=":", linewidth=1.5, label="목표 8.5")

    ax.set_ylim(0, 10.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"사이클{c}" for c in cycles], fontsize=9)
    ax.set_ylabel("점수 (/ 10)", fontsize=11)
    ax.set_title(f"[{episode_key}] 중단 시점 — 사이클별 AI 리뷰 점수", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    for xi, yi in zip(x, avg_scores):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    plt.tight_layout()
    tmp = Path(tempfile.mktemp(suffix=".png", prefix="stop_chart_"))
    fig.savefig(str(tmp), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return tmp


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


async def _send_text_reply(
    channel: discord.abc.Messageable,
    parent_message: discord.Message,
    text: str,
) -> None:
    content = (text or "").strip()
    if not content:
        return
    limit = 1900
    ref = parent_message.to_reference(fail_if_not_exists=False)
    for i in range(0, len(content), limit):
        await channel.send(
            content[i:i + limit],
            reference=ref,
            mention_author=False,
        )


async def _send_text_in_thread(
    thread: discord.abc.Messageable,
    text: str,
) -> None:
    content = (text or "").strip()
    if not content:
        return
    limit = 1900
    for i in range(0, len(content), limit):
        await thread.send(content[i:i + limit])


async def _add_reaction_safe(message: discord.Message | None, emoji: str) -> None:
    if message is None:
        return
    try:
        await message.add_reaction(emoji)
    except Exception:
        pass


def _is_daily_report_text(text: str) -> bool:
    markers = (
        "[GUARDIAN] Config 규칙 검수 결과:",
        "[GUARDIAN] 🧠 GPT 분석 리포트:",
        "[REVIEW] ✅ 자동 검수 완료",
    )
    return any(marker in (text or "") for marker in markers)


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


async def _send_text_with_token_return_message(
    channel: discord.abc.Messageable,
    text: str,
    bot_token: str,
) -> discord.Message | None:
    if bot_token:
        try:
            channel_id = int(getattr(channel, "id"))
            message_id = await _rest_send_text_return_message_id(channel_id, text, bot_token)
            if message_id and hasattr(channel, "fetch_message"):
                try:
                    return await channel.fetch_message(message_id)
                except Exception:
                    return None
            return None
        except Exception:
            pass
    try:
        return await channel.send(text)
    except Exception:
        await _send_text(channel, text)
        return None


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
    reviewer_bot_token = reviewer_bot_token or token
    fixer_bot_token = fixer_bot_token or token
    manager_bot_token = manager_bot_token or token

    intents = discord.Intents.default()
    intents.message_content = True
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    client = discord.Client(intents=intents, connector=connector)

    @client.event
    async def on_ready():
        print(f"Discord bot connected as {client.user}")

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return

        content = (message.content or "").strip()
        if not content:
            return

        parts = content.split(None, 1)
        command = parts[0].lower()
        arg_text = parts[1].strip() if len(parts) > 1 else ""

        pending_review_tier = DAILY_PENDING_REVIEW_TIER.get(message.channel.id)
        if pending_review_tier and message.author.id == pending_review_tier.get("user_id"):
            if content.lower() in {"취소", "cancel", "c"}:
                DAILY_PENDING_REVIEW_TIER.pop(message.channel.id, None)
                await _send_text_with_token(
                    message.channel,
                    message.channel.id,
                    "리뷰 등급 선택을 취소했습니다.",
                    manager_bot_token,
                )
                return
            if not content.startswith("!"):
                chosen_tier = _parse_review_tier_choice(content)
                if chosen_tier:
                    DAILY_PENDING_REVIEW_TIER.pop(message.channel.id, None)
                    command = CMD_DAILY
                    arg_text = f"{pending_review_tier.get('arg_text', '')} --review-tier {chosen_tier}".strip()
                else:
                    await _send_text_with_token(
                        message.channel,
                        message.channel.id,
                        "리뷰 등급을 `mini` 또는 `premium`으로 보내주세요. `1/2`도 됩니다. 취소는 `취소`.",
                        manager_bot_token,
                    )
                    return

        if command == CMD_MEITNER:
            question = arg_text
            if not question:
                await message.channel.send("사용법: !meitner <repo에 대해 물어볼 내용>")
                return
            await message.channel.send("Meitner가 저장소를 탐색하는 중입니다.")
            try:
                await run_meitner_agent(
                    message.channel,
                    question,
                    message.channel.id,
                    token,
                )
            except Exception as exc:
                await message.channel.send(f"Meitner 실행 실패: {type(exc).__name__}: {exc}")
            return

        # ── !status ───────────────────────────────────────────────────────────
        if command == CMD_STATUS and not arg_text:
            ch_id = message.channel.id
            status = DAILY_STATUS.get(ch_id)
            if status:
                proc_info = DAILY_PROCESS_INFO.get(ch_id, {})
                metrics = DAILY_SESSION_METRICS.get(ch_id, {})
                pid = proc_info.get("pid")
                stage = proc_info.get("stage")
                if _pid_alive(pid):
                    proc_line = f"\n🧠 백그라운드 프로세스: 실행 중 (`{stage}`, PID {pid})"
                elif pid:
                    proc_line = f"\n🧠 백그라운드 프로세스: 종료됨 (`{stage}`, PID {pid})"
                else:
                    proc_line = "\n🧠 백그라운드 프로세스: 없음"
                token_line = (
                    f"\n🔢 세션 누적 토큰: {int(metrics.get('total_tokens', 0)):,}"
                    f" ({int(metrics.get('prompt_tokens', 0)):,} in + {int(metrics.get('completion_tokens', 0)):,} out)"
                )
                cost_line = (
                    f"\n💸 세션 누적 비용(Codex CLI 제외): ${float(metrics.get('simulation', 0.0)) + float(metrics.get('chapter', 0.0)) + float(metrics.get('auto_chapter', 0.0)) + float(metrics.get('auto_review', 0.0)) + float(metrics.get('final_review', 0.0)) + float(metrics.get('feedback_parse', 0.0)):.4f}"
                )
                start_t = DAILY_START_TIMES.get(ch_id)
                if start_t is not None:
                    _el = time.monotonic() - start_t
                    _em, _es = int(_el // 60), int(_el % 60)
                    elapsed_line = f"\n⏱️ 경과 시간: {_em}분 {_es:02d}초"
                else:
                    elapsed_line = ""
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    f"📊 현재 파이프라인 상태: **{status}**{elapsed_line}{proc_line}{token_line}{cost_line}",
                    manager_bot_token,
                )
            else:
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    "📊 현재 실행 중인 파이프라인 없음.",
                    manager_bot_token,
                )
            return

        # ── !stop ─────────────────────────────────────────────────────────────
        if command == CMD_PIPELINE_STOP and not arg_text:
            ch_id = message.channel.id
            ev = DAILY_STOP_EVENTS.get(ch_id)
            if ev and not ev.is_set():
                ev.set()
                DAILY_STATUS[ch_id] = "🛑 중단 요청됨..."
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    "🛑 중단 요청 전송. 현재 단계가 끝나는 즉시 파이프라인을 멈춥니다.",
                    manager_bot_token,
                )
                # 중단 시점 점수 차트 생성 & 전송
                ep_key = DAILY_EPISODE_KEYS.get(ch_id)
                if ep_key:
                    try:
                        chart_path = await asyncio.to_thread(_make_stop_chart, ep_key)
                        if chart_path and chart_path.exists():
                            await _send_file_with_token(
                                message.channel,
                                ch_id,
                                chart_path,
                                f"📊 [{ep_key}] 중단 시점 — 사이클별 점수 현황",
                                manager_bot_token,
                            )
                            chart_path.unlink(missing_ok=True)
                    except Exception:
                        pass  # 차트 실패해도 중단은 정상 처리됨
            else:
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    "⚠️ 실행 중인 파이프라인이 없거나 이미 중단됐습니다.",
                    manager_bot_token,
                )
            return

        # ── !chapter ──────────────────────────────────────────────────────────
        if command == CMD_CHAPTER:
            ch_id = message.channel.id
            chapter_path = DAILY_CHAPTER_PATHS.get(ch_id)
            if chapter_path is None or not chapter_path.exists():
                chapter_path = _find_latest(REPO_ROOT / "output" / "daily", "*/*/*_chapter.txt") \
                    or _find_latest(REPO_ROOT / "output" / "daily", "*/*/chapter.txt") \
                    or _find_latest(REPO_ROOT / "output" / "daily", "*/*/*_chapter.md")
                if chapter_path and chapter_path.exists():
                    DAILY_CHAPTER_PATHS[ch_id] = chapter_path
            if chapter_path and chapter_path.exists():
                word_count = len(chapter_path.read_text(encoding="utf-8", errors="replace").split())
                try:
                    await _send_file(message.channel, chapter_path, f"📖 최근 챕터 — {word_count}단어 (`{chapter_path.name}`)")
                except Exception:
                    await _send_text_with_token(
                        message.channel, ch_id,
                        f"📖 챕터 파일 경로: `{chapter_path.relative_to(REPO_ROOT)}` ({word_count}단어)\n파일 전송 실패 — 직접 열어보세요.",
                        manager_bot_token,
                    )
            else:
                await _send_text_with_token(
                    message.channel, ch_id,
                    "⚠️ 이 채널에 저장된 챕터가 없습니다. `!novel-daily <번호>`를 먼저 실행하세요.",
                    manager_bot_token,
                )
            return

        # ── !novel-daily <episode_key> ─────────────────────────────────────────
        if command == CMD_DAILY:
            daily_args = arg_text.split()
            if not daily_args:
                await message.channel.send(
                    "사용법: `!novel-daily <번호 또는 episode_key>`\n"
                    "예: `!novel-daily 1` / `!novel-daily 15`\n"
                    "옵션: `--target-words 3500 --budget 4.0 --protagonist kim_sumin --review-tier mini|premium`"
                )
                return
            episode_key = daily_args[0]
            tw = 3500
            budget_val = 4.0
            protagonist = "kim_sumin"
            review_tier: str | None = None
            remaining = daily_args[1:]
            i = 0
            while i < len(remaining):
                if remaining[i] == "--target-words" and i + 1 < len(remaining):
                    tw = int(remaining[i + 1]); i += 2
                elif remaining[i] == "--budget" and i + 1 < len(remaining):
                    budget_val = float(remaining[i + 1]); i += 2
                elif remaining[i] == "--protagonist" and i + 1 < len(remaining):
                    protagonist = remaining[i + 1]; i += 2
                elif remaining[i] == "--review-tier" and i + 1 < len(remaining):
                    review_tier = _parse_review_tier_choice(remaining[i + 1]); i += 2
                else:
                    i += 1

            ch_id = message.channel.id

            if review_tier is None:
                DAILY_PENDING_REVIEW_TIER[ch_id] = {
                    "user_id": message.author.id,
                    "arg_text": arg_text,
                }
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    "리뷰 등급을 골라주세요.\n"
                    "`1` 또는 `mini` — GPT-4o-mini (빠르고 저렴)\n"
                    "`2` 또는 `premium` — GPT-4o (정밀, 비쌈)\n"
                    "`3` 또는 `codex` — Codex CLI (OpenAI 비용 없음)\n"
                    "답장으로 보내주시면 시작합니다. 취소는 `취소`.",
                    manager_bot_token,
                )
                return
            DAILY_PENDING_REVIEW_TIER.pop(ch_id, None)

            # Cancel previous pipeline if still running
            old_ev = DAILY_STOP_EVENTS.get(ch_id)
            if old_ev and not old_ev.is_set():
                old_ev.set()

            feedback_q: asyncio.Queue = asyncio.Queue()
            stop_ev: asyncio.Event = asyncio.Event()
            DAILY_FEEDBACK_QUEUES[ch_id] = feedback_q
            DAILY_STOP_EVENTS[ch_id] = stop_ev
            DAILY_STATUS[ch_id] = f"시작 대기 — {episode_key}"
            DAILY_EPISODE_KEYS[ch_id] = episode_key  # for stop chart
            DAILY_START_TIMES[ch_id] = time.monotonic()
            DAILY_SESSION_METRICS[ch_id] = {
                "simulation": 0.0,
                "chapter": 0.0,
                "auto_chapter": 0.0,
                "auto_review": 0.0,
                "final_review": 0.0,
                "feedback_parse": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            def _set_status(s: str) -> None:
                DAILY_STATUS[ch_id] = s

            def _set_process(stage: str | None, pid: int | None, command_text: str | None) -> None:
                if stage is None or pid is None:
                    DAILY_PROCESS_INFO.pop(ch_id, None)
                    return
                DAILY_PROCESS_INFO[ch_id] = {
                    "stage": stage,
                    "pid": pid,
                    "command": command_text or "",
                }

            def _set_metrics(metrics: dict[str, Any]) -> None:
                DAILY_SESSION_METRICS[ch_id] = dict(metrics)

            def _on_start_wait() -> None:
                """파이프라인이 피드백 대기 상태 진입 시 호출."""
                DAILY_WAITING_FEEDBACK.add(ch_id)

            def _on_end_wait() -> None:
                """피드백 수신 또는 파이프라인 종료 시 호출."""
                DAILY_WAITING_FEEDBACK.discard(ch_id)

            anchor_messages: dict[str, discord.Message | None] = {
                "start": None,
                "guardian_rules": None,
                "guardian_gpt": None,
                "manager": None,
                "programmer": None,
                "sim": None,
                "chapter": None,
                "review": None,
                "auto": None,
                "auto_review": None,
                "fixer": None,
                "yaml_fixer": None,
                "choice": None,
            }
            anchor_threads: dict[str, discord.abc.Messageable | None] = {
                "start": None,
                "guardian_rules": None,
                "guardian_gpt": None,
                "manager": None,
                "programmer": None,
                "sim": None,
                "chapter": None,
                "review": None,
                "auto": None,
                "auto_review": None,
                "fixer": None,
                "yaml_fixer": None,
                "choice": None,
            }

            def _token_for_key(key: str | None) -> str:
                if key in {"start", "manager", "choice"}:
                    return manager_bot_token
                if key in {"guardian_rules", "guardian_gpt", "review", "auto_review"}:
                    return reviewer_bot_token
                if key in {"auto", "fixer", "yaml_fixer", "programmer"}:
                    return fixer_bot_token
                return ""

            def _anchor_key_for_text(text: str) -> str | None:
                if text.startswith(f"{DAILY_TAG}[START] 🎬 "):
                    return "start"
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🔍 Config 검수 중"):
                    return "guardian_rules"
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🤖 GPT 컨텍스트 분석 중"):
                    return "guardian_gpt"
                if text.startswith(f"{DAILY_TAG}[MANAGER] 🧠 "):
                    return "manager"
                if text.startswith(f"{DAILY_TAG}[PROGRAMMER] 🧪 코드 검수 시작"):
                    return "programmer"
                if text.startswith(f"{DAILY_TAG}[SIM] ⚙️ 시뮬레이션 시작"):
                    return "sim"
                if text.startswith(f"{DAILY_TAG}[CHAPTER] 📖 챕터 생성 중"):
                    return "chapter"
                if text.startswith(f"{DAILY_TAG}[REVIEW] 🔍 품질 자동 검수 중"):
                    return "review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 🚀 "):
                    return "auto"
                if text.startswith(f"{DAILY_TAG}[AUTO] 🔄 AI 자동 개선 루프"):
                    return "auto"  # 사이클마다 새 뜨레드 생성
                if text.startswith(f"{DAILY_TAG}[AUTO] 📊 AI 리뷰 결과"):
                    return "auto_review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 📋 Codex 수정 진단"):
                    return "auto_review"  # Reviewer 봇 스레드에 진단 내용 게시
                if text.startswith(f"{DAILY_TAG}[FIXER] 🔧 Codex 수정 시작"):
                    return "fixer"
                if text.startswith(f"{DAILY_TAG}[FIXER] 🔍 YAML 검수 시작"):
                    return "yaml_fixer"
                if text.startswith(f"{DAILY_TAG}[CHOICE] 📋 "):
                    return "choice"
                return None

            def _thread_route_for_text(text: str) -> str | None:
                if (
                    text.startswith(f"{DAILY_TAG}[START] run:")
                ):
                    return "start"
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] Config 규칙 검수 결과:") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ⚠️ Config 변경 요청"):
                    return "guardian_rules"
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🧠 GPT 분석 리포트:") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ✅ Config 검수 완료") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ⚠️ GPT 분석 실패"):
                    return "guardian_gpt"
                if text.startswith(f"{DAILY_TAG}[MANAGER] "):
                    if "🧠 매니저 분석" not in text:
                        return "manager"
                if text.startswith(f"{DAILY_TAG}[PROGRAMMER] "):
                    if "🧪 코드 검수 시작" not in text:
                        return "programmer"
                if text.startswith(f"{DAILY_TAG}[SIM] "):
                    if "시뮬레이션 시작" not in text:
                        return "sim"
                if text.startswith(f"{DAILY_TAG}[CHAPTER] "):
                    if "챕터 생성 중" not in text:
                        return "chapter"
                if text.startswith(f"{DAILY_TAG}[REVIEW] "):
                    if "품질 자동 검수 중" not in text:
                        return "review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 🧾 AI 리뷰 상세"):
                    return "auto_review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 📋 Codex 수정 진단"):
                    return "auto_review"
                if text.startswith(f"{DAILY_TAG}[AUTO] "):
                    if "AI 자동 개선 루프 시작" not in text:
                        return "auto"
                if text.startswith(f"{DAILY_TAG}[FIXER] 🔍 "):
                    if "YAML 검수 시작" not in text:
                        return "yaml_fixer"
                if text.startswith(f"{DAILY_TAG}[FIXER] "):
                    if "Codex 수정 시작" not in text and "YAML 검수 시작" not in text:
                        return "fixer"
                if text.startswith(f"{DAILY_TAG}[CHOICE] "):
                    if "📋 " not in text[:20]:
                        return "choice"
                return None

            def _direct_token_for_text(text: str) -> str:
                if (
                    text.startswith(f"{DAILY_TAG}[WAIT] ")
                    or text.startswith(f"{DAILY_TAG}[DONE] ")
                    or text.startswith(f"{DAILY_TAG}[ERROR] ")
                ):
                    return manager_bot_token
                return ""

            def _completion_keys_for_text(text: str) -> list[str]:
                keys: list[str] = []
                if (
                    text.startswith(f"{DAILY_TAG}[WAIT] ")
                    or text.startswith(f"{DAILY_TAG}[DONE] ")
                    or text.startswith(f"{DAILY_TAG}[ERROR] ")
                    or text.startswith(f"{DAILY_TAG}[REVIEW] ✅ 자동 검수 완료")
                ):
                    keys.append("start")
                if text.startswith(f"{DAILY_TAG}[DONE] "):
                    keys.append("choice")
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🔍 Config 검수 중"):
                    keys.append("start")
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🤖 GPT 컨텍스트 분석 중"):
                    keys.append("guardian_rules")
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] ✅ Config 검수 완료") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ⚠️ GPT 분석 실패"):
                    keys.append("guardian_gpt")
                if text.startswith(f"{DAILY_TAG}[MANAGER] 📋 매니저 지시사항:") or text.startswith(f"{DAILY_TAG}[MANAGER] ⚠️ 매니저 분석 실패"):
                    keys.append("manager")
                if (
                    text.startswith(f"{DAILY_TAG}[PROGRAMMER] ⏪ 로컬 검증 실패")
                    or text.startswith(f"{DAILY_TAG}[PROGRAMMER] ✅ 코드리뷰 통과")
                    or text.startswith(f"{DAILY_TAG}[PROGRAMMER] ⏪ 코드리뷰 reject")
                ):
                    keys.append("programmer")
                if text.startswith(f"{DAILY_TAG}[SIM] ✅ 시뮬레이션 완료"):
                    keys.append("sim")
                if text.startswith(f"{DAILY_TAG}[CHAPTER] ✅ 챕터 완성"):
                    keys.append("chapter")
                if text.startswith(f"{DAILY_TAG}[REVIEW] ✅ 자동 검수 완료"):
                    keys.append("review")
                if (
                    text.startswith(f"{DAILY_TAG}[AUTO] ✅ 품질 통과")
                    or text.startswith(f"{DAILY_TAG}[AUTO] ⚠️ 최대 사이클")
                    or text.startswith(f"{DAILY_TAG}[AUTO] ⚠️ 리뷰 실패")
                    or text.startswith(f"{DAILY_TAG}[AUTO] ❌ Codex Fixer 실패")
                ):
                    keys.append("auto")
                if (
                    text.startswith(f"{DAILY_TAG}[FIXER] ✅ Codex 수정 완료")
                    or text.startswith(f"{DAILY_TAG}[FIXER] ❌ Codex 수정 실패")
                ):
                    keys.append("fixer")
                if (
                    text.startswith(f"{DAILY_TAG}[FIXER] ✅ YAML 검수 완료")
                    or text.startswith(f"{DAILY_TAG}[FIXER] ⚠️ YAML 검수")
                ):
                    keys.append("yaml_fixer")
                return keys

            async def _ensure_anchor_thread(key: str, anchor_message: discord.Message) -> discord.abc.Messageable | None:
                anchor_threads[key] = None
                try:
                    thread = await anchor_message.create_thread(
                        name=f"daily-{key}-{anchor_message.id}",
                        auto_archive_duration=1440,
                    )
                    anchor_threads[key] = thread
                    return thread
                except Exception:
                    return None

            async def _notify(text: str) -> None:
                sent_anchor: discord.Message | None = None
                anchor_key = _anchor_key_for_text(text)
                if anchor_key is not None:
                    sent_anchor = await _send_text_with_token_return_message(
                        message.channel,
                        text,
                        _token_for_key(anchor_key),
                    )
                    anchor_messages[anchor_key] = sent_anchor
                    if sent_anchor is not None:
                        await _ensure_anchor_thread(anchor_key, sent_anchor)
                    return

                if sent_anchor is None:
                    route_key = _thread_route_for_text(text)
                    if route_key is not None:
                        thread_target = anchor_threads.get(route_key)
                        anchor_message = anchor_messages.get(route_key)
                        if thread_target is not None:
                            await _send_text_with_token(
                                thread_target,
                                int(getattr(thread_target, "id")),
                                text,
                                _token_for_key(route_key),
                            )
                        else:
                            await _send_text_with_token(
                                message.channel,
                                message.channel.id,
                                text,
                                _token_for_key(route_key),
                            )
                    else:
                        await _send_text_with_token(
                            message.channel,
                            message.channel.id,
                            text,
                            _direct_token_for_text(text),
                        )

                for key in _completion_keys_for_text(text):
                    anchor_message = anchor_messages.get(key)
                    if anchor_message is None:
                        continue
                    anchor_token = _token_for_key(key)
                    reacted = False
                    if anchor_token:
                        try:
                            await _rest_add_reaction(
                                int(getattr(anchor_message.channel, "id", message.channel.id)),
                                int(anchor_message.id),
                                "✅",
                                anchor_token,
                            )
                            reacted = True
                        except Exception:
                            pass
                    if not reacted:
                        await _add_reaction_safe(anchor_message, "✅")
                    thread = anchor_threads.get(key)
                    if thread is not None:
                        try:
                            await _send_text_with_token(
                                thread, int(getattr(thread, "id")), "✅", anchor_token
                            )
                        except Exception:
                            pass

            async def _upload(path: Path, note: str) -> None:
                try:
                    await _send_file(message.channel, path, note)
                except Exception:
                    await _send_text(message.channel, f"{note} (파일 업로드 실패: `{path.name}`)")

            await _send_text_with_token(
                message.channel,
                ch_id,
                f"▶️ `!novel-daily {episode_key}` 시작\n"
                f"리뷰 등급: `{review_tier}`\n"
                "진행 상황 확인: `!status` | 중단: `!stop`",
                manager_bot_token,
            )

            async def _run_daily_task() -> None:
                _ep_key = episode_key  # 재시작 루프에서 변경 가능
                _stop_ev = stop_ev
                _feedback_q = feedback_q
                MAX_AUTO_RESTARTS = 5
                auto_restarts = 0
                try:
                    while True:
                        result = await run_daily_pipeline(
                            episode_key=_ep_key,
                            target_words=tw,
                            budget=budget_val,
                            protagonist=protagonist,
                            feedback_queue=_feedback_q,
                            feedback_timeout_hours=24.0,
                            notify=_notify,
                            upload=_upload,
                            no_discord=False,
                            stop_event=_stop_ev,
                            review_tier=review_tier,
                            set_status=_set_status,
                            set_process=_set_process,
                            set_metrics=_set_metrics,
                            on_start_wait=_on_start_wait,
                            on_end_wait=_on_end_wait,
                            reset_emotions=(auto_restarts > 0),
                        )
                        if result and result.get("chapter_path"):
                            DAILY_CHAPTER_PATHS[ch_id] = Path(result["chapter_path"])

                        # 코드/스토리 수정 후 자동 재시작
                        choice = (result or {}).get("choice", "")
                        approved = (result or {}).get("approved", True)
                        user_stopped = _stop_ev.is_set()
                        if (
                            not user_stopped
                            and not approved
                            and choice in ("code", "story")
                            and auto_restarts < MAX_AUTO_RESTARTS
                        ):
                            auto_restarts += 1
                            # 새 stop event + feedback queue 준비
                            _stop_ev = asyncio.Event()
                            _feedback_q = asyncio.Queue()
                            DAILY_STOP_EVENTS[ch_id] = _stop_ev
                            DAILY_FEEDBACK_QUEUES[ch_id] = _feedback_q
                            DAILY_START_TIMES[ch_id] = time.monotonic()
                            await _notify(
                                f"🔄 **자동 재시작** ({auto_restarts}/{MAX_AUTO_RESTARTS}) "
                                f"— 수정된 코드로 `{_ep_key}` 파이프라인 재실행 중..."
                            )
                            continue

                        # 최대 재시작 도달 or 다른 종료 조건
                        if (
                            not user_stopped
                            and not approved
                            and choice in ("code", "story")
                            and auto_restarts >= MAX_AUTO_RESTARTS
                        ):
                            await _notify(
                                f"🛑 자동 재시작 {MAX_AUTO_RESTARTS}회 도달 — 파이프라인을 멈춥니다.\n"
                                f"수동으로 계속하려면: `!novel-daily {_ep_key}`"
                            )
                        break
                except Exception as exc:
                    DAILY_STATUS[ch_id] = f"실패 — {type(exc).__name__}"
                    await _send_text(
                        message.channel,
                        f"[ERROR] {type(exc).__name__}: {exc}",
                    )
                finally:
                    DAILY_PENDING_REVIEW_TIER.pop(ch_id, None)
                    DAILY_WAITING_FEEDBACK.discard(ch_id)
                    DAILY_PROCESS_INFO.pop(ch_id, None)
                    DAILY_START_TIMES.pop(ch_id, None)
                    if DAILY_FEEDBACK_QUEUES.get(ch_id) is _feedback_q:
                        DAILY_FEEDBACK_QUEUES.pop(ch_id, None)
                    if DAILY_STOP_EVENTS.get(ch_id) is _stop_ev:
                        DAILY_STOP_EVENTS.pop(ch_id, None)

            asyncio.create_task(_run_daily_task())
            return

        # ── !approve <req_id> ─────────────────────────────────────────────────
        if command == CMD_APPROVE:
            req_id = arg_text.split()[0] if arg_text else ""
            if not req_id:
                await message.channel.send("사용법: `!approve <req_id>`")
                return

            pending_path = REPO_ROOT / "data" / "pending_config_changes.json"
            if not pending_path.exists():
                await message.channel.send("⚠️ pending_config_changes.json 파일 없음")
                return

            with pending_path.open(encoding="utf-8") as f:
                pending = json.load(f)

            found = None
            for req in pending.get("requests", []):
                if req.get("id") == req_id:
                    found = req
                    break

            if found is None:
                await message.channel.send(f"⚠️ 요청 ID `{req_id}` 를 찾을 수 없음")
                return

            if found.get("status") != "pending":
                await message.channel.send(f"⚠️ `{req_id}` 상태: `{found.get('status')}` — 이미 처리됨")
                return

            # Validate not locked
            target_file = found.get("file", "")
            try:
                _assert_not_locked(target_file)
            except PermissionError as e:
                await message.channel.send(f"❌ 잠긴 파일입니다: {e}")
                return

            # Apply the proposed diff via patch
            proposed_diff = found.get("proposed_diff", "")
            apply_ok = True
            apply_note = ""
            if proposed_diff:
                diff_tmp = REPO_ROOT / "data" / f"_tmp_{req_id}.patch"
                diff_tmp.write_text(proposed_diff, encoding="utf-8")
                rc, out, err = _run_cmd(["git", "apply", "--check", str(diff_tmp)], timeout_sec=15)
                if rc == 0:
                    rc2, _, err2 = _run_cmd(["git", "apply", str(diff_tmp)], timeout_sec=15)
                    if rc2 != 0:
                        apply_ok = False
                        apply_note = f"git apply 실패: {err2[:300]}"
                else:
                    apply_ok = False
                    apply_note = f"패치 검증 실패: {err[:300]}"
                try:
                    diff_tmp.unlink()
                except OSError:
                    pass

            found["status"] = "approved" if apply_ok else "apply_failed"
            found["applied_at"] = datetime.utcnow().isoformat() + "Z"
            if not apply_ok:
                found["apply_error"] = apply_note

            with pending_path.open("w", encoding="utf-8") as f:
                json.dump(pending, f, ensure_ascii=False, indent=2)

            if apply_ok:
                await message.channel.send(
                    f"✅ `{req_id}` 승인 완료 — `{target_file}` 패치 적용됨\n"
                    f"설명: {found.get('description','')}"
                )
            else:
                await message.channel.send(
                    f"⚠️ `{req_id}` 승인됐으나 패치 적용 실패:\n```\n{apply_note}\n```\n"
                    "수동으로 변경 사항을 적용해 주세요."
                )
            return

        # ── !reject <req_id> [reason] ─────────────────────────────────────────
        if command == CMD_REJECT:
            parts_reject = arg_text.split(None, 1)
            if not parts_reject:
                await message.channel.send("사용법: `!reject <req_id> [이유]`")
                return
            req_id = parts_reject[0]
            reason = parts_reject[1] if len(parts_reject) > 1 else ""

            pending_path = REPO_ROOT / "data" / "pending_config_changes.json"
            if not pending_path.exists():
                await message.channel.send("⚠️ pending_config_changes.json 파일 없음")
                return

            with pending_path.open(encoding="utf-8") as f:
                pending = json.load(f)

            found = None
            for req in pending.get("requests", []):
                if req.get("id") == req_id:
                    found = req
                    break

            if found is None:
                await message.channel.send(f"⚠️ 요청 ID `{req_id}` 를 찾을 수 없음")
                return

            found["status"] = "rejected"
            found["rejected_at"] = datetime.utcnow().isoformat() + "Z"
            if reason:
                found["reject_reason"] = reason

            with pending_path.open("w", encoding="utf-8") as f:
                json.dump(pending, f, ensure_ascii=False, indent=2)

            await message.channel.send(
                f"❌ `{req_id}` 거절됨\n"
                + (f"이유: {reason}" if reason else "")
            )
            return

        # ── Daily pipeline feedback relay ─────────────────────────────────────
        # 스코어카드가 올라온 이후(DAILY_WAITING_FEEDBACK) 에만 피드백으로 처리.
        ch_id = message.channel.id
        if ch_id in DAILY_WAITING_FEEDBACK and not content.startswith("!"):
            q = DAILY_FEEDBACK_QUEUES.get(ch_id)
            if q:
                await q.put(content)
                DAILY_WAITING_FEEDBACK.discard(ch_id)  # 한 번만 받음
                await message.channel.send(
                    "📝 피드백 수신 완료.\n"
                    "이제 내용을 분석해서 상태 파일에 반영하고, 끝나면 다음에 무엇을 하면 되는지 안내하겠습니다."
                )
            return

    await client.start(token)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
