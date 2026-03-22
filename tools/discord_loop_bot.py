#!/usr/bin/env python3
"""
Discord bot for the Novel Writer project.

Supports: !novel-daily, !meitner, !status, !usage, !stop, !chapter, !approve, !reject, !emotion
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import signal
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
from tools.daily_pipeline import run_daily_pipeline, _generate_quality_chart
from tools.config_guardian import _assert_not_locked
from tools.inline_optimizer import SESSION_BENCHMARK_LOG
from tools.quality_reviewer import resolve_episode_file


ROOT_OUTPUT_DIR = REPO_ROOT / "output"
DISCORD_LOOP_STATE_PATH = REPO_ROOT / "data" / "discord_loop_state.json"
USAGE_STATE_KEY = "__usage_sessions__"

DAILY_TAG = ""

CMD_MEITNER = "!meitner"
CMD_DAILY = "!novel-daily"
CMD_APPROVE = "!approve"
CMD_REJECT = "!reject"
CMD_STATUS = "!status"
CMD_USAGE = "!usage"
CMD_PIPELINE_STOP = "!stop"
CMD_CHAPTER = "!chapter"
CMD_BENCHMARK = "!benchmark"
CMD_PARAMETER = "!parameter"
CMD_REBOOT = "!reboot"
CMD_SHUTDOWN = "!shutdown"
CMD_EMOTION = "!emotion"

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
DAILY_START_TIMES_WALL: dict[int, float] = {}  # channel_id → pipeline start time (wall clock)
DAILY_REVIEW_TIERS: dict[int, str] = {}        # channel_id → review_tier ("mini" / "premium")


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _session_cost_total(metrics: dict[str, Any]) -> float:
    return sum(
        float(metrics.get(key, 0.0))
        for key in (
            "guardian",
            "simulation",
            "chapter",
            "auto_chapter",
            "manager",
            "auto_review",
            "code_review",
            "regen_check",
            "final_review",
            "feedback_parse",
        )
    )


def _format_usage_summary(
    metrics: dict[str, Any] | None,
    start_time: float | None = None,
) -> str:
    metrics = metrics or {}
    token_total = int(metrics.get("total_tokens", 0))
    prompt_total = int(metrics.get("prompt_tokens", 0))
    completion_total = int(metrics.get("completion_tokens", 0))
    cost_total = _session_cost_total(metrics)
    parts = [
        f"🔢 세션 누적 토큰: {token_total:,} ({prompt_total:,} in + {completion_total:,} out)",
        f"💸 세션 누적 비용(Codex CLI 제외): ${cost_total:.4f}",
    ]
    if start_time is not None:
        elapsed = time.monotonic() - start_time
        elapsed_min, elapsed_sec = int(elapsed // 60), int(elapsed % 60)
        parts.insert(0, f"⏱️ 경과 시간: {elapsed_min}분 {elapsed_sec:02d}초")
    return "\n".join(parts)


def _load_discord_loop_state() -> dict[str, Any]:
    if not DISCORD_LOOP_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(DISCORD_LOOP_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_discord_loop_state(state: dict[str, Any]) -> None:
    DISCORD_LOOP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISCORD_LOOP_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _persist_usage_snapshot(
    channel_id: int,
    *,
    metrics: dict[str, Any] | None,
    status: str | None,
    episode_key: str | None,
    is_active: bool,
) -> None:
    metrics = dict(metrics or {})
    if not metrics and not status and not episode_key:
        return
    state = _load_discord_loop_state()
    usage_state = state.get(USAGE_STATE_KEY)
    if not isinstance(usage_state, dict):
        usage_state = {}
        state[USAGE_STATE_KEY] = usage_state

    started_wall = DAILY_START_TIMES_WALL.get(channel_id)
    elapsed_sec = None
    if is_active:
        started_mono = DAILY_START_TIMES.get(channel_id)
        if started_mono is not None:
            elapsed_sec = max(0, int(time.monotonic() - started_mono))
    else:
        prev = usage_state.get(str(channel_id), {})
        if isinstance(prev, dict):
            elapsed_sec = prev.get("elapsed_sec")

    usage_state[str(channel_id)] = {
        "status": status or "",
        "episode_key": episode_key or "",
        "is_active": bool(is_active),
        "metrics": metrics,
        "started_at": float(started_wall or time.time()),
        "elapsed_sec": int(elapsed_sec or 0),
        "updated_at": int(time.time()),
    }
    _save_discord_loop_state(state)


def _get_recent_usage_snapshot(channel_id: int) -> dict[str, Any] | None:
    state = _load_discord_loop_state()
    usage_state = state.get(USAGE_STATE_KEY)
    if not isinstance(usage_state, dict):
        return None
    snapshot = usage_state.get(str(channel_id))
    return snapshot if isinstance(snapshot, dict) else None


def _format_usage_summary_from_elapsed(
    metrics: dict[str, Any] | None,
    elapsed_sec: int | None,
) -> str:
    metrics = metrics or {}
    token_total = int(metrics.get("total_tokens", 0))
    prompt_total = int(metrics.get("prompt_tokens", 0))
    completion_total = int(metrics.get("completion_tokens", 0))
    cost_total = _session_cost_total(metrics)
    parts = []
    if elapsed_sec is not None:
        elapsed_min, elapsed_rem = int(elapsed_sec // 60), int(elapsed_sec % 60)
        parts.append(f"⏱️ 경과 시간: {elapsed_min}분 {elapsed_rem:02d}초")
    parts.extend(
        [
            f"🔢 세션 누적 토큰: {token_total:,} ({prompt_total:,} in + {completion_total:,} out)",
            f"💸 세션 누적 비용(Codex CLI 제외): ${cost_total:.4f}",
        ]
    )
    return "\n".join(parts)


def _parse_review_tier_choice(text: str) -> str | None:
    t = (text or "").strip().lower()
    if t in {"1", "mini", "min", "저렴", "빠르게", "가볍게"}:
        return "mini"
    if t in {"2", "premium", "prem", "프리미엄", "정밀", "고품질"}:
        return "premium"
    return None


def _parse_outer_cycles_choice(text: str) -> int | None:
    t = (text or "").strip().lower()
    aliases = {
        "한번": 1, "1회": 1,
        "두번": 2, "2회": 2,
        "세번": 3, "3회": 3,
        "네번": 4, "4회": 4,
        "다섯번": 5, "5회": 5,
        "열번": 10, "10회": 10,
    }
    if t in aliases:
        return aliases[t]
    if t.isdigit():
        value = int(t)
        if 1 <= value <= 50:
            return value
    return None


def _build_plan_preview(arg_text: str, outer_cycles: int) -> str:
    """outer_cycles 확정 후 사용자에게 보여줄 실행 플랜 미리보기."""
    from tools.daily_pipeline import (
        AUTO_OUTER_MAX_CYCLES, AUTO_INNER_MAX_CYCLES,
        AUTO_BATCH_TRIALS, AUTO_BATCH_GROUP_SIZE,
    )

    parts = arg_text.split()
    episode_key = parts[0] if parts else "?"
    review_tier = "mini"
    budget = 4.0
    target_words = 3500
    try:
        for i, p in enumerate(parts):
            if p == "--review-tier" and i + 1 < len(parts):
                review_tier = parts[i + 1]
            elif p == "--budget" and i + 1 < len(parts):
                budget = float(parts[i + 1])
            elif p == "--target-words" and i + 1 < len(parts):
                target_words = int(parts[i + 1])
    except (ValueError, IndexError):
        pass

    # Try to read episode config for simulation turns
    sim_turns_label = "?"
    try:
        from tools.quality_reviewer import resolve_episode_file
        from src.novel_writer.config_loader import load_episode
        ep_cfg = load_episode(str(resolve_episode_file(episode_key)))
        sim_turns_label = str(ep_cfg.get("max_turns") or ep_cfg.get("target_turns") or "?")
    except Exception:
        pass

    tier_label = "mini (빠름)" if review_tier == "mini" else "premium (정밀)"

    # 2-study MINI-OPT 계산
    n_sim_groups = AUTO_BATCH_TRIALS // AUTO_BATCH_GROUP_SIZE   # 5
    n_prose_per_sim = AUTO_BATCH_GROUP_SIZE                     # 5
    total_chapters_phase_a = AUTO_BATCH_TRIALS                  # 25
    total_chapters_phase_c = AUTO_INNER_MAX_CYCLES              # 3 regen
    max_chapters_per_outer = total_chapters_phase_a + total_chapters_phase_c
    max_chapters_total = AUTO_OUTER_MAX_CYCLES * max_chapters_per_outer  # per daily cycle

    # 파라미터 수: sim study 4개 + prose study 6개
    n_sim_params = 4
    n_prose_params = 6

    # 실측 비용 추정
    cost_line = ""
    try:
        from tools.cost_estimator import format_cost_estimate_for_plan
        cost_line = format_cost_estimate_for_plan(episode_key, outer_cycles)
    except Exception:
        cost_line = f"${budget:.1f}/cycle 상한 (실측 데이터 없음 — 첫 실행 후 자동 갱신)"

    # outer_cycles 기반 전체 수치 계산
    total_sims        = outer_cycles                                      # 시뮬레이션 횟수
    total_phase_a     = outer_cycles * total_chapters_phase_a             # Phase A 챕터 총합
    total_phase_c     = outer_cycles * total_chapters_phase_c             # Phase C 재생성 총합
    total_chapters    = outer_cycles * max_chapters_per_outer             # 전체 최대 챕터
    inner_max         = AUTO_INNER_MAX_CYCLES
    # AI 리뷰 횟수: Phase B 1회 + Phase C 재생성마다 1회 + 최종 1회
    total_reviews     = outer_cycles * (1 + inner_max) + 1

    sim_param_names   = "distiller_temp / target_scenes / dialogue_compaction / scene_closure"
    prose_param_names = "prose_scene_temp / para_min / para_max / transition_temp / polish_temp / hold_pressure"

    lines = [
        f"📋 **실행 플랜 확인** — `{episode_key}`",
        "─────────────────────────────────────",
        f"🔄  Outer cycles   : **{outer_cycles}회**",
        f"📊  리뷰 티어      : **{tier_label}**",
        f"📝  목표 단어수    : **{target_words:,}자**",
        f"💰  예상 비용      : {cost_line}",
        "─────────────────────────────────────",
        "**전체 규모 요약:**",
        f"  🎬  시뮬레이션      : **{total_sims}회** (outer cycle당 1회, {sim_turns_label}턴)",
        f"  📄  Phase A 챕터    : **{total_phase_a}개** ({outer_cycles}회 × {total_chapters_phase_a}개)",
        f"  🔧  Phase C 재생성  : 최대 **{total_phase_c}개** ({outer_cycles}회 × {inner_max}회)",
        f"  📦  전체 최대 챕터  : **{total_chapters}개** ({outer_cycles}×({total_chapters_phase_a}+{inner_max}))",
        f"  🔍  AI 리뷰         : 최대 **{total_reviews}회** (Phase B {outer_cycles}회 + Phase C 최대 {outer_cycles * inner_max}회 + 최종 1회)",
        "─────────────────────────────────────",
        "**각 Outer Cycle 내부 흐름:**",
        f"",
        f"  🔍 1. Config Guardian 검수",
        f"  ⚙️  2. 시뮬레이션 — {sim_turns_label}턴 (에이전트 멀티턴 → SceneDistiller 압축)",
        f"",
        f"  🧠 3. AI 자동 개선 루프",
        f"     Phase A — 2-study 파라미터 탐색 ({total_chapters_phase_a}개 챕터)",
        f"       • sim study ({n_sim_params}개)  : {sim_param_names}",
        f"         → {n_sim_groups}회 증류 × prose {n_prose_per_sim}병렬 = {total_chapters_phase_a}개 챕터 생성",
        f"       • prose study ({n_prose_params}개): {prose_param_names}",
        f"       • 샘플러: CMA-ES (데이터 5개↑) / TPE (초기)",
        f"       • 탐색 폭: ±25% → 누적 trial 증가할수록 자동 축소",
        f"     Phase B — AI 리뷰 **1회** + Factor Analysis",
        f"       → 점수 ≥ 9.5/10 달성 시 즉시 조기 종료",
        f"     Phase C — GPT Fixer 최대 **{inner_max}회** (재생성마다 AI 리뷰 1회)",
        f"       → 코드 수정 → 챕터 재생성 → AI 리뷰 → 점수 비교 → 하락 시 자동 롤백",
        f"",
        f"  📖 4. 최종 AI 리뷰 **1회** + 품질 스코어카드 출력",
        f"       평가 항목: 긴장감 / 문체 / 인과성 / 캐릭터 / 장면기능 (각 10점)",
        "─────────────────────────────────────",
        "**`1`** 로 시작  |  **`2`** 로 취소",
    ]
    return "\n".join(lines)


def _resolve_latest_daily_run_dir(ep_filter: str | None = None) -> Path | None:
    daily_dir = ROOT_OUTPUT_DIR / "daily"
    if not daily_dir.exists():
        return None

    normalized_filter: str | None = None
    if ep_filter:
        try:
            normalized_filter = resolve_episode_file(ep_filter).stem
        except Exception:
            normalized_filter = str(ep_filter).strip()

    run_dirs: list[Path] = []
    for ep_dir in daily_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        if normalized_filter and not ep_dir.name.endswith(f"_{normalized_filter}"):
            continue
        for run_dir in ep_dir.iterdir():
            if run_dir.is_dir():
                run_dirs.append(run_dir)
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _load_session_benchmark_rows(run_dir: Path) -> list[dict]:
    log_path = run_dir / SESSION_BENCHMARK_LOG
    rows: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        episode_id = None
        chapter_candidates = sorted(run_dir.glob("*_chapter.txt"))
        if chapter_candidates:
            episode_id = chapter_candidates[0].name.replace("_chapter.txt", "")
        cycle_count = len(list(run_dir.glob("auto_review_cycle*.json")))
        cycle_log = REPO_ROOT / "data" / "cycle_score_log.jsonl"
        if episode_id and cycle_count > 0 and cycle_log.exists():
            matching: list[dict] = []
            for line in cycle_log.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if record.get("episode_id") == episode_id:
                    matching.append(record)
            for record in matching[-cycle_count:]:
                cycle_idx = int(record.get("cycle_idx", 0))
                for subtrial in record.get("subtrials", []):
                    rows.append({
                        "ts": 0,
                        "episode_id": episode_id,
                        "cycle_idx": cycle_idx,
                        "trial_idx": int(subtrial.get("trial_idx", 0)),
                        "global_trial_idx": int(subtrial.get("trial_idx", 0)),
                        "score": float(subtrial.get("score", 0.0)),
                        "det": float(subtrial.get("det", 0.0)),
                        "llm": float(subtrial.get("llm", 0.0)),
                        "repetition_penalty": float(subtrial.get("repetition_penalty", 0.0)),
                        "params": dict(subtrial.get("params", {})),
                    })
    rows.sort(key=lambda row: (int(row.get("cycle_idx", 0)), int(row.get("trial_idx", 0))))
    return rows


def _build_session_benchmark_chart(rows: list[dict], ep_key: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        return None

    if not rows:
        return None

    _configure_korean_matplotlib_font(plt, fm)
    xs = [i + 1 for i in range(len(rows))]
    ys = [float(r.get("score", 0.0)) for r in rows]
    colors = ["#2ecc71" if y >= 8.5 else "#f39c12" if y >= 7.5 else "#e74c3c" for y in ys]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Benchmark — {ep_key} ({len(rows)} subtrials)", fontsize=13, fontweight="bold")
    ax.bar(xs, ys, color=colors, alpha=0.85, width=0.85)
    ax.plot(xs, ys, color="black", linewidth=1.2, alpha=0.7)
    for cycle_idx in sorted({int(r.get("cycle_idx", 0)) for r in rows}):
        cycle_rows = [i + 1 for i, r in enumerate(rows) if int(r.get("cycle_idx", 0)) == cycle_idx]
        if cycle_rows:
            ax.text(cycle_rows[0], 10.15, f"C{cycle_idx}", fontsize=8, fontweight="bold")
    ax.axhline(y=8.5, linestyle="--", linewidth=1, color="red", alpha=0.6)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("Subtrial Index")
    ax.set_ylabel("Score / 10")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    out_path = ROOT_OUTPUT_DIR / "benchmark_chart.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


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


def _resolve_alert_channel_id() -> int | None:
    raw = _env_value(
        "DISCORD_ALERT_CHANNEL_ID",
        "DISCORD_STATUS_CHANNEL_ID",
        "DISCORD_NOTIFY_CHANNEL_ID",
    )
    if not raw:
        return None
    try:
        val = int(raw)
    except ValueError:
        print(f"warning: invalid alert channel id `{raw}` (must be integer)")
        return None
    return val if val > 0 else None


def _find_latest(base_dir: Path, path_pattern: str) -> Path | None:
    files = sorted(base_dir.glob(path_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _configure_korean_matplotlib_font(plt: Any, fm: Any) -> None:
    """Pick an installed Hangul-capable font for matplotlib charts."""
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return


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
    _configure_korean_matplotlib_font(plt, fm)

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


def _make_emotion_chart(episode_key: str | None = None) -> Path | None:
    """!emotion — 씬별 긴장도 라인 + 캐릭터 등장 표시 그래프. PNG 임시 파일 경로 반환."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import tempfile
        import textwrap
    except ImportError:
        return None

    # ── 씬 파일 탐색 (최신 1개) ──────────────────────────────────────────────
    daily_dir = ROOT_OUTPUT_DIR / "daily"
    pattern = f"*{episode_key}*/**/*scenes*.json" if episode_key else "**/*scenes*.json"
    scene_files = sorted(daily_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not scene_files:
        scene_files = sorted(ROOT_OUTPUT_DIR.glob("**/*scenes*.json"), key=lambda p: p.stat().st_mtime)
    if not scene_files:
        return None

    try:
        scenes_raw = json.loads(scene_files[-1].read_text(encoding="utf-8"))
        if not isinstance(scenes_raw, list) or not scenes_raw:
            return None
    except Exception:
        return None

    # ── 긴장도 매핑 ──────────────────────────────────────────────────────────
    PACING_TENSION = {"opening": 3, "building": 6, "climax": 9, "resolution": 4}
    PACING_KO      = {"opening": "도입", "building": "고조", "climax": "절정", "resolution": "해소"}

    # emotion_trajectory 키워드 → 긴장도 보정값
    EMOTION_OFFSET = {
        "분노": +1, "공포": +1, "절망": +1, "충격": +1, "압박": +1,
        "긴장": +0.5, "불안": +0.5, "당혹": +0.5,
        "희망": -0.5, "안도": -1, "체념": -0.5, "신뢰": -0.5,
    }

    # ── 데이터 추출 ──────────────────────────────────────────────────────────
    xs       = []   # scene index
    ys       = []   # 긴장도
    labels   = []   # x축 라벨
    arcs     = []   # emotional_arc text
    pacings  = []   # pacing string
    all_chars: list[list[str]] = []

    for i, s in enumerate(scenes_raw):
        pacing = str(s.get("pacing", "building")).lower()
        base   = PACING_TENSION.get(pacing, 5)

        # emotion_trajectory로 보정
        traj = s.get("emotion_trajectory") or []
        offset = sum(EMOTION_OFFSET.get(e, 0) for e in traj)
        tension = min(10, max(1, base + offset))

        xs.append(i)
        ys.append(tension)
        labels.append(f"씬 {s.get('scene_number', i+1)}\n[{PACING_KO.get(pacing, pacing)}]")
        arcs.append(str(s.get("emotional_arc", "")).strip())
        pacings.append(pacing)
        all_chars.append([c.split()[-1] for c in s.get("characters_present", [])])  # 성만

    # ── 등장 캐릭터 목록 (전체) ──────────────────────────────────────────────
    unique_chars: list[str] = []
    for ch_list in all_chars:
        for c in ch_list:
            if c not in unique_chars:
                unique_chars.append(c)

    CHAR_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    _configure_korean_matplotlib_font(plt, fm)

    # ── 레이아웃: 메인 그래프 + 캐릭터 등장 서브플롯 ─────────────────────────
    fig, (ax_main, ax_char) = plt.subplots(
        2, 1, figsize=(max(10, len(xs) * 1.6), 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.subplots_adjust(hspace=0.05)

    ep_label  = episode_key or scene_files[-1].parent.name
    run_label = str(scene_files[-1].parent.relative_to(ROOT_OUTPUT_DIR))
    fig.suptitle(f"씬별 긴장도 & 감정 궤적 — {ep_label}\n({run_label})",
                 fontsize=12, fontweight="bold")

    # ── 메인: 긴장도 라인 ────────────────────────────────────────────────────
    PACING_COLOR = {"opening": "#5dade2", "building": "#f39c12",
                    "climax": "#e74c3c",  "resolution": "#27ae60"}

    # 구간별 색상으로 선 그리기
    for i in range(len(xs) - 1):
        col = PACING_COLOR.get(pacings[i], "#95a5a6")
        ax_main.plot(xs[i:i+2], ys[i:i+2], color=col, linewidth=2.5, solid_capstyle="round")

    # 점 + emotional_arc 주석
    for i, (x, y, arc, pacing) in enumerate(zip(xs, ys, arcs, pacings)):
        col = PACING_COLOR.get(pacing, "#95a5a6")
        ax_main.scatter(x, y, color=col, s=80, zorder=5)
        if arc:
            short = arc[:40] + "…" if len(arc) > 40 else arc
            offset_y = 0.4 if i % 2 == 0 else -0.6
            ax_main.annotate(
                short, (x, y),
                xytext=(0, 28 if offset_y > 0 else -28),
                textcoords="offset points",
                ha="center", va="bottom" if offset_y > 0 else "top",
                fontsize=7, color="#444444",
                arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.8),
            )

        # tension_peaks ⚡
        peaks = s.get("tension_peaks") or [] if (s := scenes_raw[i]) else []
        if peaks:
            ax_main.annotate("⚡", (x, y + 0.15), ha="center", fontsize=10, color="#c0392b")

    ax_main.set_xlim(-0.5, len(xs) - 0.5)
    ax_main.set_ylim(0, 11)
    ax_main.set_xticks(xs)
    ax_main.set_xticklabels(labels, fontsize=8)
    ax_main.set_ylabel("긴장도", fontsize=9)
    ax_main.set_yticks([2, 4, 6, 8, 10])
    ax_main.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_main.set_axisbelow(True)
    ax_main.tick_params(axis="x", bottom=False)

    # 범례
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=PACING_COLOR[p], linewidth=2.5, label=PACING_KO[p])
        for p in PACING_COLOR
    ]
    ax_main.legend(handles=legend_els, loc="upper left", fontsize=8, framealpha=0.8)

    # ── 서브: 캐릭터 등장 도트 ──────────────────────────────────────────────
    ax_char.set_xlim(-0.5, len(xs) - 0.5)
    ax_char.set_ylim(-0.5, len(unique_chars) - 0.5)
    ax_char.set_yticks(range(len(unique_chars)))
    ax_char.set_yticklabels(unique_chars, fontsize=8)
    ax_char.set_xticks(xs)
    ax_char.set_xticklabels([""] * len(xs))
    ax_char.yaxis.grid(True, linestyle=":", alpha=0.3)
    ax_char.set_xlabel("씬", fontsize=9)
    ax_char.tick_params(axis="x", bottom=False)

    for scene_i, ch_list in enumerate(all_chars):
        for char in ch_list:
            if char in unique_chars:
                ci = unique_chars.index(char)
                ax_char.scatter(scene_i, ci,
                                color=CHAR_COLORS[ci % len(CHAR_COLORS)],
                                s=90, marker="o", zorder=4)

    tmp = Path(tempfile.mktemp(suffix=".png", prefix="emotion_chart_"))
    fig.savefig(str(tmp), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return tmp


def _build_emotion_text_summary(episode_key: str | None = None) -> str | None:
    """!emotion — 씬별 감정 데이터를 텍스트로 요약. 이미지 전송 후 보조 메시지로 사용."""
    daily_dir = ROOT_OUTPUT_DIR / "daily"
    pattern = f"*{episode_key}*/**/*scenes*.json" if episode_key else "**/*scenes*.json"
    scene_files = sorted(daily_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not scene_files:
        scene_files = sorted(ROOT_OUTPUT_DIR.glob("**/*scenes*.json"), key=lambda p: p.stat().st_mtime)
    if not scene_files:
        return None

    try:
        scenes_raw = json.loads(scene_files[-1].read_text(encoding="utf-8"))
        if not isinstance(scenes_raw, list) or not scenes_raw:
            return None
    except Exception:
        return None

    PACING_KO = {"opening": "도입", "building": "고조", "climax": "절정", "resolution": "해소"}

    lines: list[str] = []
    ep_label = episode_key or scene_files[-1].parent.name
    lines.append(f"**📝 [{ep_label}] 씬별 감정 요약**\n")

    for s in scenes_raw:
        scene_num = s.get("scene_number", "?")
        pacing = str(s.get("pacing", "building")).lower()
        pacing_ko = PACING_KO.get(pacing, pacing)
        arc = str(s.get("emotional_arc", "")).strip()
        traj = s.get("emotion_trajectory") or []
        peaks = s.get("tension_peaks") or []
        delta = str(s.get("relationship_delta", "")).strip()

        block: list[str] = [f"**씬 {scene_num}** [{pacing_ko}]"]
        if arc:
            block.append(f"  감정 흐름: {arc}")
        if traj:
            block.append(f"  궤적: {' → '.join(traj)}")
        if peaks:
            block.append(f"  긴장 포인트: {' / '.join(peaks)}")
        if delta:
            block.append(f"  관계 변화: {delta}")
        lines.append("\n".join(block))

    return "\n\n".join(lines)


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
        "[GUARDIAN] 🧭 GPT 생성용 브리핑:",
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
            for attempt in range(3):
                async with session.post(
                    f"https://discord.com/api/v10/channels/{channel_id}/messages",
                    headers=headers,
                    json={"content": chunk},
                ) as resp:
                    if resp.status == 429:
                        try:
                            data = await resp.json()
                            retry_after = float(data.get("retry_after", 1.0))
                        except Exception:
                            retry_after = 1.0
                        await asyncio.sleep(retry_after)
                        continue
                    if resp.status >= 300:
                        body = await resp.text()
                        raise RuntimeError(f"REST text send failed: {resp.status} {body[:240]}")
                    break


async def _rest_send_text_return_message_id(channel_id: int, text: str, bot_token: str) -> int | None:
    content = str(text).strip()
    if not content:
        return None
    headers = {"Authorization": f"Bot {bot_token}"}
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        for attempt in range(3):
            async with session.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                headers=headers,
                json={"content": content[:1900]},
            ) as resp:
                if resp.status == 429:
                    try:
                        data = await resp.json()
                        retry_after = float(data.get("retry_after", 1.0))
                    except Exception:
                        retry_after = 1.0
                    await asyncio.sleep(retry_after)
                    continue
                body = await resp.text()
                if resp.status >= 300:
                    raise RuntimeError(f"REST text send failed: {resp.status} {body[:240]}")
                try:
                    data = json.loads(body)
                    mid = data.get("id")
                    return int(mid) if mid else None
                except Exception:
                    return None
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


async def _notify_system_channel(
    client: discord.Client,
    channel_id: int | None,
    text: str,
    bot_token: str,
) -> None:
    if not channel_id:
        return
    try:
        await _rest_send_text(channel_id, text, bot_token)
        return
    except Exception:
        pass

    channel = client.get_channel(channel_id)
    if channel is None:
        try:
            channel = await client.fetch_channel(channel_id)
        except Exception:
            return
    try:
        await _send_text(channel, text)
    except Exception:
        return


async def _send_team_reconnect_celebration(
    channel_id: int | None,
    main_bot_token: str,
    reviewer_bot_token: str,
    fixer_bot_token: str,
    manager_bot_token: str,
) -> None:
    """On connect, make all four bots post unique comeback lines and react to each other."""
    if not channel_id:
        return

    squad = [
        (
            "simulator",
            main_bot_token,
            random.choice(
                [
                    "🎬 우리가 돌아왔다. 오늘도 서사는 불타오른다.",
                    "🎥 시뮬레이터 복귀. 다음 장면, 더 세게 간다.",
                    "🚀 다시 연결 완료. 오늘의 전개는 한층 더 과감하다.",
                ]
            ),
        ),
        (
            "reviewer",
            reviewer_bot_token,
            random.choice(
                [
                    "🧪 리뷰어 복귀 완료. 문장 결, 리듬, 완성도까지 꼼꼼히 본다.",
                    "📏 리뷰어 접속. 날카로운 기준으로 품질을 끌어올린다.",
                    "🔍 리뷰어 등장. 디테일까지 놓치지 않고 확인한다.",
                ]
            ),
        ),
        (
            "programmer",
            fixer_bot_token,
            random.choice(
                [
                    "🛠️ 프로그래머 입장. 막힌 구간은 코드로 뚫고, 속도는 끝까지 끌어올린다.",
                    "⚙️ 프로그래머 컴백. 병목은 정리하고 루프는 더 빠르게 돈다.",
                    "🧩 프로그래머 연결됨. 문제는 분해해서 바로 해결한다.",
                ]
            ),
        ),
        (
            "manager",
            manager_bot_token,
            random.choice(
                [
                    "🧠 매니저 연결됨. 팀 시동 완료, 이제 결과로 말한다.",
                    "📣 매니저 복귀. 우선순위 정리 끝, 바로 진행한다.",
                    "🎯 매니저 온라인. 목표 고정, 실행은 단단하게 간다.",
                ]
            ),
        ),
    ]

    sent_messages: list[tuple[str, int, str]] = []
    manager_message_id: int | None = None
    for role, bot_token, text in squad:
        if not bot_token:
            continue
        try:
            msg_id = await _rest_send_text_return_message_id(channel_id, text, bot_token)
            if msg_id:
                sent_messages.append((role, msg_id, bot_token))
                if role == "manager":
                    manager_message_id = msg_id
        except Exception:
            continue
        await asyncio.sleep(0.15)

    if not sent_messages:
        return

    # 매니저 온라인 직후 명령어 안내 전송
    _cmd_guide = (
        "```\n"
        "📋 명령어 목록\n"
        "─────────────────────────────────────\n"
        "!novel-daily <ep>   소설 생성 파이프라인 시작\n"
        "  옵션: --target-words 3500\n"
        "        --budget 4.0\n"
        "        --protagonist kim_sumin\n"
        "        --review-tier mini|premium\n"
        "        --outer-cycles 1~50\n"
        "  플랜 승인 팝업 → 1 (시작) / 2 (취소)\n"
        "─────────────────────────────────────\n"
        "!status             현재 파이프라인 상태·진행 단계\n"
        "!usage              세션 토큰·비용 사용량\n"
        "!stop               파이프라인 중단 (차트 자동 생성)\n"
        "!chapter            최근 생성된 챕터 파일 다운로드\n"
        "!benchmark [ep]     에피소드별 점수 추이\n"
        "!parameter          Optuna 파라미터 현황 및 변화 이력\n"
        "!emotion [ep]       최근 씬 감정 궤적 차트\n"
        "─────────────────────────────────────\n"
        "!approve <req_id>   Guardian config 변경 승인\n"
        "!reject <req_id>    Guardian config 변경 거절\n"
        "─────────────────────────────────────\n"
        "!reboot             봇 재부팅\n"
        "!shutdown           봇 완전 종료\n"
        "!meitner <질문>      저장소 구조·코드 질문\n"
        "```"
    )
    try:
        await _rest_send_text_return_message_id(channel_id, _cmd_guide, manager_bot_token)
    except Exception:
        pass

    emoji_waves = {
        "simulator": ["🌈", "⚡", "🚀", "🫶", "🎉", "✨"],
        "reviewer": ["🟣", "🔵", "🟢", "✨", "📘", "🧠"],
        "programmer": ["🧩", "🔥", "🌟", "💫", "⚙️", "🛠️"],
        "manager": ["🎯", "🧠", "🏁", "🎉", "📣", "✅"],
    }

    # Each bot adds a different colorful reaction to every posted comeback message.
    for _, msg_id, _ in sent_messages:
        for idx, (role, _, reactor_token) in enumerate(squad):
            if not reactor_token:
                continue
            palette = list(emoji_waves.get(role, ["✨"]))
            random.shuffle(palette)
            emoji = palette[idx % len(palette)]
            try:
                await _rest_add_reaction(channel_id, msg_id, emoji, reactor_token)
            except Exception:
                pass
            await asyncio.sleep(0.08)

    # User-requested behavior: when manager announces reconnect,
    # the other three bots should leave a 👍 under manager's message.
    if manager_message_id:
        for role, reactor_token, _ in squad:
            if role == "manager" or not reactor_token:
                continue
            try:
                await _rest_add_reaction(channel_id, manager_message_id, "👍", reactor_token)
            except Exception:
                pass
            await asyncio.sleep(0.06)


async def _send_team_disconnect_farewell(
    channel_id: int | None,
    main_bot_token: str,
    reviewer_bot_token: str,
    fixer_bot_token: str,
    manager_bot_token: str,
) -> None:
    """When disconnecting, post manager-only farewell in the channel."""
    if not channel_id:
        return

    if not manager_bot_token:
        return

    text = random.choice(
        [
            "🧠 매니저 오프라인 전환. 준비 끝나면 다시 집결하자.",
            "📣 매니저 퇴장. 다음 접속 때 바로 재정렬해서 시작한다.",
            "🏁 매니저 로그아웃. 다음 라운드 목표는 이미 정해뒀다.",
        ]
    )
    try:
        await _rest_send_text_return_message_id(channel_id, text, manager_bot_token)
    except Exception:
        return


# ── Benchmark helpers ──────────────────────────────────────────────────────

def _collect_benchmark_rows(ep_filter: str | None = None) -> list[dict]:
    """Scan output/daily for all completed runs that have at least one review JSON."""
    daily_dir = ROOT_OUTPUT_DIR / "daily"
    rows: list[dict] = []
    if not daily_dir.exists():
        return rows

    for ep_dir in sorted(daily_dir.iterdir(), key=lambda p: p.name):
        if not ep_dir.is_dir():
            continue
        # ep_dir.name = "20260319_ep01_academic_presentation"
        name_parts = ep_dir.name.split("_", 1)
        if len(name_parts) < 2:
            continue
        date_str, ep_key = name_parts[0], name_parts[1]
        if ep_filter and ep_filter.lower() not in ep_key.lower():
            continue
        try:
            run_date = datetime.strptime(date_str, "%Y%m%d").strftime("%m-%d")
        except Exception:
            run_date = date_str

        for run_dir in sorted(ep_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            review_files = sorted(run_dir.glob("auto_review_cycle*.json"))
            if not review_files:
                continue
            # Take last cycle's scores
            try:
                review_data = json.loads(review_files[-1].read_text(encoding="utf-8"))
            except Exception:
                continue
            thrill    = int(review_data.get("thrill_score_10", 0) or 0)
            style     = int(review_data.get("style_score_10", 0) or 0)
            causality = int(review_data.get("causality_score_10", 0) or 0)
            character = int(review_data.get("character_score_10", 0) or 0)
            scene_fn  = int(review_data.get("scene_function_score_10", 0) or 0)
            avg = (thrill + style + causality + character + scene_fn) / 5.0

            sim_files = list(run_dir.glob("*_simulation.json"))
            total_turns: int | None = None
            if sim_files:
                try:
                    sd = json.loads(sim_files[0].read_text(encoding="utf-8"))
                    total_turns = sd.get("total_turns")
                except Exception:
                    pass

            try:
                ts = datetime.strptime(run_dir.name, "%H%M%S").strftime("%H:%M")
            except Exception:
                ts = run_dir.name

            rows.append({
                "ep_key":      ep_key,
                "date":        run_date,
                "time":        ts,
                "thrill":      thrill,
                "style":       style,
                "causality":   causality,
                "character":   character,
                "scene_fn":    scene_fn,
                "avg":         avg,
                "cycles":      len(review_files),
                "total_turns": total_turns,
            })

    return rows


def _build_benchmark_chart(rows: list[dict], ep_key: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        return None

    if not rows:
        return None

    _configure_korean_matplotlib_font(plt, fm)

    xs = list(range(1, len(rows) + 1))
    labels = [f"#{i}\n{r['date']}\n{r['time']}" for i, r in enumerate(rows, 1)]

    thrill_v    = [r["thrill"]    for r in rows]
    style_v     = [r["style"]     for r in rows]
    causality_v = [r["causality"] for r in rows]
    character_v = [r["character"] for r in rows]
    scene_fn_v  = [r["scene_fn"]  for r in rows]
    avg_v       = [r["avg"]       for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Benchmark — {ep_key}  (총 {len(rows)}회)", fontsize=13, fontweight="bold")

    # Left: score trend lines
    ax = axes[0]
    ax.set_title("점수 추이", fontsize=11)
    score_series = [
        (thrill_v,    "긴장감", "#e74c3c"),
        (style_v,     "문체",   "#3498db"),
        (causality_v, "인과성", "#2ecc71"),
        (character_v, "캐릭터", "#f39c12"),
        (scene_fn_v,  "씬기능", "#9b59b6"),
    ]
    for vals, label, color in score_series:
        ax.plot(xs, vals, marker="o", linewidth=1.4, markersize=5, label=label, color=color, alpha=0.8)
    ax.plot(xs, avg_v, marker="D", linewidth=2.5, markersize=6, label="평균", color="black")
    ax.axhline(y=8.5, linestyle="--", linewidth=1, color="red", alpha=0.5, label="목표(8.5)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 10.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_ylabel("점수 / 10")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    # Right: avg bar chart colored by value
    ax2 = axes[1]
    ax2.set_title("평균 점수 (실행별)", fontsize=11)
    bar_colors = ["#2ecc71" if v >= 8.5 else "#f39c12" if v >= 7.0 else "#e74c3c" for v in avg_v]
    bars = ax2.bar(xs, avg_v, color=bar_colors, alpha=0.85, width=0.6)
    ax2.axhline(y=8.5, linestyle="--", linewidth=1.2, color="red", alpha=0.6, label="목표(8.5)")
    for bar, val in zip(bars, avg_v):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylim(0, 10.5)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax2.set_ylabel("평균 점수 / 10")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = ROOT_OUTPUT_DIR / "benchmark_chart.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_parameter_report(
    session_cycle_rows: list[dict] | None,
) -> str:
    """Build !parameter report: param tables by group + per-cycle change history.

    session_cycle_rows: list of cycle_score_log records from current/latest session.
    Each record: {cycle_idx, cycle_params, ai_review: {avg, ...}, subtrials: [...]}
    Falls back to data/rl_policy.json for current values if no session data.
    """
    import json as _json
    from pathlib import Path as _Path

    _REPO = _Path(__file__).resolve().parent.parent
    _policy_path = _REPO / "data" / "rl_policy.json"
    try:
        current_policy = _json.loads(_policy_path.read_text(encoding="utf-8"))
    except Exception:
        current_policy = {}

    # ── Parameter groups ────────────────────────────────────────────────────
    GROUPS = {
        "Distiller": [
            "distiller_temperature", "distiller_max_tokens",
            "target_scenes", "scene_target_bias", "scene_target_min", "scene_target_max",
            "prose_history_max_episodes",
        ],
        "Prose": [
            "prose_transition_temperature",
            "prose_paragraph_min_sentences", "prose_paragraph_max_sentences",
            "prose_scene_readability_temperature",
        ],
        "Polish": [
            "prose_polish_temperature", "prose_anchor_fix_temperature",
        ],
        "Flags / Reader": [
            "prose_enable_term_gloss",
            "prefer_concrete_transition_cue", "prefer_scene_exit_on_stall",
            "director_fallback_cast_size",  # 고정값=4, 탐색 제외
            "reader_prefers_dialogue_compaction",
            "reader_prefers_analytical_wording_reduction",
            "repetition_jaccard_threshold",
        ],
        "Guardian (per-episode)": [
            "prose_scene_temperature",        # Guardian이 화별 문체 온도 결정
            "hold_pressure_peak",             # Guardian이 압박 유지 여부 결정
            "prefer_concrete_offer_detail",   # Guardian이 제안 구체화 결정
            "prefer_concrete_threat_detail",  # Guardian이 위협 구체화 결정
        ],
    }

    lines: list[str] = []

    # ── Per-cycle history table ─────────────────────────────────────────────
    if session_cycle_rows:
        lines.append("**📈 사이클별 파라미터 변화**")
        lines.append("```")
        # Collect all param keys that actually changed
        all_params_in_cycles = set()
        for r in session_cycle_rows:
            all_params_in_cycles.update(r.get("cycle_params", {}).keys())
        tracked = [p for grp in GROUPS.values() for p in grp if p in all_params_in_cycles]

        # Header row
        cycle_cols = [f"outer{r['cycle_idx']}" for r in session_cycle_rows]
        col_w = max(9, max(len(c) for c in cycle_cols))
        hdr = f"{'파라미터':<38} " + " ".join(f"{c:>{col_w}}" for c in cycle_cols) + f"  {'현재':>{col_w}}"
        lines.append(hdr)
        lines.append("─" * len(hdr))

        for param in tracked:
            vals = []
            for r in session_cycle_rows:
                v = r.get("cycle_params", {}).get(param)
                if v is None:
                    vals.append("-")
                elif isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            cur_v = current_policy.get(param)
            cur_str = f"{cur_v:.3f}" if isinstance(cur_v, float) else str(cur_v) if cur_v is not None else "-"
            row = f"{param:<38} " + " ".join(f"{v:>{col_w}}" for v in vals) + f"  {cur_str:>{col_w}}"
            lines.append(row)

        # Score row
        lines.append("─" * len(hdr))
        score_vals = [f"{r['ai_review'].get('avg', 0):.1f}" for r in session_cycle_rows]
        lines.append(
            f"{'AI avg':>38} " + " ".join(f"{v:>{col_w}}" for v in score_vals) + f"  {'(현재)':>{col_w}}"
        )
        lines.append("```")
        lines.append("")

    # ── Current value tables by group ───────────────────────────────────────
    lines.append("**🔧 현재 파라미터 (그룹별)**")
    for grp_name, params in GROUPS.items():
        lines.append(f"**{grp_name}**")
        lines.append("```")
        lines.append(f"  {'파라미터':<42} {'현재값':>10}")
        lines.append("  " + "─" * 54)
        for p in params:
            v = current_policy.get(p)
            if v is None:
                continue
            if isinstance(v, dict):
                v_str = ", ".join(f"{dk}={dv}" for dk, dv in v.items())
            elif isinstance(v, float):
                v_str = f"{v:.4f}"
            else:
                v_str = str(v)
            lines.append(f"  {p:<42} {v_str:>10}")
        lines.append("```")

    return "\n".join(lines)


async def async_main() -> None:
    os.chdir(REPO_ROOT)
    load_project_env(REPO_ROOT)
    _force_load_env_keys(
        [
            "OPENAI_API_KEY",
            "DISCORD_BOT_TOKEN",
            "DISCORD_BOT_TOKEN2",
            "DISCORD_BOT_TOKEN3",
            "DISCORD_BOT_TOKEN4",
            "DISCORD_ALERT_CHANNEL_ID",
            "DISCORD_STATUS_CHANNEL_ID",
            "DISCORD_NOTIFY_CHANNEL_ID",
        ]
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
    alert_channel_id = _resolve_alert_channel_id()
    last_disconnect_notice_ts = 0.0
    startup_team_hello_sent = False
    startup_celebrated_channels: set[int] = set()
    last_active_channel_id = alert_channel_id
    manual_shutdown_in_progress = False
    shutdown_farewell_sent = False

    intents = discord.Intents.default()
    intents.message_content = True
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    client = discord.Client(intents=intents, connector=connector)

    @client.event
    async def on_ready():
        nonlocal startup_team_hello_sent
        print(f"Discord bot connected as {client.user}")
        if startup_team_hello_sent:
            return
        startup_team_hello_sent = True
        # If a fixed alert channel is configured, celebrate there immediately.
        # Otherwise, we'll celebrate in the first active channel that receives a user message.
        if alert_channel_id:
            await _send_team_reconnect_celebration(
                alert_channel_id,
                token,
                reviewer_bot_token,
                fixer_bot_token,
                manager_bot_token,
            )
            startup_celebrated_channels.add(alert_channel_id)

    @client.event
    async def on_disconnect():
        nonlocal last_disconnect_notice_ts, last_active_channel_id, manual_shutdown_in_progress
        if manual_shutdown_in_progress:
            return
        now = time.time()
        if now - last_disconnect_notice_ts < 15:
            return
        last_disconnect_notice_ts = now
        notify_channel_id = alert_channel_id or last_active_channel_id
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[WARN] Discord gateway disconnected at {ts}")
        await _notify_system_channel(
            client,
            notify_channel_id,
            f"⚠️ Discord 연결 끊김 감지 ({ts})\n자동 재연결을 시도합니다.",
            manager_bot_token,
        )
        await _send_team_disconnect_farewell(
            notify_channel_id,
            token,
            reviewer_bot_token,
            fixer_bot_token,
            manager_bot_token,
        )

    @client.event
    async def on_resumed():
        notify_channel_id = alert_channel_id or last_active_channel_id
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] Discord gateway resumed at {ts}")
        await _notify_system_channel(
            client,
            notify_channel_id,
            f"✅ Discord 연결 복구 ({ts})",
            manager_bot_token,
        )

    @client.event
    async def on_message(message: discord.Message):
        nonlocal startup_celebrated_channels, last_active_channel_id
        if message.author.bot:
            return

        content = (message.content or "").strip()
        if not content:
            return
        last_active_channel_id = message.channel.id

        # No fixed alert channel? Then celebrate in the first channel where a user talks.
        if startup_team_hello_sent and message.channel.id not in startup_celebrated_channels:
            await _send_team_reconnect_celebration(
                message.channel.id,
                token,
                reviewer_bot_token,
                fixer_bot_token,
                manager_bot_token,
            )
            startup_celebrated_channels.add(message.channel.id)

        parts = content.split(None, 1)
        command = parts[0].lower()
        arg_text = parts[1].strip() if len(parts) > 1 else ""

        pending_review_tier = DAILY_PENDING_REVIEW_TIER.get(message.channel.id)
        # 만료된 plan_approval 상태 자동 정리
        if pending_review_tier:
            expires_at = pending_review_tier.get("expires_at")
            if expires_at and time.monotonic() > expires_at:
                DAILY_PENDING_REVIEW_TIER.pop(message.channel.id, None)
                pending_review_tier = None
        if pending_review_tier and message.author.id == pending_review_tier.get("user_id"):
            if content.lower() in {"취소", "cancel", "c"}:
                DAILY_PENDING_REVIEW_TIER.pop(message.channel.id, None)
                _cancel_stage = pending_review_tier.get("stage", "review_tier")
                _cancel_msg = (
                    "❌ 실행이 취소됐습니다. 다시 `!novel-daily <episode>` 로 설정할 수 있어요."
                    if _cancel_stage == "plan_approval"
                    else "설정이 취소됐습니다."
                )
                await _send_text_with_token(
                    message.channel,
                    message.channel.id,
                    _cancel_msg,
                    manager_bot_token,
                )
                return
            if not content.startswith("!"):
                stage = str(pending_review_tier.get("stage", "review_tier"))
                if stage == "review_tier":
                    chosen_tier = _parse_review_tier_choice(content)
                    if chosen_tier:
                        next_arg_text = f"{pending_review_tier.get('arg_text', '')} --review-tier {chosen_tier}".strip()
                        DAILY_PENDING_REVIEW_TIER[message.channel.id] = {
                            "user_id": message.author.id,
                            "stage": "outer_cycles",
                            "arg_text": next_arg_text,
                        }
                        await _send_text_with_token(
                            message.channel,
                            message.channel.id,
                            "이번 세션에서 outer cycle을 몇 번 돌릴까요? (`1`~`50`)\n"
                            "`3` 빠른 확인 | `10` 기본 추천 | `25` 충분한 탐색 | `50` 최대\n"
                            "숫자로 답장해주세요. 취소는 `취소`.",
                            manager_bot_token,
                        )
                        return
                    await _send_text_with_token(
                        message.channel,
                        message.channel.id,
                        "리뷰 등급을 `mini` 또는 `premium`으로 보내주세요. `1/2`도 됩니다. 취소는 `취소`.",
                        manager_bot_token,
                    )
                    return

                if stage == "outer_cycles":
                    chosen_outer_cycles = _parse_outer_cycles_choice(content)
                    if chosen_outer_cycles is None:
                        await _send_text_with_token(
                            message.channel,
                            message.channel.id,
                            "outer cycle 수를 `1`에서 `50` 사이 숫자로 보내주세요. 취소는 `취소`.",
                            manager_bot_token,
                        )
                        return

                    final_arg_text = (
                        f"{pending_review_tier.get('arg_text', '')} --outer-cycles {chosen_outer_cycles}"
                    ).strip()
                    DAILY_PENDING_REVIEW_TIER[message.channel.id] = {
                        "user_id": message.author.id,
                        "stage": "plan_approval",
                        "arg_text": final_arg_text,
                        "expires_at": time.monotonic() + 1800,  # 30분 TTL
                    }
                    plan_msg = _build_plan_preview(final_arg_text, chosen_outer_cycles)
                    await _send_text_with_token(
                        message.channel,
                        message.channel.id,
                        plan_msg,
                        manager_bot_token,
                    )
                    return

                # stage == "plan_approval"
                lowered = content.strip().lower()
                if lowered in {"1", "yes", "y", "승인", "시작", "ok", "ㅇㅋ"}:
                    DAILY_PENDING_REVIEW_TIER.pop(message.channel.id, None)
                    command = CMD_DAILY
                    arg_text = pending_review_tier.get("arg_text", "")
                elif lowered in {"2", "no", "n", "취소", "cancel", "거절"}:
                    DAILY_PENDING_REVIEW_TIER.pop(message.channel.id, None)
                    await _send_text_with_token(
                        message.channel,
                        message.channel.id,
                        "❌ 실행이 취소됐습니다. 다시 `!novel-daily <episode>` 로 설정할 수 있어요.",
                        manager_bot_token,
                    )
                    return
                else:
                    await _send_text_with_token(
                        message.channel,
                        message.channel.id,
                        "**`1`** 로 시작  |  **`2`** 로 취소",
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
                start_t = DAILY_START_TIMES.get(ch_id)
                usage_block = "\n" + _format_usage_summary(metrics, start_t)
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    f"📊 현재 파이프라인 상태: **{status}**{proc_line}{usage_block}",
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

        # ── !usage ────────────────────────────────────────────────────────────
        if command == CMD_USAGE and not arg_text:
            ch_id = message.channel.id
            active_ev = DAILY_STOP_EVENTS.get(ch_id)
            active_metrics = DAILY_SESSION_METRICS.get(ch_id)
            is_active = bool(active_ev and not active_ev.is_set() and active_metrics)
            if is_active:
                start_t = DAILY_START_TIMES.get(ch_id)
                status = DAILY_STATUS.get(ch_id)
                prefix = f"📈 현재 세션 사용량" + (f" ({status})" if status else "")
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    f"{prefix}\n{_format_usage_summary(active_metrics, start_t)}",
                    manager_bot_token,
                )
                # 중간 차트 생성 + 업로드
                _ep_key = DAILY_EPISODE_KEYS.get(ch_id)
                _run_dir = _resolve_latest_daily_run_dir(_ep_key) if _ep_key else None
                _tier = DAILY_REVIEW_TIERS.get(ch_id, "premium")
                if _run_dir:
                    try:
                        _chart = await asyncio.to_thread(
                            _generate_quality_chart, _ep_key, _run_dir, active_metrics, None, _tier
                        )
                        if _chart:
                            await _send_file_with_token(message.channel, ch_id, _chart, "📊 중간 품질 차트", manager_bot_token)
                    except Exception as _ce:
                        logger.warning("[USAGE] chart gen failed: %s", _ce)
                return

            metrics = DAILY_SESSION_METRICS.get(ch_id)
            status = DAILY_STATUS.get(ch_id)
            if metrics:
                recent_snapshot = _get_recent_usage_snapshot(ch_id)
                snapshot_text = _format_usage_summary_from_elapsed(
                    metrics,
                    recent_snapshot.get("elapsed_sec") if recent_snapshot else None,
                )
                prefix = f"📈 최근 세션 사용량" + (f" ({status})" if status else "")
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    f"{prefix}\n{snapshot_text}",
                    manager_bot_token,
                )
                return

            snapshot = _get_recent_usage_snapshot(ch_id)
            if snapshot:
                snapshot_metrics = snapshot.get("metrics") if isinstance(snapshot.get("metrics"), dict) else {}
                snapshot_status = str(snapshot.get("status", "") or "").strip()
                prefix = "📈 최근 세션 사용량"
                if snapshot_status:
                    prefix += f" ({snapshot_status})"
                episode_key = str(snapshot.get("episode_key", "") or "").strip()
                if episode_key:
                    prefix += f"\n에피소드: `{episode_key}`"
                body = _format_usage_summary_from_elapsed(snapshot_metrics, snapshot.get("elapsed_sec"))
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    f"{prefix}\n{body}",
                    manager_bot_token,
                )
            else:
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    "📈 아직 이 채널의 사용량 기록이 없습니다.",
                    manager_bot_token,
                )
            return

        # ── !reboot ───────────────────────────────────────────────────────────
        if command == CMD_SHUTDOWN and not arg_text:
            ch_id = message.channel.id
            ev = DAILY_STOP_EVENTS.get(ch_id)
            if ev and not ev.is_set():
                ev.set()
            await _send_text_with_token(
                message.channel,
                ch_id,
                "⏹️ 봇 종료합니다. 재시작하려면 서버에서 직접 실행해주세요.",
                manager_bot_token,
            )
            await asyncio.sleep(1)
            sys.exit(0)

        if command == CMD_REBOOT and not arg_text:
            ch_id = message.channel.id
            DAILY_PENDING_REVIEW_TIER.pop(ch_id, None)  # plan_approval 상태 정리
            # 실행 중인 파이프라인이 있으면 먼저 중단 요청
            ev = DAILY_STOP_EVENTS.get(ch_id)
            if ev and not ev.is_set():
                ev.set()
            await _send_text_with_token(
                message.channel,
                ch_id,
                "🔄 봇 재부팅 중... 잠시 후 다시 연결됩니다.",
                manager_bot_token,
            )
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # ── !stop ─────────────────────────────────────────────────────────────
        if command == CMD_PIPELINE_STOP and not arg_text:
            ch_id = message.channel.id
            DAILY_PENDING_REVIEW_TIER.pop(ch_id, None)  # plan_approval 상태 정리
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
                    await _send_file_with_token(
                        message.channel, ch_id,
                        chapter_path, f"📖 최근 챕터 — {word_count}단어 (`{chapter_path.name}`)",
                        manager_bot_token,
                    )
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

        # ── !emotion [episode_key] ────────────────────────────────────────────
        if command == CMD_EMOTION:
            ch_id = message.channel.id
            ep_filter = arg_text.strip() or DAILY_EPISODE_KEYS.get(ch_id) or None
            await _send_text_with_token(
                message.channel, ch_id,
                f"🎭 감정 궤적 차트 생성 중{f' — `{ep_filter}`' if ep_filter else ''}...",
                manager_bot_token,
            )
            try:
                chart_path = await asyncio.to_thread(_make_emotion_chart, ep_filter)
                if chart_path and chart_path.exists():
                    ep_label = ep_filter or "최근 씬"
                    await _send_file_with_token(
                        message.channel, ch_id,
                        chart_path,
                        f"🎭 [{ep_label}] 씬별 감정 궤적",
                        manager_bot_token,
                    )
                    chart_path.unlink(missing_ok=True)
                    # 이미지 아래에 텍스트 요약도 전송
                    text_summary = await asyncio.to_thread(_build_emotion_text_summary, ep_filter)
                    if text_summary:
                        await _send_text_with_token(
                            message.channel, ch_id, text_summary, manager_bot_token,
                        )
                else:
                    await _send_text_with_token(
                        message.channel, ch_id,
                        "⚠️ 씬 파일을 찾을 수 없습니다. `!novel-daily <episode>` 를 먼저 실행하세요.",
                        manager_bot_token,
                    )
            except Exception as _e:
                await _send_text_with_token(
                    message.channel, ch_id,
                    f"❌ 차트 생성 실패: {_e}",
                    manager_bot_token,
                )
            return

        # ── !benchmark ────────────────────────────────────────────────────────
        if command == CMD_BENCHMARK:
            ep_filter = arg_text.strip() or None
            await _send_text_with_token(
                message.channel, message.channel.id,
                "🔍 벤치마크 데이터 수집 중...",
                manager_bot_token,
            )
            active_snapshot = _get_recent_usage_snapshot(message.channel.id)
            active_episode = None
            if active_snapshot and active_snapshot.get("is_active"):
                active_episode = str(active_snapshot.get("episode_key", "") or "").strip()
            target_episode = active_episode or ep_filter
            run_dir = await asyncio.to_thread(_resolve_latest_daily_run_dir, target_episode)
            rows = await asyncio.to_thread(_load_session_benchmark_rows, run_dir) if run_dir else []
            if not rows:
                await _send_text_with_token(
                    message.channel, message.channel.id,
                    "⚠️ 현재 세션 또는 가장 최근 세션의 벤치마크 데이터가 없습니다.",
                    manager_bot_token,
                )
                return
            ep_key_name = str(rows[-1].get("episode_id", run_dir.parent.name.split("_", 1)[-1] if run_dir else "unknown"))
            scores = [float(r.get("score", 0.0)) for r in rows]
            repetition_penalties = [float(r.get("repetition_penalty", 0.0)) for r in rows]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            worst_idx = min(range(len(scores)), key=lambda i: scores[i])
            cycle_ids = sorted({int(r.get("cycle_idx", 0)) for r in rows})
            mode_label = "현재 세션" if active_episode else "가장 최근 세션"
            last_rows = rows[-10:]
            lines = [
                f"📊 **벤치마크 — {ep_key_name}** ({mode_label})",
                f"run: `{run_dir.relative_to(REPO_ROOT) if run_dir else '-'}`",
                f"누적 subtrials: `{len(rows)}` | outer cycles 기록: `{', '.join(str(c) for c in cycle_ids)}`",
                f"최고: `t{int(rows[best_idx].get('trial_idx', best_idx))}` {scores[best_idx]:.3f} | 최저: `t{int(rows[worst_idx].get('trial_idx', worst_idx))}` {scores[worst_idx]:.3f}",
                f"시작→최근: `{scores[0]:.3f} → {scores[-1]:.3f}`",
                f"평균 repetition_penalty: `-{sum(repetition_penalties) / max(1, len(repetition_penalties)):.3f}`",
                "```",
                "최근 10개 subtrials",
            ]
            for row in last_rows:
                lines.append(
                    f"C{int(row.get('cycle_idx', 0))} "
                    f"t{int(row.get('trial_idx', 0)):>2} "
                    f"score={float(row.get('score', 0.0)):.3f} "
                    f"(det {float(row.get('det', 0.0)):.3f} / llm {float(row.get('llm', 0.0)):.3f} / rep -{float(row.get('repetition_penalty', 0.0)):.3f})"
                )
            lines.append("```")
            await _send_text_with_token(
                message.channel,
                message.channel.id,
                "\n".join(lines),
                manager_bot_token,
            )

            chart_path = await asyncio.to_thread(_build_session_benchmark_chart, rows, ep_key_name)
            if chart_path and chart_path.exists():
                try:
                    await _send_file_with_token(
                        message.channel,
                        message.channel.id,
                        chart_path,
                        f"📈 세션 벤치마크 차트 — {ep_key_name}",
                        manager_bot_token,
                    )
                except Exception:
                    pass
                chart_path.unlink(missing_ok=True)
            return

        # ── !parameter ────────────────────────────────────────────────────────
        if command == CMD_PARAMETER:
            ch_id = message.channel.id
            active_metrics = DAILY_SESSION_METRICS.get(ch_id)
            active_ev = DAILY_STOP_EVENTS.get(ch_id)
            is_active = bool(active_ev and not active_ev.is_set() and active_metrics)
            await _send_text_with_token(
                message.channel, ch_id,
                "🔍 파라미터 분석 중...",
                manager_bot_token,
            )

            def _load_parameter_report() -> str:
                import json as _json
                from datetime import date as _date
                _log_path = REPO_ROOT / "data" / "cycle_score_log.jsonl"
                cycle_rows: list[dict] = []
                if _log_path.exists():
                    for line in _log_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            cycle_rows.append(_json.loads(line))
                        except Exception:
                            pass
                if cycle_rows:
                    if is_active:
                        # Current session: only today's entries
                        today = str(_date.today())
                        cycle_rows = [r for r in cycle_rows if r.get("date", "") == today]
                    else:
                        # Latest completed session: last date block
                        last_date = cycle_rows[-1].get("date", "")
                        cycle_rows = [r for r in cycle_rows if r.get("date", "") == last_date]
                return _build_parameter_report(cycle_rows or None)

            report = await asyncio.to_thread(_load_parameter_report)
            # Split if too long for Discord (2000 char limit)
            if len(report) <= 1900:
                await _send_text_with_token(
                    message.channel, ch_id, report, manager_bot_token,
                )
            else:
                # Send in two parts
                mid = report.find("\n**🔧")
                part1 = report[:mid] if mid > 0 else report[:1900]
                part2 = report[mid:] if mid > 0 else report[1900:]
                await _send_text_with_token(
                    message.channel, ch_id, part1, manager_bot_token,
                )
                if part2:
                    await _send_text_with_token(
                        message.channel, ch_id, part2, manager_bot_token,
                    )
            return

        # ── !novel-daily <episode_key> ─────────────────────────────────────────
        if command == CMD_DAILY:
            daily_args = arg_text.split()
            if not daily_args:
                await message.channel.send(
                    "사용법: `!novel-daily <번호 또는 episode_key>`\n"
                    "예: `!novel-daily 1` / `!novel-daily 15`\n"
                    "옵션: `--target-words 3500 --budget 4.0 --protagonist kim_sumin --review-tier mini|premium --outer-cycles 3`"
                )
                return
            episode_key = daily_args[0]
            tw = 3500
            budget_val = 4.0
            protagonist = "kim_sumin"
            review_tier: str | None = None
            outer_cycles: int | None = None
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
                elif remaining[i] == "--outer-cycles" and i + 1 < len(remaining):
                    outer_cycles = _parse_outer_cycles_choice(remaining[i + 1]); i += 2
                else:
                    i += 1

            ch_id = message.channel.id

            if review_tier is None:
                DAILY_PENDING_REVIEW_TIER[ch_id] = {
                    "user_id": message.author.id,
                    "stage": "review_tier",
                    "arg_text": arg_text,
                }
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    "리뷰 등급을 골라주세요.\n"
                    "`1` 또는 `mini` — 빠르고 저렴\n"
                    "`2` 또는 `premium` — 정밀, 고품질\n"
                    "답장으로 보내주시면 시작합니다. 취소는 `취소`.",
                    manager_bot_token,
                )
                return
            if outer_cycles is None:
                DAILY_PENDING_REVIEW_TIER[ch_id] = {
                    "user_id": message.author.id,
                    "stage": "outer_cycles",
                    "arg_text": arg_text,
                }
                await _send_text_with_token(
                    message.channel,
                    ch_id,
                    "이번 세션에서 outer cycle을 몇 번 돌릴까요?\n"
                    "`1` 빠르게 확인\n"
                    "`2` 중간\n"
                    "`3` 기본 추천\n"
                    "`4` 더 길게\n"
                    "`5` 최대 탐색\n"
                    "숫자로 답장해주세요. 취소는 `취소`.",
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
            DAILY_REVIEW_TIERS[ch_id] = review_tier or "premium"
            DAILY_START_TIMES[ch_id] = time.monotonic()
            DAILY_START_TIMES_WALL[ch_id] = time.time()
            DAILY_SESSION_METRICS[ch_id] = {
                "guardian": 0.0,
                "simulation": 0.0,
                "chapter": 0.0,
                "auto_chapter": 0.0,
                "manager": 0.0,
                "auto_review": 0.0,
                "code_review": 0.0,
                "regen_check": 0.0,
                "final_review": 0.0,
                "feedback_parse": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model_token_totals": {},
            }
            _persist_usage_snapshot(
                ch_id,
                metrics=DAILY_SESSION_METRICS[ch_id],
                status=DAILY_STATUS.get(ch_id),
                episode_key=episode_key,
                is_active=True,
            )

            def _set_status(s: str) -> None:
                DAILY_STATUS[ch_id] = s
                ev = DAILY_STOP_EVENTS.get(ch_id)
                _persist_usage_snapshot(
                    ch_id,
                    metrics=DAILY_SESSION_METRICS.get(ch_id),
                    status=s,
                    episode_key=DAILY_EPISODE_KEYS.get(ch_id),
                    is_active=bool(ev and not ev.is_set()),
                )

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
                ev = DAILY_STOP_EVENTS.get(ch_id)
                _persist_usage_snapshot(
                    ch_id,
                    metrics=DAILY_SESSION_METRICS.get(ch_id),
                    status=DAILY_STATUS.get(ch_id),
                    episode_key=DAILY_EPISODE_KEYS.get(ch_id),
                    is_active=bool(ev and not ev.is_set()),
                )

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
                "optimize": None,
                "chapter": None,
                "review": None,
                "auto": None,
                "auto_review": None,
                "fixer": None,
                "yaml_fixer": None,
                "choice": None,
                "mini_opt": None,
                "mini_opt_sim": None,
                "mini_opt_score": None,
                "mini_opt_prog": None,
            }
            anchor_threads: dict[str, discord.abc.Messageable | None] = {
                "start": None,
                "reset": None,
                "guardian_rules": None,
                "guardian_gpt": None,
                "manager": None,
                "programmer": None,
                "sim": None,
                "optimize": None,
                "chapter": None,
                "review": None,
                "auto": None,
                "auto_review": None,
                "fixer": None,
                "yaml_fixer": None,
                "choice": None,
                "mini_opt": None,
                "mini_opt_sim": None,
                "mini_opt_score": None,
                "mini_opt_prog": None,
            }

            def _token_for_key(key: str | None) -> str:
                if key in {"start", "manager", "choice", "mini_opt", "auto_review"}:
                    return manager_bot_token
                if key in {"guardian_rules", "guardian_gpt", "review", "mini_opt_score"}:
                    return reviewer_bot_token
                if key in {"auto", "fixer", "yaml_fixer", "programmer", "reset",
                           "mini_opt_prog"}:
                    return fixer_bot_token
                if key in {"mini_opt_sim"}:
                    return ""  # Simulator = main_bot_token (empty = main bot)
                return ""

            def _anchor_key_for_text(text: str) -> str | None:
                if text.startswith(f"{DAILY_TAG}[START] 🎬 "):
                    return "start"
                if text.startswith(f"{DAILY_TAG}[RESET] ♻️ "):
                    return "reset"
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
                if text.startswith(f"{DAILY_TAG}[OPTIMIZE] 🔬 인라인 최적화 시작"):
                    return "optimize"
                if text.startswith(f"{DAILY_TAG}[CHAPTER] 📖 챕터 생성 중"):
                    return "chapter"
                if text.startswith(f"{DAILY_TAG}[REVIEW] 🔍 품질 자동 검수 중"):
                    return "review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 🚀 AI 자동 개선 루프"):
                    return "auto"  # 사이클마다 새 쓰레드 생성
                if text.startswith(f"{DAILY_TAG}[AUTO] 📊 AI 리뷰 결과"):
                    return "auto_review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 📊 Phase B 리뷰 결과"):
                    return "auto_review"  # Manager anchor + thread
                if text.startswith(f"{DAILY_TAG}[MINI-OPT] outer") and "완료" not in text:
                    return "mini_opt"  # Manager anchor, new thread per outer cycle (group or 2-study)
                if text.startswith(f"{DAILY_TAG}[FIXER] 🔧 Codex 수정 시작"):
                    return "fixer"
                if text.startswith(f"{DAILY_TAG}[FIXER] 🔍 YAML 검수 시작"):
                    return "yaml_fixer"
                if text.startswith(f"{DAILY_TAG}[CHOICE] 📋 "):
                    return "choice"
                return None

            def _thread_route_for_text(text: str) -> str | None:
                if text.startswith(f"{DAILY_TAG}[START] run:"):
                    return "start"
                if text.startswith(f"{DAILY_TAG}[RESET] 🗂️ "):
                    return "reset"
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] Config 규칙 검수 결과:") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ⚠️ Config 변경 요청"):
                    return "guardian_rules"
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🧠 GPT 분석 리포트:") or text.startswith(f"{DAILY_TAG}[GUARDIAN] 🧭 GPT 생성용 브리핑:") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ✅ Config 검수 완료") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ⚠️ GPT 분석 실패"):
                    return "guardian_gpt"
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 💸"):
                    return "guardian_rules"
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
                if text.startswith(f"{DAILY_TAG}[OPTIMIZE] "):
                    if "인라인 최적화 시작" not in text:
                        return "optimize"
                if text.startswith(f"{DAILY_TAG}[REVIEW] "):
                    if "품질 자동 검수 중" not in text:
                        return "review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 🧾 AI 리뷰 상세"):
                    return "auto_review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 📋 Codex 수정 진단"):
                    return "auto_review"
                if text.startswith(f"{DAILY_TAG}[AUTO] 📊 Factor Analysis"):
                    return "auto_review"  # Factor analysis detail → Phase B thread
                if text.startswith(f"{DAILY_TAG}[AUTO] "):
                    if "AI 자동 개선 루프 시작" not in text:
                        return "auto"
                if text.startswith(f"{DAILY_TAG}[MINI-OPT-SIM] "):
                    return "mini_opt_sim"
                if text.startswith(f"{DAILY_TAG}[MINI-OPT-SCORE] "):
                    return "mini_opt_score"
                if text.startswith(f"{DAILY_TAG}[MINI-OPT-PROG] "):
                    return "mini_opt_prog"
                if text.startswith(f"{DAILY_TAG}[MINI-OPT] "):
                    return "mini_opt"  # group 완료 summary → Manager in thread
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
                if text.startswith(f"{DAILY_TAG}[AUTO] 🚀 "):
                    return fixer_bot_token
                return ""

            def _completion_keys_for_text(text: str) -> list[str]:
                keys: list[str] = []
                if text.startswith(f"{DAILY_TAG}[START] run:"):
                    keys.append("start")
                if text.startswith(f"{DAILY_TAG}[RESET] 🗂️ "):
                    keys.append("reset")
                if (
                    text.startswith(f"{DAILY_TAG}[WAIT] ")
                    or text.startswith(f"{DAILY_TAG}[DONE] ")
                    or text.startswith(f"{DAILY_TAG}[ERROR] ")
                    or text.startswith(f"{DAILY_TAG}[REVIEW] ✅ 자동 검수 완료")
                ):
                    keys.append("start")
                if text.startswith(f"{DAILY_TAG}[DONE] "):
                    keys.append("choice")
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🔍 Config 검수 중") or text.startswith(f"{DAILY_TAG}[START] 🎬 "):
                    keys.append("reset")
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🔍 Config 검수 중"):
                    keys.append("start")
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] 🤖 GPT 컨텍스트 분석 중"):
                    keys.append("guardian_rules")
                if text.startswith(f"{DAILY_TAG}[GUARDIAN] ✅ Config 검수 완료") or text.startswith(f"{DAILY_TAG}[GUARDIAN] ⚠️ GPT 분석 실패"):
                    keys.append("guardian_gpt")
                    keys.append("guardian_rules")
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
                if text.startswith(f"{DAILY_TAG}[OPTIMIZE] ✅ "):
                    keys.append("optimize")
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
                    keys.append("auto_review")
                if text.startswith(f"{DAILY_TAG}[AUTO] 🔄 AI 자동 개선 루프") or text.startswith(f"{DAILY_TAG}[FIXER] 🔧 Codex 수정 시작"):
                    keys.append("auto_review")
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
                    # mini_opt sub-keys all share the same thread (different bots post inside)
                    if key == "mini_opt":
                        anchor_threads["mini_opt_sim"] = thread
                        anchor_threads["mini_opt_score"] = thread
                        anchor_threads["mini_opt_prog"] = thread
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
                    await _send_file_with_token(
                        message.channel, message.channel.id, path, note, manager_bot_token
                    )
                except Exception:
                    await _send_text_with_token(
                        message.channel, message.channel.id,
                        f"{note} (파일 업로드 실패: `{path.name}`)",
                        manager_bot_token,
                    )

            await _send_text_with_token(
                message.channel,
                ch_id,
                f"▶️ `!novel-daily {episode_key}` 시작\n"
                f"리뷰 등급: `{review_tier}`\n"
                f"outer cycles: `{outer_cycles}`",
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
                            outer_max_cycles=outer_cycles,
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
                    _persist_usage_snapshot(
                        ch_id,
                        metrics=DAILY_SESSION_METRICS.get(ch_id),
                        status=DAILY_STATUS.get(ch_id),
                        episode_key=DAILY_EPISODE_KEYS.get(ch_id),
                        is_active=False,
                    )
                    DAILY_PENDING_REVIEW_TIER.pop(ch_id, None)
                    DAILY_WAITING_FEEDBACK.discard(ch_id)
                    DAILY_PROCESS_INFO.pop(ch_id, None)
                    DAILY_START_TIMES.pop(ch_id, None)
                    DAILY_START_TIMES_WALL.pop(ch_id, None)
                    if DAILY_FEEDBACK_QUEUES.get(ch_id) is _feedback_q:
                        DAILY_FEEDBACK_QUEUES.pop(ch_id, None)
                    if DAILY_STOP_EVENTS.get(ch_id) is _stop_ev:
                        DAILY_STOP_EVENTS.pop(ch_id, None)

            asyncio.create_task(_run_daily_task())
            return

        # ── !approve [req_id] ─────────────────────────────────────────────────
        if command == CMD_APPROVE:
            req_id = arg_text.split()[0] if arg_text else ""
            # 인자 없으면 플랜 승인으로 처리
            if not req_id:
                plan_pending = DAILY_PENDING_REVIEW_TIER.get(message.channel.id)
                if (
                    plan_pending
                    and plan_pending.get("stage") == "plan_approval"
                    and message.author.id == plan_pending.get("user_id")
                ):
                    DAILY_PENDING_REVIEW_TIER.pop(message.channel.id, None)
                    command = CMD_DAILY
                    arg_text = plan_pending.get("arg_text", "")
                    # CMD_DAILY 핸들러로 fall-through (아래 if command == CMD_DAILY 블록 진입)
                else:
                    await message.channel.send("사용법: `!approve <req_id>`\n플랜 승인은 `1` 또는 `2` 로 입력해주세요.")
                    return

        # config change approve (req_id 있을 때만)
        if command == CMD_APPROVE and req_id:
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

    async def _send_shutdown_farewell(sig_name: str) -> None:
        nonlocal shutdown_farewell_sent, manual_shutdown_in_progress
        if shutdown_farewell_sent:
            return
        shutdown_farewell_sent = True
        manual_shutdown_in_progress = True
        notify_channel_id = alert_channel_id or last_active_channel_id
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await _notify_system_channel(
            client,
            notify_channel_id,
            f"🛑 프로세스 종료 감지 ({sig_name}, {ts})\n팀이 잠시 오프라인 전환됩니다.",
            manager_bot_token,
        )
        await _send_team_disconnect_farewell(
            notify_channel_id,
            token,
            reviewer_bot_token,
            fixer_bot_token,
            manager_bot_token,
        )

    loop = asyncio.get_running_loop()

    def _handle_signal(sig_name: str) -> None:
        async def _graceful_shutdown() -> None:
            await _send_shutdown_farewell(sig_name)
            await client.close()
        asyncio.create_task(_graceful_shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig.name)
        except NotImplementedError:
            pass

    await client.start(token)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
