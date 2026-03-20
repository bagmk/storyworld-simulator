#!/usr/bin/env python3
"""
Daily Pipeline Orchestrator — Agent 1→2→3→4 pipeline for one episode cycle.

Flow:
  1. Config Guardian  — consistency check + config report
  2. Simulator        — simulate.py  (실시간 Turn 진행률 Discord 전송)
  3. Chapter Gen      — generate_chapter.py → chapter.txt
  4. Quality Reviewer — auto-checks + scorecard → Discord
  5. Wait for user feedback (up to 24h)
  6. Parse feedback + update story_state.json

Triggered by:
  - CLI: python tools/daily_pipeline.py --episode ep01_academic_presentation [--no-discord]
  - Discord: !novel-daily <episode_key>

stop_event: asyncio.Event — set it to interrupt the pipeline at any step boundary.
status_fn:  callback(str) — called to update a shared status string (for !daily-status).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.novel_writer.env_loader import load_project_env
from src.novel_writer.llm_client import LLMClient
from tools.config_guardian import run_guardian
from tools.quality_reviewer import (
    run_quality_review,
    parse_feedback_with_llm,
    update_story_state,
    resolve_episode_file,
)

DATA_DIR = REPO_ROOT / "data"
STORY_STATE_PATH = DATA_DIR / "story_state.json"
PENDING_PATH = DATA_DIR / "pending_config_changes.json"
OUTPUT_DIR = REPO_ROOT / "output"

DAILY_TAG = "[DAILY]"

NotifyFn = Callable[[str], Awaitable[None]] | None
UploadFn = Callable[[Path, str], Awaitable[None]] | None
StatusFn = Callable[[str], None] | None      # sync callback to update shared status string


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _allocate_daily_output_dir(episode_key: str) -> Path:
    now = datetime.now()
    run_dir = OUTPUT_DIR / "daily" / f"{now.strftime('%Y%m%d')}_{episode_key}" / now.strftime("%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _get_cycle_number(episode_key: str) -> int:
    state = _load_json(STORY_STATE_PATH)
    ep_data = state.get("episode_summaries", {}).get(episode_key, {})
    return ep_data.get("cycle_count", 0) + 1


def _increment_cycle(episode_key: str) -> None:
    state = _load_json(STORY_STATE_PATH)
    ep_data = state.setdefault("episode_summaries", {}).setdefault(episode_key, {})
    ep_data["cycle_count"] = ep_data.get("cycle_count", 0) + 1
    _save_json(STORY_STATE_PATH, state)


async def _stream_subprocess(
    cmd: list[str],
    on_line: Callable[[str], Awaitable[None]] | None = None,
    stop_event: asyncio.Event | None = None,
    timeout_sec: int = 3600,
    on_heartbeat: Callable[[int], Awaitable[None]] | None = None,
    heartbeat_sec: int = 120,
) -> tuple[int, str]:
    """
    Run a subprocess, stream stdout+stderr line-by-line through on_line callback.
    on_heartbeat(elapsed_sec) is called every heartbeat_sec if no output.
    Returns (returncode, combined_output).
    If stop_event is set, sends SIGTERM and returns rc=-1.
    """
    env = os.environ.copy()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )

    output_lines: list[str] = []
    last_output_time: list[float] = [asyncio.get_event_loop().time()]

    async def _read_lines() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line)
            last_output_time[0] = asyncio.get_event_loop().time()
            if on_line:
                await on_line(line)

    async def _watch_stop() -> None:
        if stop_event is None:
            return
        stop_wait = asyncio.create_task(stop_event.wait())
        proc_wait = asyncio.create_task(proc.wait())
        done, pending = await asyncio.wait(
            {stop_wait, proc_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if stop_wait in done and stop_event.is_set() and proc.returncode is None:
            proc.terminate()

    async def _heartbeat() -> None:
        if not on_heartbeat:
            return
        start = asyncio.get_event_loop().time()
        while True:
            await asyncio.sleep(heartbeat_sec)
            if proc.returncode is not None:
                break
            silence = asyncio.get_event_loop().time() - last_output_time[0]
            if silence >= heartbeat_sec * 0.8:
                elapsed = int(asyncio.get_event_loop().time() - start)
                await on_heartbeat(elapsed)

    try:
        await asyncio.wait_for(
            asyncio.gather(_read_lines(), _watch_stop(), _heartbeat()),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        proc.terminate()
        await proc.wait()
        return -2, "\n".join(output_lines)
    except Exception:
        pass

    await proc.wait()

    if stop_event and stop_event.is_set():
        return -1, "\n".join(output_lines)

    return proc.returncode or 0, "\n".join(output_lines)


# ── Step 1: Config Guardian ────────────────────────────────────────────────────

def _load_context_window_for_guardian(episode_key: str, window: int = 3) -> dict:
    """
    Load story_context.yaml + the current episode ±window YAML files.
    Returns {"story_context": str, "episodes": list[{"key": str, "role": str, "text": str}]}
    """
    ep_dir = REPO_ROOT / "config" / "episodes"
    story_context_path = REPO_ROOT / "config" / "story_context.yaml"

    story_context_text = ""
    if story_context_path.exists():
        story_context_text = story_context_path.read_text(encoding="utf-8")

    # Build sorted list of all episode files
    all_ep_files = sorted(ep_dir.glob("ep*.yaml"))

    # Find index of current episode
    current_idx = None
    for i, f in enumerate(all_ep_files):
        if f.stem == episode_key or f.stem.startswith(episode_key.split("_")[0] + "_") or f.stem == episode_key:
            current_idx = i
            break
    # Fallback: match by ep number prefix
    if current_idx is None:
        ep_num = episode_key[:4]  # e.g. "ep22"
        for i, f in enumerate(all_ep_files):
            if f.stem.startswith(ep_num):
                current_idx = i
                break

    episodes_ctx = []
    if current_idx is not None:
        start = max(0, current_idx - window)
        end = min(len(all_ep_files) - 1, current_idx + window)
        for i in range(start, end + 1):
            f = all_ep_files[i]
            if i < current_idx:
                role = f"이전 {current_idx - i}화 전"
            elif i == current_idx:
                role = "현재 화 (작성 대상)"
            else:
                role = f"다음 {i - current_idx}화 후"
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                text = "(읽기 실패)"
            episodes_ctx.append({"key": f.stem, "role": role, "text": text})

    return {"story_context": story_context_text, "episodes": episodes_ctx}


def _build_guardian_gpt_prompt(context: dict, rule_report: str) -> str:
    ep_blocks = "\n\n".join(
        f"### [{ep['role']}] {ep['key']}\n```yaml\n{ep['text']}\n```"
        for ep in context["episodes"]
    )
    return f"""당신은 소설 프로젝트의 Config Guardian입니다.
아래 자료를 읽고 현재 작성 대상 에피소드의 컨텍스트 일관성을 분석하세요.

## 전체 스토리 컨텍스트 (story_context.yaml)
```yaml
{context["story_context"]}
```

## 인접 에피소드 YAML (현재 화 ±3화)
{ep_blocks}

## 규칙 기반 자동 검수 결과
{rule_report}

---
다음 항목을 중심으로 한국어로 분석 리포트를 작성하세요:

1. **스토리 흐름 일관성**: 현재 화가 앞뒤 화의 사건/감정 흐름과 자연스럽게 연결되는지
2. **캐릭터 Arc 점검**: 현재 화에서 주요 캐릭터의 행동/감정이 전체 arc와 맞는지
3. **클루/복선 관리**: 이전 화에서 심은 복선이 적절히 활용/유지되고 있는지
4. **게이트 준수**: story_context의 gates 규칙을 위반하는 내용이 없는지
5. **개선 제안**: 현재 화 config에서 수정하면 좋을 구체적인 항목 (있으면)

분석은 간결하고 실용적으로 작성하세요."""


async def step_guardian(
    episode_key: str,
    run_dir: Path,
    cycle: int,
    notify: NotifyFn,
    upload: UploadFn,
    set_status: StatusFn,
    stop_event: asyncio.Event | None,
) -> tuple[bool, Path | None]:
    """Returns (success, guardian_briefing_path). briefing_path is None if GPT analysis failed."""
    if stop_event and stop_event.is_set():
        return False

    if set_status:
        set_status("1/4 Config Guardian 검수 중...")
    if notify:
        await notify(f"{DAILY_TAG}[GUARDIAN] 🔍 Config 검수 중...")

    try:
        report_text, _ = await asyncio.to_thread(
            run_guardian, Path("config/episodes"), False,
        )
    except Exception as exc:
        if notify:
            await notify(f"{DAILY_TAG}[GUARDIAN] ❌ Config 검수 실패: {type(exc).__name__}: {exc}")
        return False, None

    # Save rule-based report locally
    report_path = run_dir / "config_check.txt"
    report_path.write_text(report_text, encoding="utf-8")

    if notify:
        await notify(f"{DAILY_TAG}[GUARDIAN] Config 규칙 검수 결과:\n{report_text}")

    pending = _load_json(PENDING_PATH)
    pending_requests = [r for r in pending.get("requests", []) if r.get("status") == "pending"]

    if pending_requests:
        req_list = "\n".join(
            f"  • `{r['id']}`: {r['description']}\n"
            f"    → `!approve {r['id']}` 또는 `!reject {r['id']}`"
            for r in pending_requests
        )
        if notify:
            await notify(f"{DAILY_TAG}[GUARDIAN] ⚠️ Config 변경 요청 {len(pending_requests)}건:\n{req_list}")

    # ── GPT 컨텍스트 분석 ──
    if notify:
        await notify(f"{DAILY_TAG}[GUARDIAN] 🤖 GPT 컨텍스트 분석 중 (±3화 + story_context)...")

    try:
        context = await asyncio.to_thread(_load_context_window_for_guardian, episode_key)
        prompt = _build_guardian_gpt_prompt(context, report_text)

        llm = LLMClient(
            model="gpt-4o-mini",
            premium_model="gpt-4o",
            budget_usd=1.5,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        gpt_report = await asyncio.to_thread(
            llm.chat,
            [{"role": "user", "content": prompt}],
            use_premium=True,
            purpose="guardian_context_analysis",
            max_tokens=1500,
        )

        briefing_path = run_dir / "guardian_gpt_report.txt"
        briefing_path.write_text(gpt_report, encoding="utf-8")

        if notify:
            await notify(f"{DAILY_TAG}[GUARDIAN] 🧠 GPT 분석 리포트:\n{gpt_report}")

        if notify:
            await notify(f"{DAILY_TAG}[GUARDIAN] ✅ Config 검수 완료 — 브리핑이 챕터 생성에 사용됩니다")

        return True, briefing_path

    except Exception as exc:
        if notify:
            await notify(f"{DAILY_TAG}[GUARDIAN] ⚠️ GPT 분석 실패 (계속 진행): {type(exc).__name__}: {exc}")

    if notify:
        await notify(f"{DAILY_TAG}[GUARDIAN] ✅ Config 검수 완료")

    return True, None


# ── Step 2: Simulator ──────────────────────────────────────────────────────────

async def step_simulator(
    episode_key: str,
    run_dir: Path,
    cycle: int,
    budget: float,
    notify: NotifyFn,
    set_status: StatusFn,
    stop_event: asyncio.Event | None,
    guardian_briefing_path: Path | None = None,
) -> bool:
    if stop_event and stop_event.is_set():
        return False

    if set_status:
        set_status("2/4 시뮬레이션 준비 중...")
    if notify:
        await notify(f"{DAILY_TAG}[SIM] ⚙️ 시뮬레이션 시작 (사이클 {cycle})...")

    ep_file = resolve_episode_file(episode_key)

    cmd = [
        "python3", "simulate.py",
        "--episode", str(ep_file),
        "--characters", "config/characters.yaml",
        "--world", "config/world_facts.yaml",
        "--storyline", "config/storyline.yaml",
        "--output", str(run_dir),
        "--budget", str(budget * 0.5),
    ]
    if guardian_briefing_path and guardian_briefing_path.exists():
        cmd += ["--guardian-briefing", str(guardian_briefing_path)]

    turn_re = re.compile(r"Turn\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    last_notified_checkpoint = -1

    async def on_line(line: str) -> None:
        nonlocal last_notified_checkpoint
        m = turn_re.search(line)
        if not m:
            return
        turn = int(m.group(1))
        total = max(int(m.group(2)), 1)
        pct = int(turn / total * 100)

        for checkpoint in (25, 50, 75, 100):
            if pct >= checkpoint and last_notified_checkpoint < checkpoint:
                last_notified_checkpoint = checkpoint
                status_str = f"2/4 시뮬레이션 {checkpoint}% (Turn {turn}/{total})"
                if set_status:
                    set_status(status_str)
                if notify:
                    await notify(f"{DAILY_TAG}[SIM] ⚙️ {checkpoint}% (Turn {turn}/{total})")

    async def on_heartbeat_sim(elapsed: int) -> None:
        mins = elapsed // 60
        status = DAILY_STATUS_REF[0] if DAILY_STATUS_REF[0] else "시뮬레이션 중"
        if notify:
            await notify(f"{DAILY_TAG}[SIM] ⏳ LLM 응답 대기 중... ({mins}분 경과) — 현재: {status}")

    DAILY_STATUS_REF: list[str] = [""]
    original_set_status = set_status
    def _wrapped_set_status(s: str) -> None:
        DAILY_STATUS_REF[0] = s
        if original_set_status:
            original_set_status(s)
    set_status = _wrapped_set_status

    rc, output = await _stream_subprocess(
        cmd, on_line=on_line, stop_event=stop_event, timeout_sec=1800,
        on_heartbeat=on_heartbeat_sim, heartbeat_sec=120,
    )

    if rc == -1:
        if notify:
            await notify(f"{DAILY_TAG}[SIM] 🛑 사용자 중단 요청으로 시뮬레이션 종료")
        return False
    if rc == -2:
        if notify:
            await notify(f"{DAILY_TAG}[SIM] ❌ 시뮬레이션 타임아웃 (30분)")
        return False
    if rc != 0:
        err_preview = output[-800:] if output else "(출력 없음)"
        if notify:
            await notify(f"{DAILY_TAG}[SIM] ❌ 시뮬레이션 실패 (rc={rc}):\n```\n{err_preview}\n```")
        return False

    if set_status:
        set_status("2/4 시뮬레이션 완료")
    if notify:
        await notify(f"{DAILY_TAG}[SIM] ✅ 시뮬레이션 완료")
    return True


# ── Step 3: Chapter Generator ─────────────────────────────────────────────────

async def step_chapter_gen(
    episode_key: str,
    run_dir: Path,
    cycle: int,
    target_words: int,
    budget: float,
    protagonist: str,
    notify: NotifyFn,
    upload: UploadFn,
    set_status: StatusFn,
    stop_event: asyncio.Event | None,
    guardian_briefing_path: Path | None = None,
) -> Path | None:
    if stop_event and stop_event.is_set():
        return None

    if set_status:
        set_status("3/4 챕터 생성 중...")
    if notify:
        await notify(f"{DAILY_TAG}[CHAPTER] 📖 챕터 생성 중... (보통 5~10분)")

    ep_file = resolve_episode_file(episode_key)
    episode_id = ep_file.stem

    prev_review = None
    if cycle > 1:
        candidates = sorted(run_dir.parent.glob(f"**/{episode_id}_*review*.txt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            prev_review = candidates[-1]

    cmd = [
        "python3", "generate_chapter.py",
        "--episode", episode_id,
        "--episode-config", str(ep_file),
        "--protagonist", protagonist,
        "--output", str(run_dir),
        "--words", str(target_words),
        "--budget", str(budget * 0.5),
    ]
    if prev_review:
        cmd += ["--reader-review-md", str(prev_review)]
    if guardian_briefing_path and guardian_briefing_path.exists():
        cmd += ["--guardian-briefing", str(guardian_briefing_path)]

    scene_re = re.compile(r"scene\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    last_scene_checkpoint = -1

    async def on_line(line: str) -> None:
        nonlocal last_scene_checkpoint
        m = scene_re.search(line)
        if not m:
            return
        scene = int(m.group(1))
        total = max(int(m.group(2)), 1)
        pct = int(scene / total * 100)
        for checkpoint in (50, 100):
            if pct >= checkpoint and last_scene_checkpoint < checkpoint:
                last_scene_checkpoint = checkpoint
                if set_status:
                    set_status(f"3/4 챕터 생성 {checkpoint}% (Scene {scene}/{total})")
                if notify:
                    await notify(f"{DAILY_TAG}[CHAPTER] 📝 {checkpoint}% (Scene {scene}/{total})")

    async def on_heartbeat_chapter(elapsed: int) -> None:
        mins = elapsed // 60
        if notify:
            await notify(f"{DAILY_TAG}[CHAPTER] ⏳ 글 작성 중... ({mins}분 경과)")

    rc, output = await _stream_subprocess(
        cmd, on_line=on_line, stop_event=stop_event, timeout_sec=1800,
        on_heartbeat=on_heartbeat_chapter, heartbeat_sec=120,
    )

    if rc == -1:
        if notify:
            await notify(f"{DAILY_TAG}[CHAPTER] 🛑 사용자 중단 요청으로 챕터 생성 종료")
        return None
    if rc == -2:
        if notify:
            await notify(f"{DAILY_TAG}[CHAPTER] ❌ 챕터 생성 타임아웃")
        return None
    if rc != 0:
        err_preview = output[-800:] if output else "(출력 없음)"
        if notify:
            await notify(f"{DAILY_TAG}[CHAPTER] ❌ 챕터 생성 실패 (rc={rc}):\n```\n{err_preview}\n```")
        return None

    # Find the output file
    chapter_out = next(
        (p for p in [run_dir / f"{episode_id}_chapter.txt", run_dir / "chapter.txt"] if p.exists()),
        None,
    )
    if chapter_out is None:
        found = sorted(run_dir.glob("*chapter*.txt"), key=lambda p: p.stat().st_mtime)
        chapter_out = found[-1] if found else None

    if chapter_out is None:
        if notify:
            await notify(f"{DAILY_TAG}[CHAPTER] ❌ chapter.txt 파일 없음 (run_dir: `{run_dir}`)")
        return None

    word_count = len(chapter_out.read_text(encoding="utf-8", errors="replace").split())
    if set_status:
        set_status(f"3/4 챕터 완성 ({word_count}단어)")
    # 챕터 본문은 txt 파일로 전송
    if upload:
        await upload(chapter_out, f"📖 {ep_file.stem} — {word_count}단어")
    elif notify:
        await notify(f"{DAILY_TAG}[CHAPTER] ✅ 챕터 완성 ({word_count}단어) — `{chapter_out.name}`")

    return chapter_out


# ── Step 4: Quality Review ────────────────────────────────────────────────────

async def step_quality_review(
    episode_key: str,
    chapter_path: Path | None,
    run_dir: Path,
    cycle: int,
    notify: NotifyFn,
    upload: UploadFn,
    set_status: StatusFn,
    stop_event: asyncio.Event | None,
) -> str | None:
    if stop_event and stop_event.is_set():
        return None

    if set_status:
        set_status("4/4 품질 검수 중...")
    if notify:
        await notify(f"{DAILY_TAG}[REVIEW] 🔍 품질 자동 검수 중...")

    try:
        scorecard, auto_results = await asyncio.to_thread(
            run_quality_review, episode_key, chapter_path, run_dir, False,
        )
    except Exception as exc:
        if notify:
            await notify(f"{DAILY_TAG}[REVIEW] ❌ 검수 실패: {type(exc).__name__}: {exc}")
        return None

    scorecard_path = run_dir / "scorecard.txt"
    scorecard_path.write_text(scorecard, encoding="utf-8")

    if set_status:
        set_status("피드백 대기 중...")
    if notify:
        await notify(f"{DAILY_TAG}[REVIEW] ✅ 자동 검수 완료\n\n{scorecard}")

    return scorecard


# ── Feedback wait ─────────────────────────────────────────────────────────────

async def wait_for_feedback(
    feedback_queue: asyncio.Queue | None,
    timeout_hours: float,
    notify: NotifyFn,
    stop_event: asyncio.Event | None,
    on_start_wait: Callable[[], None] | None = None,
    on_end_wait: Callable[[], None] | None = None,
) -> str | None:
    if feedback_queue is None:
        return None

    # 피드백 대기 시작 — 이제부터 채널 메시지를 피드백으로 받음
    if on_start_wait:
        on_start_wait()

    if notify:
        await notify(
            f"{DAILY_TAG}[WAIT] ⏳ 피드백 대기 중 (최대 {timeout_hours:.0f}시간)\n"
            "읽어보시고 자유롭게 피드백 남겨주세요.\n"
            "'다음으로 가자' 또는 'next' → 다음 에피소드로 진행\n"
            "`!stop` → 파이프라인 중단"
        )

    timeout_sec = timeout_hours * 3600

    # Wait for either feedback or stop signal
    feedback_task = asyncio.create_task(feedback_queue.get())
    stop_task = asyncio.create_task(stop_event.wait()) if stop_event else None

    tasks = [feedback_task]
    if stop_task:
        tasks.append(stop_task)

    try:
        done, pending = await asyncio.wait(tasks, timeout=timeout_sec, return_when=asyncio.FIRST_COMPLETED)
    except Exception:
        done, pending = set(), set(tasks)

    for t in pending:
        t.cancel()

    if on_end_wait:
        on_end_wait()

    if stop_task and stop_task in done:
        if notify:
            await notify(f"{DAILY_TAG}[WAIT] 🛑 중단 요청으로 피드백 대기 종료")
        return None

    if feedback_task in done:
        try:
            return feedback_task.result()
        except Exception:
            pass

    if notify:
        await notify(f"{DAILY_TAG}[WAIT] ⏰ 피드백 대기 시간 초과 ({timeout_hours:.0f}시간)")
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_daily_pipeline(
    episode_key: str,
    target_words: int = 3500,
    budget: float = 4.0,
    protagonist: str = "kim_sumin",
    feedback_queue: asyncio.Queue | None = None,
    feedback_timeout_hours: float = 24.0,
    notify: NotifyFn = None,
    upload: UploadFn = None,
    no_discord: bool = False,
    stop_event: asyncio.Event | None = None,
    set_status: StatusFn = None,
    on_start_wait: Callable[[], None] | None = None,
    on_end_wait: Callable[[], None] | None = None,
) -> dict[str, Any]:
    load_project_env(REPO_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve episode key (support bare numbers)
    ep_file = resolve_episode_file(episode_key)
    episode_key = ep_file.stem  # normalise to full key e.g. "ep01_academic_presentation"

    cycle = _get_cycle_number(episode_key)
    run_dir = _allocate_daily_output_dir(episode_key)

    if set_status:
        set_status(f"시작 — {episode_key} 사이클 {cycle}")
    if notify:
        await notify(
            f"{DAILY_TAG}[START] 🎬 `{episode_key}` 파이프라인 시작 (사이클 {cycle})\n"
            f"- run: `{run_dir.relative_to(REPO_ROOT)}`\n"
            f"- 진행 상황: `!status` | 중단: `!stop`"
        )

    # ── Step 1: Config Guardian ──
    ok1, guardian_briefing_path = await step_guardian(episode_key, run_dir, cycle, notify, upload, set_status, stop_event)
    if not ok1:
        if set_status:
            set_status("중단됨 (Guardian 단계)")
        return {"success": False, "step": "guardian", "cycle": cycle}

    # ── Step 2: Simulator ──
    ok2 = await step_simulator(episode_key, run_dir, cycle, budget, notify, set_status, stop_event,
                               guardian_briefing_path=guardian_briefing_path)
    if not ok2:
        if set_status:
            set_status("중단됨 (Simulator 단계)")
        return {"success": False, "step": "simulator", "cycle": cycle}

    # ── Step 3: Chapter generation ──
    chapter_path = await step_chapter_gen(
        episode_key, run_dir, cycle, target_words, budget, protagonist,
        notify, upload, set_status, stop_event,
        guardian_briefing_path=guardian_briefing_path,
    )
    if chapter_path is None:
        if set_status:
            set_status("중단됨 (Chapter Gen 단계)")
        return {"success": False, "step": "chapter_gen", "cycle": cycle}

    # ── Step 4: Quality review ──
    scorecard = await step_quality_review(
        episode_key, chapter_path, run_dir, cycle, notify, upload, set_status, stop_event,
    )

    _increment_cycle(episode_key)

    if no_discord or feedback_queue is None:
        if set_status:
            set_status("완료 (no-discord)")
        if notify:
            await notify(
                f"{DAILY_TAG}[DONE] ✅ 파이프라인 완료 (no-discord 모드)\n"
                f"- chapter: `{chapter_path.relative_to(REPO_ROOT)}`"
            )
        return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": None, "feedback": None}

    # ── Step 5: Wait for feedback ──
    raw_feedback = await wait_for_feedback(
        feedback_queue, feedback_timeout_hours, notify, stop_event,
        on_start_wait=on_start_wait, on_end_wait=on_end_wait,
    )
    if raw_feedback is None:
        if set_status:
            set_status("완료 (피드백 없음)")
        return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": None, "feedback": None}

    # ── Step 6: Parse feedback + update story_state ──
    if set_status:
        set_status("피드백 분석 중...")
    llm = LLMClient(
        model="gpt-4o-mini", premium_model="gpt-4o", budget_usd=1.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    with ep_file.open(encoding="utf-8") as f:
        episode_data = (yaml.safe_load(f) or {}).get("episode", {})

    parsed = parse_feedback_with_llm(raw_feedback, episode_key, llm)
    update_story_state(STORY_STATE_PATH, episode_key, episode_data, parsed)

    approved = parsed.get("approved_next_episode", False)
    if set_status:
        set_status(f"완료 — {'승인됨' if approved else '재시도 예정'}")
    if notify:
        if approved:
            await notify(
                f"{DAILY_TAG}[DONE] ✅ 피드백 저장 완료. 다음 에피소드 승인됨.\n"
                "다음 에피소드를 시작하려면: `!novel-daily <번호>`"
            )
        else:
            issues = parsed.get("specific_issues", [])
            issue_str = "\n".join(f"  - {i}" for i in issues) if issues else "  (코멘트 참조)"
            await notify(
                f"{DAILY_TAG}[DONE] 📝 피드백 저장 완료. 다음 사이클에서 `{episode_key}` 재시도.\n"
                f"개선 포인트:\n{issue_str}"
            )

    return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": approved, "feedback": parsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Pipeline")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--target-words", type=int, default=3500)
    parser.add_argument("--budget", type=float, default=4.0)
    parser.add_argument("--protagonist", default="kim_sumin")
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    async def _run():
        result = await run_daily_pipeline(
            episode_key=args.episode,
            target_words=args.target_words,
            budget=args.budget,
            protagonist=args.protagonist,
            no_discord=args.no_discord,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    main()
