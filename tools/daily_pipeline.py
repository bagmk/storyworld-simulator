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

# Auto-improve loop settings
AUTO_IMPROVE_MAX_CYCLES = 3       # Codex fixer 최대 반복 횟수
AUTO_IMPROVE_SCORE_THRESHOLD = 8.5  # thrill+style 평균 이 점수 이상이면 통과 (10점 만점)
# 근거: novel-loop 실측 데이터상 좋은 챕터 평균 8.0, 최솟값 7.5.
# 7.0 기준은 사이클1에서 바로 통과되어 Codex가 실행되지 않음.

# Fixer: config/episodes는 건드리지 않음. 이 파일들만 수정 대상.
FIXER_TARGET_FILES = [
    "src/novel_writer/prose_generator.py",
    "src/novel_writer/scene_distiller.py",
    "src/novel_writer/director.py",
    "src/novel_writer/orchestrator.py",
    "generate_chapter.py",
    "simulate.py",
]

NotifyFn = Callable[[str], Awaitable[None]] | None
UploadFn = Callable[[Path, str], Awaitable[None]] | None
StatusFn = Callable[[str], None] | None      # sync callback to update shared status string
ProcessFn = Callable[[str | None, int | None, str | None], None] | None


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
    on_process_started: Callable[[int], None] | None = None,
    on_process_ended: Callable[[], None] | None = None,
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
    if on_process_started:
        on_process_started(proc.pid)

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
    if on_process_ended:
        on_process_ended()

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
        return False, None

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
    set_process: ProcessFn = None,
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
    current_turn_label = "준비 중"

    async def on_line(line: str) -> None:
        nonlocal current_turn_label
        if not line:
            return
        m = turn_re.search(line)
        if m:
            turn = int(m.group(1))
            total = max(int(m.group(2)), 1)
            current_turn_label = f"Turn {turn}/{total}"
            status_str = f"2/4 시뮬레이션 Turn {turn}/{total}"
            if set_status:
                set_status(status_str)
            if notify:
                await notify(f"{DAILY_TAG}[SIM] ⚙️ {current_turn_label}")
            return
        if notify:
            await notify(f"{DAILY_TAG}[SIM] ⚙️ {current_turn_label} | {line}")

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
        on_process_started=(lambda pid: set_process("simulator", pid, " ".join(cmd))) if set_process else None,
        on_process_ended=(lambda: set_process(None, None, None)) if set_process else None,
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
    set_process: ProcessFn = None,
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
    stage_re = re.compile(r"stage\s+(\d+)", re.IGNORECASE)
    current_scene_label: str | None = None
    current_stage_label = "준비 중"

    async def on_line(line: str) -> None:
        nonlocal current_scene_label, current_stage_label
        if not line:
            return
        stage_match = stage_re.search(line)
        if stage_match:
            current_stage_label = f"Stage {stage_match.group(1)}"
            if set_status and current_scene_label is None:
                set_status(f"3/4 챕터 생성 {current_stage_label}")
        m = scene_re.search(line)
        if m:
            scene = int(m.group(1))
            total = max(int(m.group(2)), 1)
            current_scene_label = f"Scene {scene}/{total}"
            if set_status:
                set_status(f"3/4 챕터 생성 {current_scene_label}")
            if notify:
                await notify(f"{DAILY_TAG}[CHAPTER] 📝 {current_scene_label}")
            return
        if notify:
            prefix = current_scene_label or current_stage_label
            emoji = "📝" if current_scene_label else "🧩"
            await notify(f"{DAILY_TAG}[CHAPTER] {emoji} {prefix} | {line}")

    async def on_heartbeat_chapter(elapsed: int) -> None:
        mins = elapsed // 60
        if notify:
            prefix = current_scene_label or current_stage_label
            await notify(f"{DAILY_TAG}[CHAPTER] ⏳ {prefix} | 글 작성 중... ({mins}분 경과)")

    rc, output = await _stream_subprocess(
        cmd, on_line=on_line, stop_event=stop_event, timeout_sec=1800,
        on_heartbeat=on_heartbeat_chapter, heartbeat_sec=120,
        on_process_started=(lambda pid: set_process("chapter", pid, " ".join(cmd))) if set_process else None,
        on_process_ended=(lambda: set_process(None, None, None)) if set_process else None,
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


# ── Auto-improve loop (AI Reviewer → Codex Fixer → Chapter regen) ─────────────

def _build_ai_reviewer_prompt(chapter_text: str) -> str:
    return (
        "You are a high-school student reader reviewing a Korean techno-thriller novel chapter.\n"
        "Focus only on readability, sentence flow, immersion, and whether it feels fun to read.\n"
        "Do NOT evaluate plot logic or story structure.\n"
        "Return strict JSON only:\n"
        "{\n"
        '  "thrill_score_10": int,\n'
        '  "style_score_10": int,\n'
        '  "one_line_verdict": string,\n'
        '  "what_felt_good": [string, ...],\n'
        '  "what_felt_boring_or_hard": [string, ...],\n'
        '  "style_tips": [string, ...],\n'
        '  "reader_comment": string\n'
        "}\n"
        "Rules: Be honest and concrete. Each list ≥ 3 items. reader_comment 4~6 sentences. Korean only.\n\n"
        f"Chapter text:\n{chapter_text[:16000]}"
    )


def _build_codex_fixer_prompt(review_json: dict) -> str:
    issues = review_json.get("what_felt_boring_or_hard", [])
    tips = review_json.get("style_tips", [])
    comment = review_json.get("reader_comment", "")
    thrill = review_json.get("thrill_score_10", "?")
    style = review_json.get("style_score_10", "?")
    return (
        f"독자 AI 리뷰 결과: 긴장감={thrill}/10, 문체={style}/10\n\n"
        f"문제점:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
        f"개선 팁:\n" + "\n".join(f"- {t}" for t in tips) + "\n\n"
        f"리뷰 전문: {comment}\n\n"
        "위 독자 리뷰를 바탕으로, 소설 생성 코드의 품질을 개선하라.\n"
        "수정 대상 파일:\n"
        "- src/novel_writer/prose_generator.py (씬 프롬프트, 가독성 규칙)\n"
        "- src/novel_writer/scene_distiller.py (씬 압축 로직)\n"
        "- src/novel_writer/director.py (씬 진행 판단)\n"
        "- generate_chapter.py (챕터 생성 파이프라인)\n\n"
        "규칙:\n"
        "1. config/episodes/ 파일은 절대 수정 금지\n"
        "2. prose_generator.py의 콜론 대사 금지 규칙(COLON_DIALOGUE_LABEL_BAN)은 수정/삭제 금지\n"
        "3. 독자가 지적한 문제를 실제 코드 수정으로 해결하라\n"
        "4. 수정 후 변경된 파일 목록을 한국어로 요약하라\n"
        "5. 기존 변경사항을 보존한 채 최소 diff로 수정하라\n"
        "6. 확인 질문 없이 바로 수정하라"
    )


def _backup_target_files(run_dir: Path, fixer_cycle: int) -> Path:
    """Codex 실행 전 수정 대상 파일을 백업. 백업 디렉토리 경로 반환."""
    backup_dir = run_dir / f"backup_before_fixer_cycle{fixer_cycle}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in FIXER_TARGET_FILES:
        src = REPO_ROOT / rel_path
        if src.exists():
            dst = backup_dir / src.name
            dst.write_bytes(src.read_bytes())
    return backup_dir


async def _git_commit_fixer_changes(fixer_cycle: int, episode_key: str, summary: str) -> tuple[bool, str]:
    """Codex가 수정한 파일을 git add + commit. 성공 여부와 메시지 반환."""
    # Stage only fixer target files that actually changed
    add_cmd = ["git", "add"] + [f for f in FIXER_TARGET_FILES if (REPO_ROOT / f).exists()]
    proc_add = await asyncio.create_subprocess_exec(
        *add_cmd, cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    await proc_add.wait()

    # Check if there's anything to commit
    proc_diff = await asyncio.create_subprocess_exec(
        "git", "diff", "--cached", "--quiet",
        cwd=str(REPO_ROOT),
    )
    await proc_diff.wait()
    if proc_diff.returncode == 0:
        return False, "변경된 파일 없음 (이미 최신 상태)"

    short_summary = summary[:200].replace("\n", " ")
    commit_msg = (
        f"auto: codex fixer improvements — {episode_key} cycle {fixer_cycle}\n\n"
        f"{short_summary}\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    proc_commit = await asyncio.create_subprocess_exec(
        "git", "commit", "-m", commit_msg,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc_commit.communicate()
    out = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
    if proc_commit.returncode != 0:
        return False, f"git commit 실패: {out}"
    return True, out


async def _run_codex_fixer(prompt: str, run_dir: Path, fixer_cycle: int) -> tuple[bool, str]:
    """Codex CLI로 코드를 직접 수정. 성공 여부와 요약 반환."""
    summary_path = run_dir / f"fixer_cycle{fixer_cycle}_summary.md"
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        "-o", str(summary_path),
        prompt,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=900)
    except asyncio.TimeoutError:
        proc.terminate()
        return False, "Codex fixer 타임아웃 (15분)"

    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    if proc.returncode != 0:
        return False, f"Codex 실패 (rc={proc.returncode})\n{output[-800:]}"

    summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else output[-1000:]
    return True, summary


async def step_auto_improve_loop(
    episode_key: str,
    run_dir: Path,
    chapter_path: Path,
    target_words: int,
    budget: float,
    protagonist: str,
    guardian_briefing_path: Path | None,
    notify: NotifyFn,
    set_status: StatusFn,
    stop_event: asyncio.Event | None,
    max_cycles: int = AUTO_IMPROVE_MAX_CYCLES,
    score_threshold: int = AUTO_IMPROVE_SCORE_THRESHOLD,
) -> Path:
    """
    AI 리뷰 → Codex Fixer → 챕터 재생성 루프.
    점수 통과 또는 max_cycles 도달 시 최종 chapter_path 반환.
    """
    current_chapter = chapter_path

    for fixer_cycle in range(1, max_cycles + 1):
        if stop_event and stop_event.is_set():
            break

        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — 리뷰 중...")
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 🔄 AI 자동 개선 루프 {fixer_cycle}/{max_cycles} 시작")

        # ── AI 리뷰 ──
        chapter_text = current_chapter.read_text(encoding="utf-8", errors="replace")
        try:
            llm = LLMClient(
                model="gpt-4o-mini",
                premium_model="gpt-4o",
                budget_usd=2.0,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            review_raw = await asyncio.to_thread(
                llm.chat,
                [{"role": "user", "content": _build_ai_reviewer_prompt(chapter_text)}],
                use_premium=True,
                purpose="auto_improve_reviewer",
                max_tokens=1200,
            )
            cleaned = re.sub(r"```(?:json)?\n?", "", review_raw).strip().rstrip("`")
            review_json = json.loads(cleaned)
        except Exception as exc:
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] ⚠️ 리뷰 실패 ({exc}), 루프 종료")
            break

        thrill = int(review_json.get("thrill_score_10", 0))
        style = int(review_json.get("style_score_10", 0))
        avg = (thrill + style) / 2
        verdict = review_json.get("one_line_verdict", "")

        # 리뷰 저장
        review_path = run_dir / f"auto_review_cycle{fixer_cycle}.json"
        review_path.write_text(json.dumps(review_json, ensure_ascii=False, indent=2), encoding="utf-8")

        if notify:
            await notify(
                f"{DAILY_TAG}[AUTO] 📊 AI 리뷰 결과 (사이클 {fixer_cycle})\n"
                f"긴장감: {thrill}/10 | 문체: {style}/10 | 평균: {avg:.1f}/10\n"
                f"한줄평: {verdict}"
            )

        # ── 점수 통과 확인 ──
        if avg >= score_threshold:
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] ✅ 품질 통과 (평균 {avg:.1f} ≥ {score_threshold}) "
                    f"— 사이클 {fixer_cycle}에서 완료"
                )
            break

        if fixer_cycle == max_cycles:
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] ⚠️ 최대 사이클({max_cycles}) 도달 (평균 {avg:.1f}) "
                    "— 현재 버전으로 진행"
                )
            break

        # ── Codex Fixer ──
        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — Codex 코드 수정 중...")
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 🔧 Codex Fixer 실행 중... (코드 자동 수정)")

        # 수정 전 백업
        backup_dir = await asyncio.to_thread(_backup_target_files, run_dir, fixer_cycle)
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 💾 이전 버전 백업 완료 → `{backup_dir.name}/`")

        fixer_prompt = _build_codex_fixer_prompt(review_json)
        ok, summary = await _run_codex_fixer(fixer_prompt, run_dir, fixer_cycle)

        if not ok:
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] ❌ Codex Fixer 실패: {summary}")
            break

        if notify:
            await notify(f"{DAILY_TAG}[AUTO] ✅ 코드 수정 완료:\n{summary[:800]}")

        # 수정 후 자동 git commit
        committed, commit_msg = await _git_commit_fixer_changes(fixer_cycle, episode_key, summary)
        if notify:
            if committed:
                await notify(f"{DAILY_TAG}[AUTO] 📦 git commit 완료 (사이클 {fixer_cycle}): `{commit_msg[:120]}`")
            else:
                await notify(f"{DAILY_TAG}[AUTO] ℹ️ git commit 스킵: {commit_msg}")

        # ── 챕터 재생성 ──
        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — 챕터 재생성 중...")
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 📖 수정된 코드로 챕터 재생성 중...")

        new_chapter = await step_chapter_gen(
            episode_key, run_dir, fixer_cycle, target_words, budget, protagonist,
            notify=None,  # 재생성은 조용히
            upload=None,
            set_status=None,
            stop_event=stop_event,
            guardian_briefing_path=guardian_briefing_path,
        )
        if new_chapter:
            current_chapter = new_chapter
            if notify:
                wc = len(current_chapter.read_text(encoding="utf-8").split())
                await notify(f"{DAILY_TAG}[AUTO] 📝 재생성 완료 ({wc}단어)")
        else:
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] ⚠️ 챕터 재생성 실패 — 이전 버전 유지")
            break

    return current_chapter


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
            f"{DAILY_TAG}[WAIT] ⏳ 피드백을 기다리고 있습니다. (최대 {timeout_hours:.0f}시간)\n"
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
    set_process: ProcessFn = None,
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
        await notify(f"{DAILY_TAG}[START] 🎬 `{episode_key}` 파이프라인 시작 (사이클 {cycle})")
        await notify(
            f"{DAILY_TAG}[START] run: `{run_dir.relative_to(REPO_ROOT)}`\n"
            "진행 상황: `!status` | 중단: `!stop`"
        )

    # ── Step 1: Config Guardian ──
    ok1, guardian_briefing_path = await step_guardian(episode_key, run_dir, cycle, notify, upload, set_status, stop_event)
    if not ok1:
        if set_status:
            set_status("중단됨 (Guardian 단계)")
        return {"success": False, "step": "guardian", "cycle": cycle}

    # ── Step 2: Simulator ──
    ok2 = await step_simulator(episode_key, run_dir, cycle, budget, notify, set_status, stop_event,
                               set_process=set_process,
                               guardian_briefing_path=guardian_briefing_path)
    if not ok2:
        if set_status:
            set_status("중단됨 (Simulator 단계)")
        return {"success": False, "step": "simulator", "cycle": cycle}

    # ── Step 3: Chapter generation ──
    chapter_path = await step_chapter_gen(
        episode_key, run_dir, cycle, target_words, budget, protagonist,
        notify, upload, set_status, stop_event,
        set_process=set_process,
        guardian_briefing_path=guardian_briefing_path,
    )
    if chapter_path is None:
        if set_status:
            set_status("중단됨 (Chapter Gen 단계)")
        return {"success": False, "step": "chapter_gen", "cycle": cycle}

    # ── Step 3.5: AI 자동 개선 루프 (AI 리뷰 → Codex Fixer → 챕터 재생성) ──
    if notify:
        await notify(
            f"{DAILY_TAG}[AUTO] 🚀 AI 자동 개선 루프 시작 "
            f"(최대 {AUTO_IMPROVE_MAX_CYCLES}사이클, 목표 평균 {AUTO_IMPROVE_SCORE_THRESHOLD}/10)"
        )
    chapter_path = await step_auto_improve_loop(
        episode_key, run_dir, chapter_path, target_words, budget, protagonist,
        guardian_briefing_path=guardian_briefing_path,
        notify=notify,
        set_status=set_status,
        stop_event=stop_event,
    )

    # ── Step 4: Final quality review → user ──
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
        if notify:
            await notify(
                f"{DAILY_TAG}[DONE] ℹ️ 피드백이 없어 여기서 마무리했습니다.\n"
                "원하면 나중에 다시 `!novel-daily <번호>`로 실행하거나, 새 피드백 기준으로 재시작하면 됩니다."
            )
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
                "이 에피소드는 여기서 마무리됐고, 다음엔 `!novel-daily <번호>`로 다음 화를 시작하면 됩니다."
            )
        else:
            issues = parsed.get("specific_issues", [])
            issue_str = "\n".join(f"  - {i}" for i in issues) if issues else "  (코멘트 참조)"
            await notify(
                f"{DAILY_TAG}[DONE] 📝 피드백 저장 완료. 다음 사이클에서 `{episode_key}` 재시도.\n"
                f"개선 포인트:\n{issue_str}\n"
                "지금은 여기서 멈추며, 같은 화를 다시 돌리려면 `!novel-daily <번호>`를 다시 실행하면 됩니다."
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
