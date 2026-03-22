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
import ast
import asyncio
import copy
import difflib
import json
import logging
import os
import re
import shutil
import sys
import time
import yaml

logger = logging.getLogger(__name__)
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.novel_writer.env_loader import load_project_env
from src.novel_writer import database as story_db
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
EPISODE_ARCHIVE_DIR = DATA_DIR / "episode_archives"
OUTPUT_DIR = REPO_ROOT / "output"

DAILY_TAG = ""

# Serialise concurrent Codex fixer runs (prevents simultaneous file writes across channels)
_CODEX_FIXER_LOCK: asyncio.Lock | None = None

def _get_codex_fixer_lock() -> asyncio.Lock:
    global _CODEX_FIXER_LOCK
    if _CODEX_FIXER_LOCK is None:
        _CODEX_FIXER_LOCK = asyncio.Lock()
    return _CODEX_FIXER_LOCK


def _resolve_python_cmd() -> str:
    """Use the same interpreter that launched this process (venv-safe)."""
    return sys.executable or "python3"


CODEX_DISABLED = True  # Codex 크레딧 소진 — GPT API fallback 사용

def _resolve_codex_exec_cmd(codex_model: str | None = None) -> list[str] | None:
    """
    Resolve a runnable Codex CLI command.
    Returns None if Codex is unavailable in PATH and project venv, or if CODEX_DISABLED.
    """
    if CODEX_DISABLED:
        return None
    codex_bin = shutil.which("codex")
    if not codex_bin:
        venv_codex = REPO_ROOT / ".venv" / "bin" / "codex"
        if venv_codex.exists():
            codex_bin = str(venv_codex)
    if not codex_bin:
        return None
    cmd = [codex_bin, "exec"]
    if codex_model:
        cmd += ["-m", codex_model]
    return cmd

# Auto-improve loop settings
AUTO_IMPROVE_MAX_CYCLES = 20      # (legacy) 이전 interleaved 루프 최대 사이클 수
AUTO_IMPROVE_SCORE_THRESHOLD = 9.5  # 평균 점수 통과 기준 (10점 만점)
AUTO_OUTER_MAX_CYCLES = 3         # 배치 파라미터 탐색 outer 루프 최대 반복 수
AUTO_INNER_MAX_CYCLES = 3         # outer 루프 1회당 Codex fix inner 루프 최대 수
AUTO_BATCH_TRIALS = 25            # outer 루프 1회당 파라미터 서브트라이얼 수
AUTO_BATCH_GROUP_SIZE = 5         # 서브트라이얼 병렬 처리 그룹 크기
# 근거: novel-loop 실측 데이터상 좋은 챕터 평균 8.0, 최솟값 7.5.
# 7.0 기준은 사이클1에서 바로 통과되어 Codex가 실행되지 않음.

# Manager agent settings (novel-loop 구조 이식)
MANAGER_PERIOD = 5          # N번 daily 사이클마다 심층 회고 (novel-loop의 manager_period와 동일)
MANAGER_HISTORY_MAX = 5     # 히스토리에서 최대 불러올 과거 리뷰 개수

# Fixer: config/episodes는 건드리지 않음. 이 파일들만 수정 대상.
FIXER_TARGET_FILES = [
    "src/novel_writer/prose_generator.py",
    "src/novel_writer/scene_distiller.py",
    "src/novel_writer/director.py",
    "src/novel_writer/orchestrator.py",
    "generate_chapter.py",
    "simulate.py",
    "tools/inline_optimizer.py",   # Codex can expand Optuna param space
    "data/rl_policy.json",         # Codex can register new param defaults
]

SIMULATION_RELEVANT_FIXER_FILES = {
    "simulate.py",
    "src/novel_writer/director.py",
    "src/novel_writer/orchestrator.py",
}

SCENE_CACHE_SAFE_FIXER_FILES = {
    "src/novel_writer/prose_generator.py",
    "generate_chapter.py",
}

NotifyFn = Callable[[str], Awaitable[None]] | None
UploadFn = Callable[[Path, str], Awaitable[None]] | None
StatusFn = Callable[[str], None] | None      # sync callback to update shared status string
ProcessFn = Callable[[str | None, int | None, str | None], None] | None
MetricsFn = Callable[[dict[str, Any]], None] | None


def _use_premium_review_tier(review_tier: str) -> bool:
    return str(review_tier).strip().lower() == "premium"

def _use_mini_review_tier(review_tier: str) -> bool:
    return str(review_tier).strip().lower() == "mini"

def _llm_model_for_tier(review_tier: str) -> str:
    """mini 티어이면 gpt-4o-mini, 그 외엔 gpt-4o-mini (base). premium은 premium_model로 승격."""
    return "gpt-4o-mini"

def _llm_premium_model_for_tier(review_tier: str) -> str:
    """mini 티어이면 gpt-4o-mini. 그 외엔 gpt-4o."""
    return "gpt-4o-mini" if _use_mini_review_tier(review_tier) else "gpt-4o"

def _llm_review_model_for_tier(review_tier: str) -> str:
    """AI 루프 리뷰 모델. mini 티어이면 gpt-5-mini (판단 품질 유지). 그 외엔 gpt-4o."""
    return "gpt-5-mini" if _use_mini_review_tier(review_tier) else "gpt-4o"

def _codex_model_for_tier(review_tier: str) -> str | None:
    """mini 티어이면 gpt-5.4-mini, 그 외엔 None (config.toml 기본값 사용)."""
    return "gpt-5.4-mini" if _use_mini_review_tier(review_tier) else None


def _format_auto_progress_label(auto_cycle_index: int | None, auto_max_cycles: int | None) -> str:
    if auto_cycle_index is None or auto_max_cycles is None or auto_max_cycles <= 0:
        return ""
    return f" (AUTO {auto_cycle_index}/{auto_max_cycles})"


def _empty_budget_summary(budget_usd: float = 0.0) -> dict[str, Any]:
    return {
        "spent_usd": 0.0,
        "budget_usd": float(budget_usd or 0.0),
        "call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "breakdown": [],
    }


def _parse_budget_from_output(output: str) -> dict[str, Any]:
    patterns = [
        re.compile(
            r"Budget used:\s*\$([0-9.]+)\s*/\s*\$([0-9.]+).+?over\s+(\d+)\s+LLM calls(?:\s*\|\s*tokens:\s*(\d+)\s*in\s*\+\s*(\d+)\s*out\s*=\s*(\d+)\s*total)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"Budget:\s*\$([0-9.]+)\s*/\s*\$([0-9.]+)\s+over\s+(\d+)\s+LLM calls(?:\s*\|\s*tokens:\s*(\d+)\s*in\s*\+\s*(\d+)\s*out\s*=\s*(\d+)\s*total)?",
            re.IGNORECASE,
        ),
    ]
    for line in reversed(output.splitlines()):
        for pattern in patterns:
            m = pattern.search(line)
            if m:
                return {
                    "spent_usd": float(m.group(1)),
                    "budget_usd": float(m.group(2)),
                    "call_count": int(m.group(3)),
                    "prompt_tokens": int(m.group(4) or 0),
                    "completion_tokens": int(m.group(5) or 0),
                    "total_tokens": int(m.group(6) or 0),
                    "breakdown": [],
                }
    return _empty_budget_summary()


def _format_budget_line(label: str, budget: dict[str, Any]) -> str:
    return (
        f"{label}: ${float(budget.get('spent_usd', 0.0)):.4f}"
        f" / ${float(budget.get('budget_usd', 0.0)):.2f}"
        f" ({int(budget.get('call_count', 0))} calls, "
        f"{int(budget.get('total_tokens', 0))} tokens)"
    )


def _load_chapter_budget_meta(run_dir: Path, episode_id: str, chapter_path: Path | None = None) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if chapter_path is not None:
        candidates.append(chapter_path.with_name(f"{chapter_path.stem}_meta.json"))
    candidates.extend(
        [
            run_dir / f"{episode_id}_chapter_meta.json",
            run_dir / "chapter_meta.json",
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            meta = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(meta, dict):
            budget = meta.get("budget")
            if isinstance(budget, dict):
                return budget
            if "spent_usd" in meta and "total_tokens" in meta:
                return meta
    return None


def _generate_quality_chart(
    episode_key: str,
    run_dir: Path,
    cost_tracker: dict[str, Any] | None,
    time_tracker: dict[str, float] | None = None,
    review_tier: str = "premium",
) -> Path | None:
    """
    이번 파이프라인 실행 결과를 matplotlib 차트로 생성.
    - 상단: AUTO 루프 사이클별 thrill/style 점수 추세
    - 하단: 비용/모델시간/모델토큰 파이 차트 (가능한 항목을 한 줄에 배치)
    PNG 파일 경로 반환. matplotlib 미설치 시 None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import numpy as np
    except ImportError:
        return None

    # 한글 폰트 설정 (macOS AppleGothic → 없으면 기본 유지)
    _korean_fonts = ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
    _available = {f.name for f in fm.fontManager.ttflist}
    for _kf in _korean_fonts:
        if _kf in _available:
            plt.rcParams["font.family"] = _kf
            plt.rcParams["axes.unicode_minus"] = False
            break

    # ── 5개 점수 로드 ──────────────────────────────────────────────────────────
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
            row = [float(data.get(k, 0)) for k, _, _ in SCORE_DEFS]
            for (k, _, _), v in zip(SCORE_DEFS, row):
                score_data[k].append(v)
            avg_scores.append(sum(row) / len(row))
        except Exception:
            pass

    # ── 비용: 3개 카테고리로 합산 ──────────────────────────────────────────────
    tracker = cost_tracker or {}
    cat_cost = {
        "시뮬레이션": float(tracker.get("simulation", 0.0)),
        "챕터생성":   float(tracker.get("chapter", 0.0)) + float(tracker.get("auto_chapter", 0.0)),
        "리뷰":       (
            float(tracker.get("guardian", 0.0))
            + float(tracker.get("manager", 0.0))
            + float(tracker.get("auto_review", 0.0))
            + float(tracker.get("code_review", 0.0))
            + float(tracker.get("regen_check", 0.0))
            + float(tracker.get("final_review", 0.0))
            + float(tracker.get("feedback_parse", 0.0))
        ),
    }
    cost_vals_nonzero = [(l, v) for l, v in cat_cost.items() if v > 0.0001]

    # ── 소요 시간: 3개 카테고리로 합산 ────────────────────────────────────────
    tracker_t = time_tracker or {}
    cat_time = {
        "시뮬레이션": tracker_t.get("simulator", 0.0),
        "챕터생성":   tracker_t.get("chapter_gen", 0.0) + tracker_t.get("auto_improve", 0.0),
        "리뷰":       tracker_t.get("quality_review", 0.0) + tracker_t.get("guardian", 0.0),
    }
    time_vals_nonzero = [(l, v) for l, v in cat_time.items() if v > 0.5]

    # ── 모델별 시간 비율 (근사치) ───────────────────────────────────────────────
    # 각 스텝이 사용하는 모델을 알고 있으므로 시간을 비례 배분
    # simulator/chapter_gen: subprocess → gpt-4o-mini (에이전트/기본) + gpt-5-mini (디렉터/산문)
    # guardian/quality_review: review_tier에 따라 결정
    # auto_improve: Codex CLI (픽서) + gpt-5-mini (챕터 재생성 산문) + tier 모델 (리뷰)
    _is_premium = str(review_tier).strip().lower() == "premium"
    _is_mini    = _use_mini_review_tier(review_tier)
    _mt: dict[str, float] = {}

    def _mt_add(model: str, secs: float) -> None:
        _mt[model] = _mt.get(model, 0.0) + secs

    _sim = tracker_t.get("simulator", 0.0)
    _mt_add("gpt-4o-mini", _sim * 0.50)   # 에이전트 턴
    _mt_add("gpt-5-mini",  _sim * 0.50)   # 디렉터

    _ch = tracker_t.get("chapter_gen", 0.0)
    _mt_add("gpt-4o-mini",  _ch * 0.40)   # 기본 구성
    _ch_prose = "gpt-4.1-mini" if _is_mini else "gpt-5-mini"
    _mt_add(_ch_prose,       _ch * 0.60)   # 산문 생성

    _guard = tracker_t.get("guardian", 0.0)
    _mt_add("gpt-4o" if _is_premium else "gpt-4o-mini", _guard)

    _qr = tracker_t.get("quality_review", 0.0)
    _mt_add("gpt-4o" if _is_premium else "gpt-4o-mini", _qr)

    _ai = tracker_t.get("auto_improve", 0.0)
    _mt_add("Codex CLI",   _ai * 0.50)    # 픽서 subprocess
    _mt_add(_ch_prose,     _ai * 0.20)    # 챕터 재생성 산문 (챕터생성과 동일 모델)
    _mt_add("gpt-4o-mini", _ai * 0.10)    # 챕터 재생성 에이전트
    _mt_add(_llm_review_model_for_tier(review_tier), _ai * 0.20)  # 사이클 리뷰

    model_time_nonzero = [(m, t) for m, t in _mt.items() if t > 0.5]
    model_token_totals = tracker.get("model_token_totals", {})
    model_token_nonzero: list[tuple[str, int]] = []
    if isinstance(model_token_totals, dict):
        for model, total_tokens in model_token_totals.items():
            try:
                total_value = int(total_tokens or 0)
            except (TypeError, ValueError):
                total_value = 0
            if total_value > 0:
                model_token_nonzero.append((str(model), total_value))
        model_token_nonzero.sort(key=lambda item: item[1], reverse=True)

    has_scores  = len(cycles) > 0
    has_costs   = len(cost_vals_nonzero) > 0
    has_times   = len(time_vals_nonzero) > 0
    has_model_times = len(model_time_nonzero) > 0
    has_model_tokens = len(model_token_nonzero) > 0
    has_pie_row = has_costs or has_model_times or has_model_tokens
    n_rows = (
        (1 if has_scores else 0)
        + (1 if has_times else 0)
        + (1 if has_pie_row else 0)
    )
    if n_rows == 0:
        return None

    pie_cols = (1 if has_costs else 0) + (1 if has_model_times else 0) + (1 if has_model_tokens else 0)
    fig_w = 14 if pie_cols >= 3 else (12 if pie_cols == 2 else 9)
    fig = plt.figure(figsize=(fig_w, 4 * n_rows))
    grid = fig.add_gridspec(n_rows, 1)
    fig.suptitle(f"[{episode_key}] 파이프라인 품질 리포트", fontsize=13, fontweight="bold")

    ax_idx = 0

    if has_scores:
        ax = fig.add_subplot(grid[ax_idx]); ax_idx += 1
        x = list(range(len(cycles)))
        for k, label, color in SCORE_DEFS:
            ax.plot(x, score_data[k], "o-", color=color, label=label, linewidth=2, markersize=6)
        ax.plot(x, avg_scores, "^-", color="#000000", label="평균",
                linewidth=2.5, markersize=7, zorder=5)
        ax.axhline(y=AUTO_IMPROVE_SCORE_THRESHOLD, color="#e67e22", linestyle=":",
                   linewidth=1.5, label=f"목표 {AUTO_IMPROVE_SCORE_THRESHOLD}")
        ax.set_ylim(0, 10.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"사이클{c}" for c in cycles], fontsize=8)
        ax.set_ylabel("점수 (/ 10)", fontsize=10)
        ax.set_title("AI 리뷰 점수 추세 (5개 항목)", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        for xi, yi in zip(x, avg_scores):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, fontweight="bold", color="#000000")

    if has_times:
        ax = fig.add_subplot(grid[ax_idx]); ax_idx += 1
        t_labels = [l for l, _ in time_vals_nonzero]
        t_vals   = [v for _, v in time_vals_nonzero]
        total_sec = sum(t_vals)
        cat_colors = {"시뮬레이션": "#3498db", "챕터생성": "#e74c3c", "리뷰": "#2ecc71"}
        bar_colors = [cat_colors.get(l, "#9b59b6") for l in t_labels]
        bars = ax.barh(t_labels, [v / 60 for v in t_vals], color=bar_colors, edgecolor="white", height=0.55)
        for bar, sec in zip(bars, t_vals):
            mins = int(sec // 60)
            secs = int(sec % 60)
            label = f"{mins}분 {secs:02d}초" if mins > 0 else f"{secs}초"
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left", fontsize=9)
        total_mins = int(total_sec // 60)
        total_secs = int(total_sec % 60)
        ax.set_xlabel("소요 시간 (분)", fontsize=10)
        ax.set_title(f"단계별 소요 시간 (총 {total_mins}분 {total_secs:02d}초)", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, max(v / 60 for v in t_vals) * 1.25)

    if has_pie_row:
        pie_grid = grid[ax_idx].subgridspec(1, pie_cols)
        ax_idx += 1
        pie_ax_idx = 0
        model_colors = {
            "gpt-4o-mini":  "#3498db",
            "gpt-4.1-mini": "#1abc9c",
            "gpt-4o":       "#e74c3c",
            "gpt-5-mini":   "#9b59b6",
            "Codex CLI":    "#2ecc71",
        }

        if has_costs:
            ax = fig.add_subplot(pie_grid[0, pie_ax_idx]); pie_ax_idx += 1
            labels_nz = [l for l, _ in cost_vals_nonzero]
            vals_nz = [v for _, v in cost_vals_nonzero]
            total_cost = sum(vals_nz)
            cat_colors = {"시뮬레이션": "#3498db", "챕터생성": "#e74c3c", "리뷰": "#2ecc71"}
            colors = [cat_colors.get(l, "#9b59b6") for l in labels_nz]
            _, texts, autotexts = ax.pie(
                vals_nz, labels=labels_nz, autopct="%1.1f%%",
                colors=colors, startangle=140, pctdistance=0.75,
            )
            for t in texts:
                t.set_fontsize(10)
            for at in autotexts:
                at.set_fontsize(9)
            ax.set_title(f"LLM 비용 구성 (총 ${total_cost:.4f}, Codex CLI 제외)", fontsize=11)

        if has_model_times:
            ax = fig.add_subplot(pie_grid[0, pie_ax_idx]); pie_ax_idx += 1
            m_labels = [m for m, _ in model_time_nonzero]
            m_vals   = [t for _, t in model_time_nonzero]
            total_model_sec = sum(m_vals)
            m_colors = [model_colors.get(m, "#f39c12") for m in m_labels]
            _, texts, autotexts = ax.pie(
                m_vals, labels=m_labels, autopct="%1.1f%%",
                colors=m_colors, startangle=140, pctdistance=0.75,
            )
            for t in texts:
                t.set_fontsize(10)
            for at in autotexts:
                at.set_fontsize(9)
            total_model_min = int(total_model_sec // 60)
            total_model_s   = int(total_model_sec % 60)
            ax.set_title(
                f"모델별 사용 시간 비율 (총 {total_model_min}분 {total_model_s:02d}초, 근사치)",
                fontsize=11,
            )

        if has_model_tokens:
            ax = fig.add_subplot(pie_grid[0, pie_ax_idx]); pie_ax_idx += 1
            tok_labels = [m for m, _ in model_token_nonzero]
            tok_vals = [t for _, t in model_token_nonzero]
            total_model_tokens = sum(tok_vals)
            tok_colors = [model_colors.get(m, "#f39c12") for m in tok_labels]
            _, texts, autotexts = ax.pie(
                tok_vals, labels=tok_labels, autopct="%1.1f%%",
                colors=tok_colors, startangle=140, pctdistance=0.75,
            )
            for t in texts:
                t.set_fontsize(10)
            for at in autotexts:
                at.set_fontsize(9)
            ax.set_title(
                f"모델별 사용 토큰 비율 (총 {total_model_tokens:,} 토큰, 실제 API 집계)",
                fontsize=11,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chart_path = run_dir / "pipeline_report.png"
    fig.savefig(str(chart_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _total_cost_line(cost_tracker: dict[str, Any] | None) -> str:
    tracker = cost_tracker or {}
    keys = [
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
    ]
    labels = ["가디언", "시뮬", "초기챕터", "AUTO챕터", "매니저", "AUTO리뷰", "코드리뷰", "재점검", "최종리뷰", "피드백"]
    total = sum(float(tracker.get(k, 0.0)) for k in keys)
    parts = " + ".join(
        f"{l} {float(tracker.get(k, 0.0)):.4f}"
        for l, k in zip(labels, keys)
        if float(tracker.get(k, 0.0)) > 0.00001
    )
    return f"💰 LLM 비용(Codex CLI 제외): ${total:.4f} ({parts})"


def _record_budget_usage(
    cost_tracker: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    budget: dict[str, Any] | None,
    *,
    cost_key: str | None = None,
) -> None:
    if not isinstance(budget, dict):
        return
    if cost_tracker is not None and cost_key:
        cost_tracker[cost_key] = float(cost_tracker.get(cost_key, 0.0)) + float(budget.get("spent_usd", 0.0) or 0.0)
    _accumulate_usage_totals(metrics, budget)


def _build_final_usage_lines(pipeline_start: float, cost_tracker: dict[str, Any]) -> tuple[str, str, str]:
    total_elapsed = time.monotonic() - pipeline_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)
    prompt_tok = int(cost_tracker.get("prompt_tokens", 0))
    comp_tok = int(cost_tracker.get("completion_tokens", 0))
    token_line = f"🪙 토큰: 입력 {prompt_tok:,} + 출력 {comp_tok:,} = 총 {prompt_tok + comp_tok:,}"
    elapsed_line = f"⏱️ 총 소요 시간: {total_min}분 {total_sec:02d}초"
    return elapsed_line, token_line, _total_cost_line(cost_tracker)


async def _notify_stop_usage_summary(
    notify: NotifyFn,
    pipeline_start: float,
    cost_tracker: dict[str, Any],
    *,
    step_label: str,
) -> None:
    if not notify:
        return
    _elapsed_line, _token_line, _cost_line = _build_final_usage_lines(pipeline_start, cost_tracker)
    await notify(
        f"{DAILY_TAG}[STOP] 🛑 사용자 요청으로 파이프라인을 중단했습니다. (단계: {step_label})\n"
        f"{_elapsed_line}\n"
        f"{_token_line}\n"
        f"{_cost_line}"
    )


def _accumulate_usage_totals(metrics: dict[str, Any] | None, budget: dict[str, Any]) -> None:
    if metrics is None:
        return
    metrics["prompt_tokens"] = int(metrics.get("prompt_tokens", 0)) + int(budget.get("prompt_tokens", 0) or 0)
    metrics["completion_tokens"] = int(metrics.get("completion_tokens", 0)) + int(budget.get("completion_tokens", 0) or 0)
    metrics["total_tokens"] = int(metrics.get("total_tokens", 0)) + int(budget.get("total_tokens", 0) or 0)
    breakdown = budget.get("breakdown", [])
    if not isinstance(breakdown, list):
        return
    model_token_totals = metrics.get("model_token_totals")
    if not isinstance(model_token_totals, dict):
        model_token_totals = {}
        metrics["model_token_totals"] = model_token_totals
    for row in breakdown:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model", "") or "").strip()
        if not model:
            continue
        total_tokens = row.get("total_tokens")
        if total_tokens is None:
            total_tokens = int(row.get("prompt_tokens", 0) or 0) + int(row.get("completion_tokens", 0) or 0)
        try:
            token_value = int(total_tokens or 0)
        except (TypeError, ValueError):
            token_value = 0
        if token_value <= 0:
            continue
        model_token_totals[model] = int(model_token_totals.get(model, 0)) + token_value


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


def _episode_number_from_key(episode_key: str | None) -> int | None:
    if not episode_key:
        return None
    match = re.match(r"^ep(\d+)(?:_|$)", str(episode_key).strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _episode_sort_key(episode_key: str) -> tuple[int, str]:
    episode_num = _episode_number_from_key(episode_key)
    if episode_num is None:
        return (10**9, str(episode_key))
    return (episode_num, str(episode_key))


def _story_state_keys_at_or_after(state: dict[str, Any], min_episode_number: int) -> list[str]:
    episode_summaries = state.get("episode_summaries", {})
    if not isinstance(episode_summaries, dict):
        return []
    found = [
        str(key)
        for key in episode_summaries.keys()
        if (_episode_number_from_key(str(key)) or -1) >= min_episode_number
    ]
    return sorted(found, key=_episode_sort_key)


def _reset_story_state_from_episode(
    story_state_path: Path,
    episode_key: str,
    backup_dir: Path,
) -> dict[str, Any]:
    target_episode_number = _episode_number_from_key(episode_key)
    state = _load_json(story_state_path)
    if target_episode_number is None or not isinstance(state, dict):
        return {
            "changed": False,
            "archive_path": None,
            "removed_episode_keys": [],
            "remaining_episode_keys": [],
            "last_completed_episode": None,
        }

    episode_summaries = state.get("episode_summaries", {})
    if not isinstance(episode_summaries, dict):
        episode_summaries = {}

    removed_episode_keys = _story_state_keys_at_or_after(state, target_episode_number)
    removed_summaries = {
        key: episode_summaries[key]
        for key in removed_episode_keys
        if key in episode_summaries
    }
    previous_last_completed = state.get("last_completed_episode")
    previous_last_num = _episode_number_from_key(str(previous_last_completed or ""))

    should_reset = bool(removed_episode_keys) or (
        previous_last_num is not None and previous_last_num >= target_episode_number
    )
    if not should_reset:
        return {
            "changed": False,
            "archive_path": None,
            "removed_episode_keys": [],
            "remaining_episode_keys": sorted(episode_summaries.keys(), key=_episode_sort_key),
            "last_completed_episode": previous_last_completed,
        }

    backup_dir.mkdir(parents=True, exist_ok=True)
    archive_path = backup_dir / "story_state_before_reset.json"
    archive_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if removed_summaries:
        (backup_dir / "story_state_removed_entries.json").write_text(
            json.dumps(removed_summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    remaining_summaries = {
        key: value
        for key, value in episode_summaries.items()
        if key not in removed_summaries
    }
    remaining_episode_keys = sorted(remaining_summaries.keys(), key=_episode_sort_key)

    state["episode_summaries"] = remaining_summaries
    state["last_completed_episode"] = remaining_episode_keys[-1] if remaining_episode_keys else None
    state["character_states"] = {}
    state["active_clues"] = {}
    state["arc_position"] = {}
    _save_json(story_state_path, state)

    return {
        "changed": True,
        "archive_path": str(archive_path),
        "removed_episode_keys": removed_episode_keys,
        "remaining_episode_keys": remaining_episode_keys,
        "last_completed_episode": state["last_completed_episode"],
    }


def _prepare_episode_restart_state(
    episode_key: str,
    *,
    story_state_path: Path = STORY_STATE_PATH,
) -> dict[str, Any]:
    backup_dir = EPISODE_ARCHIVE_DIR / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{episode_key}_reset"
    )
    db_result = story_db.archive_and_purge_episodes_from(episode_key, backup_dir)
    state_result = _reset_story_state_from_episode(story_state_path, episode_key, backup_dir)

    changed = bool(db_result.get("episode_ids")) or bool(state_result.get("changed"))
    if not changed and backup_dir.exists():
        try:
            backup_dir.rmdir()
        except OSError:
            pass

    return {
        "changed": changed,
        "backup_dir": str(backup_dir) if changed else None,
        "db": db_result,
        "story_state": state_result,
    }


def _get_cycle_number(episode_key: str) -> int:
    state = _load_json(STORY_STATE_PATH)
    ep_data = state.get("episode_summaries", {}).get(episode_key, {})
    return ep_data.get("cycle_count", 0) + 1


def _increment_cycle(episode_key: str) -> None:
    state = _load_json(STORY_STATE_PATH)
    ep_data = state.setdefault("episode_summaries", {}).setdefault(episode_key, {})
    ep_data["cycle_count"] = ep_data.get("cycle_count", 0) + 1
    _save_json(STORY_STATE_PATH, state)


# ── subprocess 출력 → Discord 필터 유틸리티 ──────────────────────────────────
# Python logging 형식: "HH:MM:SS [LEVEL] module.name: message"
_PY_LOG_RE = re.compile(
    r'^\d{2}:\d{2}:\d{2} \[(?:INFO|DEBUG|WARNING|ERROR|CRITICAL)\] \S+:\s*'
)


def _strip_python_log(line: str) -> tuple[str, str]:
    """(level, clean_message) — Python logging prefix를 제거한다."""
    m = re.search(r'\[(INFO|DEBUG|WARNING|ERROR|CRITICAL)\]', line)
    level = m.group(1) if m else "INFO"
    msg = _PY_LOG_RE.sub('', line).strip()
    return level, msg


def _friendly_sim_line(line: str) -> str | None:
    """시뮬레이터 로그 → 사람이 읽을 수 있는 한국어. None 이면 출력 생략."""
    level, msg = _strip_python_log(line)
    if not msg:
        return None
    if level in ("ERROR", "CRITICAL"):
        return f"❌ {msg}"
    if level == "WARNING":
        return f"⚠️ {msg}"
    m = re.search(r'Loaded episode:\s*(\S+).*?max_turns=(\d+).*?clues=(\d+)', msg)
    if m:
        return f"에피소드 **{m.group(1)}** ({m.group(2)}턴 / 단서 {m.group(3)}개)"
    return None  # 나머지는 모두 생략


def _friendly_chapter_line(line: str) -> str | None:
    """챕터 생성 로그 → 사람이 읽을 수 있는 한국어. None 이면 출력 생략."""
    level, msg = _strip_python_log(line)
    if not msg:
        return None
    if level in ("ERROR", "CRITICAL"):
        return f"❌ {msg}"
    if level == "WARNING":
        return f"⚠️ {msg}"
    m = re.search(r'Target words:\s*(\d+).*?Target scenes:\s*(\d+).*?(\d+)\s*words/scene', msg)
    if m:
        return f"목표: {int(m.group(1)):,}단어 / {m.group(2)}장면 (장면당 ~{m.group(3)}단어)"
    m = re.search(r'adjusted target scenes:\s*(\d+)\s*->\s*(\d+)', msg, re.IGNORECASE)
    if m:
        return f"리더 피드백 반영: {m.group(1)}장면 → {m.group(2)}장면"
    if re.search(r'Scene data\s*[→>]', msg):
        return "씬 분석 완료"
    return None  # 나머지는 모두 생략


_FIXER_SHELL_RE = re.compile(r'^(/bin/|/usr/|rg |sed |grep |cat |python)', re.IGNORECASE)
_FIXER_CODE_RE = re.compile(r'^\s*(def |class |import |from |if |for |return |#|async )')
_FIXER_FILELINE_RE = re.compile(r'^\S+\.py:\d+:')
_FIXER_PYFILE_RE = re.compile(r'\b[\w/.-]+\.py\b')


def _friendly_fixer_line(line: str) -> str | None:
    """Codex Fixer 출력 → 사람이 읽을 수 있는 한국어. None 이면 출력 생략."""
    s = line.strip()
    if not s:
        return None
    if _FIXER_SHELL_RE.match(s):
        return None
    if _FIXER_CODE_RE.match(s):
        return None
    if _FIXER_FILELINE_RE.match(s):
        return None
    # 파일 수정 언급 (영어)
    if re.search(r'(edit|write|modif|updat|creat|patch)', s, re.IGNORECASE):
        fm = _FIXER_PYFILE_RE.search(s)
        if fm:
            return f"수정 중: `{fm.group()}`"
    # 한국어 텍스트 (Codex 분석/설명)
    if re.search(r'[\uAC00-\uD7A3]', s):
        return s[:100] + ('…' if len(s) > 100 else '')
    return None


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
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
        )
    except FileNotFoundError as exc:
        missing = cmd[0] if cmd else "unknown"
        msg = f"FileNotFoundError: {exc} (missing executable: {missing})"
        return 127, msg
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
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
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
아래 자료를 읽고 현재 작성 대상 에피소드를 위한 생성용 브리핑을 작성하세요.

## 전체 스토리 컨텍스트 (story_context.yaml)
```yaml
{context["story_context"]}
```

## 인접 에피소드 YAML (현재 화 ±3화)
{ep_blocks}

## 규칙 기반 자동 검수 결과
{rule_report}

---
목표:
- 뒤 단계의 chapter generation / director / prose_generator가 바로 프롬프트에 넣어 쓸 수 있는 짧고 실행 가능한 브리핑이어야 한다.
- 감상문/평가문처럼 길게 해설하지 말고, "무엇을 지키고 무엇을 피해야 하는지"를 작가 지시문처럼 써라.
- 앞뒤 화 연결, 캐릭터 arc, 복선/클루, gates를 반영하되 추상적 칭찬은 최소화하라.
- 다음 화 정보는 현재 화의 직접 내용처럼 요약하지 마라. 다음 화 YAML은 오직 현재 화에서 금지해야 할 조기 공개, 남겨야 할 미해결 긴장, 모순 방지 체크를 위해서만 사용하라.
- 특히 "ep02에서는 ...", "다음 화에서 ..."처럼 미래 사건을 현재 화 브리핑의 핵심 연결 bullet에 직접 쓰지 마라.

출력 형식:
다음 헤더를 반드시 그대로 사용하고, 각 항목은 짧은 bullet 위주로 작성하라.

## 이번 화 핵심 연결
- 이전 화와의 연결 2-4개
- 이번 화에서 새로 강화되는 긴장/전환 1-3개
- 다음 화 사건 요약 금지

## 캐릭터 진행 가이드
- 주인공 감정/판단/행동 축
- 주요 조연 1-3명의 역할과 압박 방식

## 클루와 복선 운용
- 이번 화에서 유지/강조할 클루 2-5개
- 이번 화에서 한 번만 짚고 과설명하면 안 되는 요소 1-3개

## 게이트와 금지 사항
- 반드시 지켜야 할 gate / timeline / 공개 범위 제약 2-5개
- 이번 화에서 하면 안 되는 조기 공개, 과잉 설명, 미래 정보 점프 1-4개

## 장면 운영 지시
- 이번 화에서 대비되어야 할 감정/압박 흐름 2-4개
- 반복되기 쉬운 설명이나 정서를 어떻게 처리할지 1-3개
- 이번 화 끝에 남겨야 할 미해결 긴장 1-3개

## 짧은 실행 메모
- prose_generator/director가 바로 쓸 수 있는 3-6개의 짧은 imperative 문장

작성 규칙:
- 각 bullet은 한두 문장 이내로 짧게.
- "자연스럽다", "잘 연결된다", "도움이 된다" 같은 평가형 문장은 피하고, 대신 유지/강조/금지/이월 같은 동사로 써라.
- 인물명, 조직명, 클루명은 실제 YAML에 있는 표현을 우선 사용하라.
- 위반/리스크가 없다면 "명시적 위반 없음"이라고 짧게 적고 끝내라.
- 미래 화에서만 발생하는 사건, 장소, 제안, 폭로는 현재 화 브리핑의 사실처럼 쓰지 마라. 필요하면 "이번 화에서 조기 공개하지 말 것", "다음 화로 이월할 것"처럼 금지 규칙으로만 변환하라.
- 결과물에는 현재 화보다 뒤의 에피소드 번호를 직접 인용하지 마라.
- 700자~1400자 안팎으로 유지하라."""


async def step_guardian(
    episode_key: str,
    run_dir: Path,
    cycle: int,
    notify: NotifyFn,
    upload: UploadFn,
    set_status: StatusFn,
    stop_event: asyncio.Event | None,
    review_tier: str = "premium",
    cost_tracker: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
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
            use_premium=_use_premium_review_tier(review_tier),
            purpose="guardian_context_analysis",
            max_tokens=1500,
        )
        guardian_budget = llm.budget_summary()
        _record_budget_usage(cost_tracker, metrics, guardian_budget, cost_key="guardian")

        briefing_path = run_dir / "guardian_briefing.txt"
        briefing_path.write_text(gpt_report, encoding="utf-8")

        if notify:
            await notify(f"{DAILY_TAG}[GUARDIAN] 🧭 GPT 생성용 브리핑:\n{gpt_report}")
            await notify(f"{DAILY_TAG}[GUARDIAN] 💸 {_format_budget_line('Guardian 분석 비용', guardian_budget)}")

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
    cost_tracker: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
    auto_cycle_index: int | None = None,
    auto_max_cycles: int | None = None,
    guardian_briefing_path: Path | None = None,
    reset_emotions: bool = False,
) -> bool:
    if stop_event and stop_event.is_set():
        return False

    if set_status:
        set_status("2/4 시뮬레이션 준비 중...")
    if notify:
        progress_label = _format_auto_progress_label(auto_cycle_index, auto_max_cycles)
        await notify(f"{DAILY_TAG}[SIM] ⚙️ 시뮬레이션 시작{progress_label}...")

    ep_file = resolve_episode_file(episode_key)

    cmd = [
        _resolve_python_cmd(), "-u", "simulate.py",
        "--episode", str(ep_file),
        "--characters", "config/characters.yaml",
        "--world", "config/world_facts.yaml",
        "--storyline", "config/storyline.yaml",
        "--output", str(run_dir),
        "--budget", str(budget * 0.5),
    ]
    if guardian_briefing_path and guardian_briefing_path.exists():
        cmd += ["--guardian-briefing", str(guardian_briefing_path)]
    if reset_emotions:
        cmd += ["--reset-emotions"]

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
        # ── 디렉터 이벤트 감지 ──────────────────────────────────────────────
        _, msg = _strip_python_log(line)
        if "[Director]" in msg:
            if "clue_injection" in msg:
                clue_m = re.search(r'Injecting clue:\s*(\S+)', msg)
                clue_id = clue_m.group(1).rstrip("}\"'") if clue_m else "?"
                if notify:
                    await notify(f"{DAILY_TAG}[SIM] 🔍 {current_turn_label} — 디렉터: 단서 주입 `{clue_id}`")
            elif "invariant_violation" in msg:
                char_m = re.search(r'for ([^:]+):', msg)
                char_name = char_m.group(1).strip() if char_m else "?"
                if notify:
                    await notify(f"{DAILY_TAG}[SIM] ⚠️ {current_turn_label} — 디렉터: 캐릭터 일관성 수정 ({char_name})")
            elif "knowledge_leak" in msg:
                if notify:
                    await notify(f"{DAILY_TAG}[SIM] ⚠️ {current_turn_label} — 디렉터: 정보 누설 차단")
            return
        # ── 기타 친화적 메시지 ──────────────────────────────────────────────
        friendly = _friendly_sim_line(line)
        if friendly and notify:
            await notify(f"{DAILY_TAG}[SIM] ⚙️ {friendly}")

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
        # 오류 로그 저장 (Codex 자동 진단용)
        try:
            (run_dir / "simulator_error.log").write_text(output or "", encoding="utf-8")
        except Exception:
            pass
        return False

    if set_status:
        set_status("2/4 시뮬레이션 완료")
    budget = _empty_budget_summary()
    sim_files = sorted(run_dir.glob("*_simulation.json"), key=lambda p: p.stat().st_mtime)
    if sim_files:
        try:
            sim_data = json.loads(sim_files[-1].read_text(encoding="utf-8"))
            budget = sim_data.get("budget", budget) or budget
        except Exception:
            pass
    _record_budget_usage(cost_tracker, metrics, budget, cost_key="simulation")
    if notify:
        await notify(f"{DAILY_TAG}[SIM] ✅ 시뮬레이션 완료")
        await notify(f"{DAILY_TAG}[SIM] 💸 {_format_budget_line('시뮬레이션 비용', budget)}")
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
    cost_tracker: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
    auto_cycle_index: int | None = None,
    auto_max_cycles: int | None = None,
    upload_version_label: str | None = None,
    precomputed_scenes_path: Path | None = None,
    guardian_briefing_path: Path | None = None,
    review_tier: str = "premium",
) -> Path | None:
    if stop_event and stop_event.is_set():
        return None

    if set_status:
        set_status("3/4 챕터 생성 중...")
    if notify:
        progress_label = _format_auto_progress_label(auto_cycle_index, auto_max_cycles)
        await notify(f"{DAILY_TAG}[CHAPTER] 📖 챕터 생성 중{progress_label}...")

    ep_file = resolve_episode_file(episode_key)
    episode_id = ep_file.stem

    prev_review = None
    if cycle > 1:
        candidates = sorted(run_dir.parent.glob(f"**/{episode_id}_*review*.txt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            prev_review = candidates[-1]

    cmd = [
        _resolve_python_cmd(), "-u", "generate_chapter.py",
        "--episode", episode_id,
        "--episode-config", str(ep_file),
        "--protagonist", protagonist,
        "--output", str(run_dir),
        "--words", str(target_words),
        "--budget", str(budget * 0.5),
    ]
    if _use_mini_review_tier(review_tier):
        cmd += ["--model", "gpt-4o-mini", "--premium", "gpt-4.1-mini"]
    if prev_review:
        cmd += ["--reader-review-md", str(prev_review)]
    if precomputed_scenes_path and precomputed_scenes_path.exists():
        cmd += ["--precomputed-scenes", str(precomputed_scenes_path)]
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
        friendly = _friendly_chapter_line(line)
        if friendly and notify:
            prefix = current_scene_label or current_stage_label
            emoji = "📝" if current_scene_label else "🧩"
            await notify(f"{DAILY_TAG}[CHAPTER] {emoji} {prefix} | {friendly}")

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
        # 오류 로그 저장 (Codex 자동 진단용)
        try:
            (run_dir / "chapter_gen_error.log").write_text(output or "", encoding="utf-8")
        except Exception:
            pass
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
    budget = _load_chapter_budget_meta(run_dir, episode_id, chapter_out) or _parse_budget_from_output(output)
    _record_budget_usage(cost_tracker, metrics, budget, cost_key="chapter")
    upload_path = chapter_out
    if upload_version_label:
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", upload_version_label).strip("_")
        if safe_label:
            versioned_path = chapter_out.with_name(f"{chapter_out.stem}_{safe_label}{chapter_out.suffix}")
            shutil.copy2(chapter_out, versioned_path)
            upload_path = versioned_path
    # 챕터 본문은 txt 파일로 전송
    if upload:
        note = f"📖 {ep_file.stem} — {word_count}단어"
        if upload_version_label:
            note = f"{note} ({upload_version_label})"
        await upload(upload_path, note)
        if notify:
            await notify(f"{DAILY_TAG}[CHAPTER] 💸 {_format_budget_line('챕터 생성 비용', budget)}")
    elif notify:
        await notify(
            f"{DAILY_TAG}[CHAPTER] ✅ 챕터 완성 ({word_count}단어) — `{upload_path.name}`\n"
            f"{_format_budget_line('챕터 생성 비용', budget)}"
        )

    return chapter_out


# ── Step 3b: Inline Optimizer ─────────────────────────────────────────────────

async def step_inline_optimize(
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
    review_tier: str = "premium",
    cost_tracker: dict | None = None,
) -> Path | None:
    if stop_event and stop_event.is_set():
        return None

    if set_status:
        set_status("3/4 챕터 최적화 중...")

    from tools.inline_optimizer import (
        run_inline_optimize,
        log_policy_score,
        should_trigger_full_optuna,
        trigger_full_optuna_background,
        POLICY_SCORE_LOG,
    )
    _QUALITY_REVIEW_LATEST = REPO_ROOT / "data" / "quality_review_latest.json"
    from src.novel_writer.rl_policy import load_policy
    from src.novel_writer.config_loader import load_episode, load_characters

    ep_file = resolve_episode_file(episode_key)

    try:
        episode_config = load_episode(str(ep_file))
    except Exception as exc:
        logger.warning("[INLINE_OPT] episode config load failed: %s", exc)
        return None

    # Use the YAML id field (matches DB), not the filename stem
    episode_id = str(episode_config.get("id") or ep_file.stem).strip()

    try:
        character_profiles = load_characters(str(REPO_ROOT / "config" / "characters.yaml"))
    except Exception as exc:
        logger.warning("[INLINE_OPT] character profiles load failed: %s", exc)
        character_profiles = None

    # Find latest reader feedback json in run_dir or parent
    reader_feedback: dict | None = None
    try:
        candidates = sorted(
            list(run_dir.glob("**/*review*.json")) + list(run_dir.parent.glob(f"**/{episode_id}*review*.json")),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            reader_feedback = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        pass

    guardian_briefing: str | None = None
    if guardian_briefing_path and guardian_briefing_path.exists():
        try:
            guardian_briefing = guardian_briefing_path.read_text(encoding="utf-8")
        except Exception:
            pass

    base_policy = load_policy()

    # Load quality review scores from previous episode to focus optimization
    quality_focus: dict | None = None
    try:
        if _QUALITY_REVIEW_LATEST.exists():
            _qr_data = json.loads(_QUALITY_REVIEW_LATEST.read_text(encoding="utf-8"))
            quality_focus = _qr_data.get("scores")
            if quality_focus:
                logger.info("[INLINE_OPT] quality_focus loaded: %s", quality_focus)
    except Exception as exc:
        logger.warning("[INLINE_OPT] quality_review_latest load failed: %s", exc)

    # Cost strategy:
    # - 5 inline trials: gpt-4o-mini only
    # - final single upgrade pass: gpt-4.1-mini
    base_model = "gpt-4o-mini"
    premium_model = "gpt-4o-mini"
    final_upgrade_model = "gpt-4.1-mini"

    if notify:
        _focus_summary = (
            ", ".join(f"{k}={v}" for k, v in quality_focus.items()) if quality_focus else "없음"
        )
        await notify(f"{DAILY_TAG}[OPTIMIZE] 🔬 인라인 최적화 시작 (5 trials) | 품질 포커스: {_focus_summary}")

    try:
        best_path, best_params, best_score, all_scores = await run_inline_optimize(
            episode_id=episode_id,
            episode_config=episode_config,
            run_dir=run_dir,
            protagonist_id=protagonist,
            protagonist_name=" ".join(w.capitalize() for w in protagonist.split("_")),
            target_words=target_words,
            budget=budget,
            character_profiles=character_profiles,
            reader_feedback=reader_feedback,
            guardian_briefing=guardian_briefing,
            base_policy=base_policy,
            base_model=base_model,
            premium_model=premium_model,
            notify_fn=notify,
            quality_focus=quality_focus,
            final_upgrade_model=final_upgrade_model,
        )
    except Exception as exc:
        logger.warning("[INLINE_OPT] run_inline_optimize failed: %s", exc)
        if notify:
            await notify(f"{DAILY_TAG}[OPTIMIZE] ❌ 인라인 최적화 실패: {exc}")
        return None

    if best_path is None or not best_path.exists():
        logger.warning("[INLINE_OPT] best chapter path invalid: %s", best_path)
        return None

    # Collect all trial scores from the optimizer run (best_score only in this path; log it)
    try:
        log_policy_score(
            episode_id=episode_id,
            best_params=best_params,
            best_score=best_score,
            all_trial_scores=all_scores,
            log_path=POLICY_SCORE_LOG,
        )
    except Exception as exc:
        logger.warning("[INLINE_OPT] log_policy_score failed: %s", exc)

    try:
        from tools.inline_optimizer import update_rl_policy
        update_rl_policy(best_params, best_score, episode_id)
        logger.info("[INLINE_OPT] rl_policy.json updated with best params")
    except Exception as exc:
        logger.warning("[INLINE_OPT] rl_policy update failed: %s", exc)

    try:
        if should_trigger_full_optuna(POLICY_SCORE_LOG):
            trigger_full_optuna_background(REPO_ROOT, trials=30)
            if notify:
                await notify(
                    f"{DAILY_TAG}[OPTIMIZE] 🔬 5화 누적 — 전체 Optuna 재튜닝 백그라운드 시작"
                )
    except Exception as exc:
        logger.warning("[INLINE_OPT] full optuna trigger failed: %s", exc)

    if set_status:
        word_count = len(best_path.read_text(encoding="utf-8", errors="replace").split())
        set_status(f"3/4 챕터 최적화 완성 ({word_count}단어)")

    if upload:
        try:
            word_count = len(best_path.read_text(encoding="utf-8", errors="replace").split())
            await upload(best_path, f"📖 {episode_id} — {word_count}단어 (inline opt best={best_score:.2f})")
        except Exception as exc:
            logger.warning("[INLINE_OPT] upload failed: %s", exc)
    if notify:
        await notify(
            f"{DAILY_TAG}[OPTIMIZE] ✅ 최적화 챕터 완성 | best_score={best_score:.2f} | `{best_path.name}`"
        )

    return best_path


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
    review_tier: str = "premium",
    cost_tracker: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
) -> str | None:
    if stop_event and stop_event.is_set():
        return None

    if set_status:
        set_status("4/4 품질 검수 중...")
    if notify:
        await notify(f"{DAILY_TAG}[REVIEW] 🔍 품질 자동 검수 중...")

    try:
        scorecard, auto_results, review_meta = await asyncio.to_thread(
            run_quality_review, episode_key, chapter_path, run_dir, False, review_tier,
        )
    except Exception as exc:
        if notify:
            await notify(f"{DAILY_TAG}[REVIEW] ❌ 검수 실패: {type(exc).__name__}: {exc}")
        return None

    scorecard_path = run_dir / "scorecard.txt"
    scorecard_path.write_text(scorecard, encoding="utf-8")

    if set_status:
        set_status("피드백 대기 중...")
    _record_budget_usage(cost_tracker, metrics, review_meta, cost_key="final_review")
    if notify:
        await notify(
            f"{DAILY_TAG}[REVIEW] ✅ 자동 검수 완료\n\n{scorecard}\n\n"
            f"{_format_budget_line('최종 리뷰 비용', review_meta)}"
        )

    return scorecard


# ── Auto-improve loop (AI Reviewer → Codex Fixer → Chapter regen) ─────────────

def _load_story_context_for_review() -> str:
    """story_context.yaml 내용을 문자열로 반환. 없거나 읽기 실패 시 빈 문자열 + 로그."""
    p = REPO_ROOT / "config" / "story_context.yaml"
    if not p.exists():
        logger.warning("[REVIEW] story_context.yaml 없음 — 스토리 컨텍스트 없이 리뷰 진행")
        return ""
    try:
        return p.read_text(encoding="utf-8")[:3000]
    except Exception as exc:
        logger.warning("[REVIEW] story_context.yaml 읽기 실패 (%s) — 스토리 컨텍스트 없이 리뷰 진행", exc)
        return ""


def _build_ai_reviewer_prompt(chapter_text: str, story_context: str = "") -> str:
    context_block = (
        f"\n## 스토리 컨텍스트 (캐릭터/세계관/인과관계 판단 참고용)\n{story_context}\n"
        if story_context else ""
    )
    return (
        "당신은 한국어 테크노스릴러 소설 챕터를 평가하는 편집자입니다.\n"
        "독자 관점(몰입감/재미)과 서사 구조 관점(인과성/캐릭터/씬 기능) 모두 평가하라.\n"
        f"{context_block}"
        "아래 JSON 형식으로만 응답하라 (다른 텍스트 없이):\n"
        "{\n"
        '  "thrill_score_10": <0~10 정수, 긴장감/몰입감>,\n'
        '  "style_score_10": <0~10 정수, 문체/가독성>,\n'
        '  "causality_score_10": <0~10 정수, 인과성/개연성 — 사건이 논리적으로 이어지는가>,\n'
        '  "character_score_10": <0~10 정수, 캐릭터 일관성/동기 — 캐릭터가 설정대로 행동하는가>,\n'
        '  "scene_function_score_10": <0~10 정수, 장면 기능성/서사 진행도 — 씬이 스토리를 앞으로 나아가게 하는가>,\n'
        '  "one_line_verdict": "<한줄평>",\n'
        '  "what_felt_good": ["<좋았던 점1>", "<좋았던 점2>", "<좋았던 점3>"],\n'
        '  "what_felt_boring_or_hard": ["<지루/어려운 점1>", "<점2>", "<점3>"],\n'
        '  "style_tips": ["<개선팁1>", "<팁2>", "<팁3>"],\n'
        '  "reader_comment": "<4~6문장 독자 코멘트>"\n'
        "}\n"
        "규칙: 솔직하고 구체적으로. 각 리스트 ≥ 3개 항목. 한국어만 사용.\n\n"
        f"챕터 본문:\n{chapter_text}"
    )


async def _run_codex_review(
    chapter_path: Path,
    run_dir: Path,
    fixer_cycle: int,
    set_process: ProcessFn = None,
) -> dict | None:
    """
    Codex CLI를 사용해 챕터를 리뷰.
    OpenAI API 호출 없이 Codex가 직접 챕터를 읽고 JSON 리뷰 반환.
    실패 시 None 반환.
    """
    review_out = run_dir / f"codex_review_cycle{fixer_cycle}.json"
    story_context_path = REPO_ROOT / "config" / "story_context.yaml"
    story_ctx_block = ""
    if story_context_path.exists():
        story_ctx_block = f"\n참고용 스토리 컨텍스트 파일: {story_context_path}\n(파일을 읽어 캐릭터/세계관/인과관계 판단에 활용하라)\n"

    prompt = (
        f"다음 경로의 한국어 소설 챕터를 읽고 편집자+독자 관점에서 리뷰하라.\n"
        f"파일 경로: {chapter_path}\n"
        f"{story_ctx_block}\n"
        "평가 항목: 긴장감/몰입감, 문체/가독성, 인과성/개연성, 캐릭터 일관성/동기, 장면 기능성/서사 진행도\n"
        "반드시 아래 JSON 형식으로만 응답하라 (다른 텍스트 없이):\n"
        "{\n"
        '  "thrill_score_10": <0~10 정수, 긴장감/몰입감>,\n'
        '  "style_score_10": <0~10 정수, 문체/가독성>,\n'
        '  "causality_score_10": <0~10 정수, 인과성/개연성>,\n'
        '  "character_score_10": <0~10 정수, 캐릭터 일관성/동기>,\n'
        '  "scene_function_score_10": <0~10 정수, 장면 기능성/서사 진행도>,\n'
        '  "one_line_verdict": "<한줄평>",\n'
        '  "what_felt_good": ["<좋았던 점1>", "<좋았던 점2>", "<좋았던 점3>"],\n'
        '  "what_felt_boring_or_hard": ["<지루/어려운 점1>", "<점2>", "<점3>"],\n'
        '  "style_tips": ["<개선팁1>", "<팁2>", "<팁3>"],\n'
        '  "reader_comment": "<4~6문장 독자 코멘트>"\n'
        "}\n"
        f"결과를 다음 경로에 저장하라: {review_out}"
    )

    cmd = _resolve_codex_exec_cmd()
    if not cmd:
        return None
    cmd += [
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        prompt,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        return None
    if set_process:
        set_process("codex_review", proc.pid, " ".join(cmd))
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=600)
    except asyncio.TimeoutError:
        proc.terminate()
        await proc.wait()
        return None
    finally:
        if set_process:
            set_process(None, None, None)

    if proc.returncode != 0:
        return None

    # Codex가 review_out 파일에 저장했으면 읽기, 아니면 stdout에서 추출
    if review_out.exists():
        try:
            raw = review_out.read_text(encoding="utf-8")
            cleaned = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")
            return json.loads(cleaned)
        except Exception:
            pass

    # fallback: stdout에서 JSON 블록 추출
    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    json_match = re.search(r"\{[\s\S]*\"thrill_score_10\"[\s\S]*\}", output)
    if json_match:
        try:
            result = json.loads(json_match.group())
            review_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception:
            pass

    return None


async def _run_codex_error_fixer(
    error_output: str,
    failed_step: str,
    run_dir: Path,
    notify: NotifyFn = None,
    set_process: ProcessFn = None,
) -> bool:
    """
    파이프라인 단계 실패 시 Codex를 호출해 오류를 진단/수정.
    성공(수정 완료 가능성 있음) 시 True, 실패 시 False 반환.
    """
    error_preview = error_output[-2000:] if len(error_output) > 2000 else error_output
    diagnosis_out = run_dir / f"codex_error_diagnosis_{failed_step}.txt"

    prompt = (
        f"아래는 '{failed_step}' 단계 실행 중 발생한 오류 로그다.\n\n"
        "```\n"
        f"{error_preview}\n"
        "```\n\n"
        "다음 작업을 수행하라:\n"
        "1. 오류의 근본 원인을 분석하라.\n"
        "2. 관련 소스 파일을 읽고 버그를 수정하라.\n"
        "3. Python 문법 오류나 KeyError, AttributeError 등 런타임 오류를 직접 수정하라.\n"
        "4. 수정 완료 후 아래 경로에 진단 요약을 저장하라:\n"
        f"   {diagnosis_out}\n\n"
        "수정 대상 파일:\n"
        "- src/novel_writer/prose_generator.py\n"
        "- src/novel_writer/scene_distiller.py\n"
        "- src/novel_writer/director.py\n"
        "- generate_chapter.py\n"
        "- generate_simulation.py\n"
        "오류 원인과 무관한 파일은 수정하지 마라."
    )

    cmd = _resolve_codex_exec_cmd()
    if not cmd:
        if notify:
            await notify(f"{DAILY_TAG}[ERROR-FIX] ℹ️ Codex CLI 없음 → GPT API로 오류 진단")
        ok, _ = await _run_gpt_fixer(
            prompt, run_dir, 0, notify=notify, model="gpt-4.1-mini",
        )
        return ok
    cmd += [
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        prompt,
    ]

    if notify:
        await notify(
            f"{DAILY_TAG}[ERROR-FIX] 🔧 `{failed_step}` 오류 감지 — Codex 자동 진단 시작..."
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        if notify:
            await notify(f"{DAILY_TAG}[ERROR-FIX] ❌ Codex 실행 파일을 찾지 못했습니다. 자동 진단을 건너뜁니다.")
        return False
    if set_process:
        set_process("codex_error_fixer", proc.pid, " ".join(cmd))

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=600)
    except asyncio.TimeoutError:
        proc.terminate()
        await proc.wait()
        if notify:
            await notify(f"{DAILY_TAG}[ERROR-FIX] ⏰ Codex 진단 타임아웃 (10분)")
        return False
    finally:
        if set_process:
            set_process(None, None, None)

    success = proc.returncode == 0
    if notify:
        if success:
            summary = ""
            if diagnosis_out.exists():
                try:
                    summary = diagnosis_out.read_text(encoding="utf-8")[:400]
                except Exception:
                    pass
            msg = f"{DAILY_TAG}[ERROR-FIX] ✅ Codex 진단/수정 완료"
            if summary:
                msg += f"\n```\n{summary}\n```"
            await notify(msg)
        else:
            out_preview = stdout.decode("utf-8", errors="replace")[-400:] if stdout else "(출력 없음)"
            await notify(
                f"{DAILY_TAG}[ERROR-FIX] ❌ Codex 진단 실패 (rc={proc.returncode}):\n```\n{out_preview}\n```"
            )
    return success


def _build_optuna_context() -> str:
    """Build a concise Optuna context report for Codex fixer prompt."""
    import json
    log_path = REPO_ROOT / "data" / "policy_score_log.jsonl"
    policy_path = REPO_ROOT / "data" / "rl_policy.json"

    lines = []
    if log_path.exists():
        try:
            entries = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if entries:
                recent = entries[-5:]  # last 5 episodes
                avg_score = sum(e.get("best_score", 0) for e in recent) / len(recent)
                best_entry = max(entries, key=lambda e: e.get("best_score", 0))
                lines.append(f"## Optuna 최적화 현황 (최근 {len(recent)}화)")
                lines.append(f"- 최근 평균 best_score: {avg_score:.3f}")
                lines.append(f"- 역대 최고: {best_entry.get('best_score', 0):.3f} (에피소드: {best_entry.get('episode_id', '?')})")
                # Show score trend
                scores = [e.get("best_score", 0) for e in recent]
                trend = "📈 상승" if len(scores) >= 2 and scores[-1] > scores[0] else "📉 정체/하락"
                lines.append(f"- 추세: {trend}")
                # Best params from last run
                if recent[-1].get("best_params"):
                    lines.append(f"- 최근 best_params: {json.dumps(recent[-1]['best_params'], ensure_ascii=False)}")
        except Exception:
            pass

    if policy_path.exists():
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            # Show which optuna studies ran
            for key in ["_optuna_prose_best", "_optuna_distiller_best", "_optuna_orchestrator_best", "_optuna_polisher_best"]:
                if key in policy:
                    study = policy[key]
                    lines.append(f"- {key}: score={study.get('score', '?')} (trial={study.get('trial', '?')})")
        except Exception:
            pass

    if not lines:
        return ""
    return "\n".join(lines)


def _build_codex_fixer_prompt(review_json: dict, manager_instructions: str = "") -> str:
    issues = review_json.get("what_felt_boring_or_hard", [])
    tips = review_json.get("style_tips", [])
    comment = review_json.get("reader_comment", "")
    thrill    = review_json.get("thrill_score_10", "?")
    style     = review_json.get("style_score_10", "?")
    causality = review_json.get("causality_score_10", "?")
    character = review_json.get("character_score_10", "?")
    scene_fn  = review_json.get("scene_function_score_10", "?")
    manager_block = (
        f"\n## 매니저 분석 지시사항 (우선 적용)\n{manager_instructions}\n"
        if manager_instructions else ""
    )
    optuna_ctx = _build_optuna_context()
    optuna_block = (
        f"## Optuna 파라미터 최적화 컨텍스트\n{optuna_ctx}\n\n"
        if optuna_ctx else ""
    )
    return (
        optuna_block
        + "## Codex의 역할 (중요)\n"
        "Optuna가 이미 숫자 파라미터(temperature, counts 등)를 최적화했습니다.\n"
        "Codex는 Optuna가 할 수 없는 작업에 집중하세요:\n\n"
        "**허용:**\n"
        "- prose_generator.py / scene_distiller.py 등의 LLM 프롬프트 텍스트 개선\n"
        "  (점수가 낮은 기준에 해당하는 system/user 프롬프트 수정)\n"
        "- 하드코딩된 값을 runtime_policy 키로 노출\n"
        "  (tools/inline_optimizer.py param space + data/rl_policy.json 기본값도 함께 추가)\n"
        "- 컴포넌트 간 인터페이스 버그 수정\n\n"
        "**금지:**\n"
        "- 숫자 파라미터 값 직접 변경 (Optuna 영역)\n"
        "- runtime_policy 키 추가 시 inline_optimizer.py와 rl_policy.json 동기화 없이 코드만 변경\n"
        "- 감정 유형 → 완성된 한국어 문장 직접 반환 매핑\n"
        "  (예: 불안→'숨을 짧게 들이쉬고…', 분노→'턱선을 굳히고…' 같은 고정 출력 절대 금지)\n"
        "  대신: 감정을 짧은 영어 힌트 태그나 행동 방향으로만 표현하고, 실제 문장은 LLM에 위임하라\n"
        "- _transition_replacement_catalog 또는 유사 고정 문구 리스트에 항목 추가/수정\n"
        "  (접속사 대체 리스트 확장 금지 — 새 stock phrase bank가 될 수 있음)\n"
        "- 특정 인물명(밀러, Miller, 모레노, Moreno 등)을 조건으로 하는 로직을 공용 파이프라인 파일에 추가\n"
        "  (인물별 특수 처리는 episode config / character policy에 격리할 것)\n"
        "- 동일한 스타일 보정 개념을 2개 이상의 파이프라인 단계에 동시 삽입\n"
        "  (각 단계 책임: director=리듬·압박 신호, scene_distiller=구조·압축, prose_generator=문장 표현, polisher=미세정리)\n"
        "  같은 규칙이 director와 prose_generator 양쪽에 들어가면 편향이 파이프라인 전체에 증폭됨\n\n"
        "**단계별 수정 책임 (이 경계를 넘지 말 것):**\n"
        "- director.py: 씬 진행 신호(압박·평탄·반복) 감지 + LLM에 방향 힌트만 전달\n"
        "- scene_distiller.py: 씬 압축·구조화 (문체 교정 로직 추가 금지)\n"
        "- prose_generator.py: 실제 문장 생성 프롬프트 (director 신호를 중복 재구현하지 말 것)\n"
        "- polisher: 미세 문체 정리 (위 단계와 동일 교정 중복 금지)\n\n"
        f"독자 AI 리뷰 결과: 긴장감={thrill}/10, 문체={style}/10, "
        f"인과성={causality}/10, 캐릭터={character}/10, 씬기능={scene_fn}/10\n\n"
        f"문제점:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
        f"개선 팁:\n" + "\n".join(f"- {t}" for t in tips) + "\n\n"
        f"리뷰 전문: {comment}\n"
        + manager_block +
        "\n위 독자 리뷰를 바탕으로, 소설 생성 코드의 품질을 개선하라.\n"
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
        "6. 확인 질문 없이 바로 수정하라\n"
        "7. 매니저가 지정한 시작 파일/함수/우선순위가 있으면 그 지점을 먼저 확인하라\n"
        "8. 첫 가설이 틀리면 한 번에 넓게 뒤지지 말고, 인접한 파일로 한 단계씩 범위를 넓혀라\n"
        "9. 수정 후에는 변경한 Python 파일에 대해 `python3 -m py_compile ...`로 빠르게 검증하라\n"
        "10. `scene_distiller.py`, `prose_generator.py`, `director.py`, `generate_chapter.py` 중 하나라도 수정했다면 `python3 -m unittest tests.test_reader_feedback_guards`도 실행하라\n"
        "11. 프롬프트/가이드 문구를 추가할 때는 길이 예산을 의식하라. 숫자/규칙을 새로 넣으면 낮은 우선순위의 기존 설명 블록을 같이 빼서 총량을 유지하라\n"
        "12. 검증이 실패하면 그대로 끝내지 말고, 실패 원인을 반영해 수정한 뒤 요약에 검증 결과를 함께 남겨라"
    )


def _load_past_reviews(episode_key: str, current_run_dir: Path, max_entries: int = MANAGER_HISTORY_MAX) -> list[dict]:
    """이전 daily 실행들의 auto_review JSON을 수집. 최신 순으로 max_entries개 반환."""
    daily_base = OUTPUT_DIR / "daily"
    past_reviews: list[dict] = []
    current_date_dir = current_run_dir.parent  # e.g. output/daily/20260319_ep01_...

    for date_dir in sorted(daily_base.glob(f"*_{episode_key}"), reverse=True):
        if date_dir == current_date_dir:
            continue  # 현재 실행은 제외
        # 해당 date_dir의 가장 최근 time_dir
        time_dirs = sorted(date_dir.iterdir(), reverse=True)
        for time_dir in time_dirs:
            review_files = sorted(time_dir.glob("auto_review_cycle*.json"), reverse=True)
            for rf in review_files:
                try:
                    data = json.loads(rf.read_text(encoding="utf-8"))
                    past_reviews.append(data)
                except Exception:
                    pass
            if review_files:
                break  # 해당 날짜의 가장 최근 실행 하나만
        if len(past_reviews) >= max_entries:
            break

    return past_reviews[:max_entries]


def _load_fixer_cycle_history(run_dir: Path) -> list[dict]:
    """
    현재 실행의 AUTO 이터레이션별 점수 + 코드 변경 요약 수집.
    반환: [{cycle, thrill, style, avg, changed_files, added_lines, removed_lines}, ...]
    """
    history: list[dict] = []
    review_files = sorted(run_dir.glob("auto_review_cycle*.json"), key=lambda p: p.name)
    for rf in review_files:
        m = re.search(r"cycle(\d+)", rf.stem)
        if not m:
            continue
        c = int(m.group(1))
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        t = data.get("thrill_score_10", 0)
        s = data.get("style_score_10", 0)

        # 해당 사이클 이전 백업에서 코드 변경 요약
        backup_dir = run_dir / f"backup_before_fixer_cycle{c}"
        changed_files: list[str] = []
        added = removed = 0
        if backup_dir.exists():
            for rel_path in FIXER_TARGET_FILES:
                src = REPO_ROOT / rel_path
                bak = backup_dir / Path(rel_path).name
                if not (src.exists() and bak.exists()):
                    continue
                old_lines = bak.read_text(encoding="utf-8", errors="replace").splitlines()
                new_lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
                if old_lines != new_lines:
                    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
                    a = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
                    r = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
                    changed_files.append(Path(rel_path).name)
                    added += a
                    removed += r

        history.append({
            "cycle": c,
            "thrill": t,
            "style": s,
            "avg": round((t + s) / 2, 1),
            "changed_files": changed_files,
            "added_lines": added,
            "removed_lines": removed,
        })
    return history


def _heuristic_manager_start_hints(current_review: dict) -> str:
    corpus_parts = []
    for key in ("what_felt_boring_or_hard", "style_tips"):
        value = current_review.get(key, [])
        if isinstance(value, list):
            corpus_parts.extend(str(item) for item in value if str(item).strip())
    corpus_parts.append(str(current_review.get("reader_comment", "")))
    corpus = " ".join(corpus_parts)

    hint_specs = [
        (
            ("시간축", "시간 순서", "순서가 섞", "되감기", "재배열", "질의응답", "복도 대치", "발표 종료 후"),
            "src/novel_writer/scene_distiller.py",
            ["normalize_scene_timeline", "_coerce_scene_turn_range", "_merge_scene_pair"],
            "씬 순서가 꼬였거나 같은 장면이 다시 재연되는 문제부터 확인",
        ),
        (
            ("반복", "설명투", "연결어", "그 직후", "잠시 뒤", "문장 흐름", "리듬", "가독성", "용어 설명"),
            "src/novel_writer/prose_generator.py",
            ["_merge_clipped_sentence_runs", "_rhythm_bridge_sentence", "_trim_post_metaphor_explanations"],
            "반복 문장, 상투 연결어, 설명성 문체 문제부터 확인",
        ),
        (
            ("멈춘", "정체", "진행", "장면 기능", "캐릭터", "동기", "질문", "응답", "긴장감"),
            "src/novel_writer/director.py",
            ["_scene_progress_signal", "_should_end_scene", "_choose_next_speaker"],
            "장면이 앞으로 나아가지 않거나 캐릭터 반응이 납작한 문제부터 확인",
        ),
        (
            ("turn_start", "turn_end", "T11", "ValueError", "파싱", "JSON", "숫자 형식"),
            "src/novel_writer/scene_distiller.py",
            ["_coerce_turn_number", "_coerce_scene_turn_range", "_normalize_scene_payload"],
            "LLM 출력 형식 변화로 인한 파싱 오류부터 확인",
        ),
    ]

    hints: list[str] = []
    seen_files: set[str] = set()
    for tokens, rel_path, functions, reason in hint_specs:
        if rel_path in seen_files:
            continue
        if any(token in corpus for token in tokens):
            seen_files.add(rel_path)
            hints.append(
                f"- 시작 후보: {rel_path} -> {', '.join(functions)} | 이유: {reason}"
            )

    if not hints:
        hints.append(
            "- 시작 후보: src/novel_writer/scene_distiller.py -> distill, _apply_scene_readability_guards | 이유: 리뷰 이슈가 씬 압축/정렬 단계에서 시작하는 경우가 가장 잦음"
        )
        hints.append(
            "- 다음 후보: src/novel_writer/prose_generator.py -> _build_scene_prompt, _merge_clipped_sentence_runs | 이유: 문체/흐름 문제의 2차 점검 지점"
        )

    return "\n".join(hints[:3])


def _build_manager_synthesis_prompt(
    current_review: dict,
    past_reviews: list[dict],
    daily_cycle: int,
    manager_period: int,
    fixer_cycle: int,
    fixer_history: list[dict] | None = None,
    param_analysis_report: str = "",
) -> str:
    """현재 + 과거 리뷰를 종합해 Codex에게 줄 타겟 지시를 생성하는 매니저 프롬프트."""
    def _review_summary(rev: dict, label: str) -> str:
        return (
            f"### {label}\n"
            f"긴장감={rev.get('thrill_score_10','?')}/10, 문체={rev.get('style_score_10','?')}/10\n"
            f"문제점: {rev.get('what_felt_boring_or_hard', [])}\n"
            f"팁: {rev.get('style_tips', [])}\n"
            f"한줄평: {rev.get('one_line_verdict', '')}\n"
        )

    is_deep = manager_period > 0 and daily_cycle % manager_period == 0
    current_block = _review_summary(current_review, "현재 사이클 리뷰")
    heuristic_hints = _heuristic_manager_start_hints(current_review)

    # ── 이터레이션별 점수+코드변경 추세 ──
    trend_block = ""
    if fixer_history:
        trend_block = "\n## 이번 실행 이터레이션별 추세 (점수 + 코드 변경 인과관계)\n"
        prev_avg: float | None = None
        for h in fixer_history:
            delta = ""
            if prev_avg is not None:
                diff_val = h["avg"] - prev_avg
                delta = f" (▲{diff_val:+.1f})" if diff_val != 0 else " (변화 없음)"
            files_str = ", ".join(h["changed_files"]) if h["changed_files"] else "변경 없음"
            trend_block += (
                f"- 사이클 {h['cycle']}: 긴장감={h['thrill']}, 문체={h['style']}, "
                f"평균={h['avg']}{delta} | 수정: {files_str}"
            )
            if h["added_lines"] or h["removed_lines"]:
                trend_block += f" (+{h['added_lines']}/-{h['removed_lines']}줄)"
            trend_block += "\n"
            prev_avg = h["avg"]

        # 점수에 영향을 준 패턴 분석 힌트
        if len(fixer_history) >= 2:
            improving = [
                h for i, h in enumerate(fixer_history[1:], 1)
                if h["avg"] > fixer_history[i - 1]["avg"] and h["changed_files"]
            ]
            worsening = [
                h for i, h in enumerate(fixer_history[1:], 1)
                if h["avg"] < fixer_history[i - 1]["avg"] and h["changed_files"]
            ]
            if improving:
                good_files = set(f for h in improving for f in h["changed_files"])
                trend_block += f"※ 점수 향상과 연관된 파일: {', '.join(good_files)}\n"
            if worsening:
                bad_files = set(f for h in worsening for f in h["changed_files"])
                trend_block += f"※ 점수 하락과 연관된 파일: {', '.join(bad_files)} — 해당 파일 수정 시 신중히\n"

    history_block = ""
    if past_reviews and is_deep:
        history_block = f"\n## 과거 {len(past_reviews)}회 리뷰 히스토리 (심층 진단)\n"
        for i, rev in enumerate(past_reviews, 1):
            history_block += _review_summary(rev, f"{i}회 전")

    depth_label = "심층 진단 (5사이클 회고)" if is_deep else "일반 종합"

    # 점수 정체 여부 계산
    stagnation_block = ""
    if fixer_history and len(fixer_history) >= 2:
        recent = fixer_history[-3:]
        avgs = [h["avg"] for h in recent]
        if max(avgs) - min(avgs) < 0.3:
            stagnation_block = (
                f"\n⚠️ 경고: 최근 {len(recent)}사이클 동안 평균 점수가 {avgs[0]:.1f}~{avgs[-1]:.1f}로 "
                f"사실상 제자리다. 지금까지와 같은 방식의 수정은 효과가 없다는 증거다.\n"
            )

    # 효과 없었던 파일 목록
    ineffective_block = ""
    if fixer_history and len(fixer_history) >= 2:
        worsening_files: set[str] = set()
        no_effect_files: set[str] = set()
        for i, h in enumerate(fixer_history[1:], 1):
            if h["avg"] <= fixer_history[i - 1]["avg"] and h["changed_files"]:
                if h["avg"] < fixer_history[i - 1]["avg"]:
                    worsening_files.update(h["changed_files"])
                else:
                    no_effect_files.update(h["changed_files"])
        if worsening_files:
            ineffective_block += f"❌ 수정 후 점수가 하락한 파일: {', '.join(worsening_files)} — 이 파일을 같은 방식으로 수정하는 것을 금지한다.\n"
        if no_effect_files:
            ineffective_block += f"⚠️ 수정했지만 점수 변화 없었던 파일: {', '.join(no_effect_files)} — 다른 접근이 필요하다.\n"

    deep_mandate = ""
    if is_deep:
        deep_mandate = (
            "\n## ⚡ 5사이클 심층 회고 — 강제 전략 재검토\n"
            "지금까지의 수정 패턴을 냉정하게 평가하라.\n"
            "- 같은 파일을 계속 건드렸는가? 그렇다면 근본 원인이 다른 파일에 있을 수 있다.\n"
            "- 점수가 올랐다가 내려갔다면 회귀 위험이 있는 수정이 있다는 뜻이다. 그 수정을 되돌려라.\n"
            "- 이번 지시사항에서는 이전에 한 번도 건드리지 않은 파일 또는 함수를 반드시 1개 이상 포함시켜라.\n"
            "- 과거 히스토리에서 반복적으로 등장하는 문제가 있다면 그것이 진짜 병목이다. 이번에 끝내라.\n"
        )

    param_analysis_block = ""
    if param_analysis_report and is_deep:
        param_analysis_block = (
            "\n## 📊 하이퍼파라미터 Factor Analysis (25 서브트라이얼 누적)\n"
            + param_analysis_report
            + "\n"
            "→ 위 분석에서 점수와 강하게 상관된 파라미터는 Optuna가 이미 최적화 중이다.\n"
            "→ **코드 수정에서는 파라미터가 아닌 생성 로직 자체의 구조적 문제에만 집중하라.**\n"
        )

    return (
        f"당신은 소설 생성 AI의 수석 매니저다. 목표는 챕터 품질 점수를 9.5 이상으로 끌어올리는 것이다.\n"
        f"현재 상황: 일일 사이클 {daily_cycle}, 픽서 내부 사이클 {fixer_cycle}, {depth_label}\n\n"
        f"## {current_block}\n"
        f"{trend_block}"
        f"{stagnation_block}"
        f"{ineffective_block}"
        f"\n## 시작점 힌트 (휴리스틱)\n{heuristic_hints}\n"
        f"{history_block}"
        f"{param_analysis_block}"
        f"{deep_mandate}\n"
        "---\n"
        "다음 Codex 수정 사이클에 줄 지시사항을 한국어로 작성하라.\n"
        "요구사항:\n"
        "1. 점수가 낮은 구체적 원인을 진단하라 — '개선 필요'가 아니라 '어떤 함수/로직이 문제인지' 지목하라.\n"
        "2. 효과 없었던 접근법은 반복하지 마라. 다른 파일, 다른 함수, 다른 전략을 제시하라.\n"
        "3. 지시사항은 Codex가 즉시 실행할 수 있을 만큼 구체적으로 — 함수명, 수정 방향, 예상 효과까지.\n"
        + ("4. 반복 패턴이 보이면 그것을 명시적으로 차단하라.\n" if is_deep else "")
        + "수정 대상: prose_generator.py, scene_distiller.py, director.py, generate_chapter.py\n"
        "형식:\n"
        "진단:\n"
        "- 현재 점수가 낮은 핵심 원인: <원인>\n"
        "- 이전 수정 중 효과 없었던 것: <파일/접근법>\n"
        "시작점:\n"
        "- 1차 파일: <파일명>\n"
        "- 1차 함수: <함수명 1~3개>\n"
        "- 이유: <왜 여기서 시작하는지>\n"
        "실행 항목:\n"
        "1. <구체적 액션 — 함수명과 수정 방향 포함>\n"
        "2. <구체적 액션>\n"
        "3. <선택적 액션>\n"
        "금지 항목:\n"
        "- <이번 사이클에서 하지 말아야 할 것>\n"
        "검증:\n"
        "- <수정 후 확인할 포인트>\n"
        "확인 질문 없이 지시사항만 작성하라."
    )


async def run_manager_agent(
    episode_key: str,
    run_dir: Path,
    current_review: dict,
    daily_cycle: int,
    fixer_cycle: int,
    notify: NotifyFn,
    review_tier: str = "premium",
    manager_period: int = MANAGER_PERIOD,
    cost_tracker: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    param_analysis_report: str = "",
) -> str:
    """
    novel-loop 매니저 에이전트 이식판.
    현재 리뷰 + 과거 daily 리뷰 히스토리를 GPT로 종합 →
    Codex fixer에게 줄 타겟 지시사항 반환.
    manager_period 사이클마다 심층 회고 수행.
    """
    past_reviews = await asyncio.to_thread(
        _load_past_reviews, episode_key, run_dir, MANAGER_HISTORY_MAX
    )
    fixer_history = await asyncio.to_thread(_load_fixer_cycle_history, run_dir)
    is_deep = manager_period > 0 and daily_cycle % manager_period == 0
    depth_label = "심층 진단" if is_deep else "일반 종합"

    if notify:
        _hist_str = f" + 과거 {len(past_reviews)}회 daily 히스토리" if past_reviews and is_deep else ""
        _trend_str = f"이터레이션 {len(fixer_history)}개 추세" if fixer_history else "첫 이터레이션"
        _fa_str = " + Factor Analysis" if is_deep else ""
        await notify(
            f"{DAILY_TAG}[MANAGER] 🧠 매니저 분석 ({depth_label}) — {_trend_str}{_hist_str}{_fa_str}"
        )

    # ── 5사이클마다: Factor Analysis 보고서 생성 및 쓰레드 게시 ──
    factor_report_text = ""
    if is_deep:
        try:
            from tools.analysis.factor_analysis import run_full_analysis
            from openai import OpenAI
            _fa_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            factor_report_text = await asyncio.to_thread(
                run_full_analysis, run_dir, _fa_client
            )
            if notify and factor_report_text:
                await notify(
                    f"{DAILY_TAG}[MANAGER] 📊 Factor Analysis 보고서 "
                    f"(누적 데이터 분석):\n{factor_report_text[:1800]}"
                )
        except Exception as _fa_exc:
            logger.warning("Factor analysis skipped: %s", _fa_exc)
            if notify:
                await notify(f"{DAILY_TAG}[MANAGER] ⚠️ Factor Analysis 생략: {_fa_exc}")

    # ── 점수 이력 보고서 쓰레드 게시 ──
    if notify and fixer_history:
        history_lines = ["**📈 점수 이력 (이번 실행)**"]
        prev_avg = None
        for h in fixer_history:
            delta_str = ""
            if prev_avg is not None:
                diff_val = h["avg"] - prev_avg
                delta_str = f" (▲{diff_val:+.1f})" if diff_val != 0 else " (변화 없음)"
            files_str = ", ".join(h["changed_files"]) if h["changed_files"] else "변경 없음"
            history_lines.append(
                f"사이클 {h['cycle']}: 긴장감={h['thrill']}, 문체={h['style']}, "
                f"평균={h['avg']}{delta_str} | 수정: {files_str}"
            )
            prev_avg = h["avg"]
        await notify(f"{DAILY_TAG}[MANAGER] " + "\n".join(history_lines))

    # ── GPT로 두 보고서를 합쳐 강한 최종 지시사항 생성 ──
    synthesis_prompt = _build_manager_synthesis_prompt(
        current_review, past_reviews, daily_cycle, manager_period, fixer_cycle,
        fixer_history=fixer_history,
        param_analysis_report=param_analysis_report,
    )
    # Factor Analysis (code-diff) 결과를 프롬프트 앞에 추가
    if factor_report_text:
        synthesis_prompt = factor_report_text + "\n\n---\n\n" + synthesis_prompt

    try:
        llm = LLMClient(
            model="gpt-4o-mini",
            premium_model="gpt-4o",
            budget_usd=1.0,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        manager_instructions = await asyncio.to_thread(
            llm.chat,
            [{"role": "user", "content": synthesis_prompt}],
            use_premium=_use_premium_review_tier(review_tier),
            purpose="manager_synthesis",
            max_tokens=1000 if is_deep else 800,
        )
        _record_budget_usage(cost_tracker, metrics, llm.budget_summary(), cost_key="manager")
        if notify:
            prefix = "🔴 통합 강한 지시사항" if is_deep else "📋 매니저 지시사항"
            await notify(f"{DAILY_TAG}[MANAGER] {prefix}:\n{manager_instructions}")
        return manager_instructions
    except Exception as exc:
        if notify:
            await notify(f"{DAILY_TAG}[MANAGER] ⚠️ 매니저 분석 실패 ({exc}), 기본 리뷰 사용")
        return ""


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


def _rollback_from_backup(backup_dir: Path) -> list[str]:
    """백업 디렉토리에서 원본 파일을 복원. 복원된 파일 목록 반환."""
    restored = []
    for rel_path in FIXER_TARGET_FILES:
        src = REPO_ROOT / rel_path
        backup_file = backup_dir / Path(rel_path).name
        if backup_file.exists():
            src.write_bytes(backup_file.read_bytes())
            restored.append(rel_path)
    return restored


def _syntax_check_target_files(paths: list[str] | None = None) -> list[str]:
    """주어진 Python 파일들의 문법을 검사. paths가 없으면 FIXER_TARGET_FILES 전체를 검사."""
    errors = []
    target_paths = paths or FIXER_TARGET_FILES
    for rel_path in target_paths:
        if not rel_path.endswith(".py"):
            continue
        src = REPO_ROOT / rel_path
        if not src.exists():
            continue
        try:
            ast.parse(src.read_text(encoding="utf-8"))
        except SyntaxError as e:
            errors.append(f"{rel_path}: 문법 오류 line {e.lineno} — {e.msg}")
    return errors


def _run_param_sync_check(changed_files: list[str], run_dir: Path) -> tuple[bool, str]:
    """Check that any new runtime_policy keys added by Codex are registered in both
    rl_policy.json (defaults) and inline_optimizer.py (param space)."""
    import re, json

    inline_opt_changed = "tools/inline_optimizer.py" in changed_files
    rl_policy_changed = "data/rl_policy.json" in changed_files
    prose_distiller_changed = any(
        f in changed_files for f in [
            "src/novel_writer/prose_generator.py",
            "src/novel_writer/scene_distiller.py",
        ]
    )

    if not (prose_distiller_changed or inline_opt_changed):
        return True, "파라미터 동기화 체크 스킵 (해당 파일 미변경)"

    # Find all runtime_policy.get() calls in changed source files
    new_keys: set[str] = set()
    for rel_path in changed_files:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists() or not rel_path.endswith(".py"):
            continue
        try:
            text = full_path.read_text(encoding="utf-8")
            keys = re.findall(r'runtime_policy\.get\(["\'](\w+)["\']', text)
            new_keys.update(keys)
        except Exception:
            pass

    if not new_keys:
        return True, "새 runtime_policy 키 없음"

    # Check rl_policy.json has defaults for all keys
    rl_path = REPO_ROOT / "data" / "rl_policy.json"
    try:
        rl_policy = json.loads(rl_path.read_text(encoding="utf-8")) if rl_path.exists() else {}
    except Exception:
        rl_policy = {}

    missing_in_rl = [k for k in new_keys if k not in rl_policy and not k.startswith("_")]

    # Check inline_optimizer.py has suggest calls for new keys
    inline_path = REPO_ROOT / "tools" / "inline_optimizer.py"
    try:
        inline_text = inline_path.read_text(encoding="utf-8") if inline_path.exists() else ""
    except Exception:
        inline_text = ""
    missing_in_inline = [k for k in new_keys if k not in inline_text and not k.startswith("_")]

    issues = []
    if missing_in_rl:
        issues.append(f"rl_policy.json 기본값 누락: {missing_in_rl}")
    if missing_in_inline:
        issues.append(f"inline_optimizer.py param_space 누락: {missing_in_inline}")

    if issues:
        msg = "파라미터 동기화 경고:\n" + "\n".join(f"  - {i}" for i in issues)
        # Log as warning but don't block (Codex may have added them correctly)
        logger.warning("[PARAM_SYNC] %s", msg)
        (run_dir / "param_sync_warning.txt").write_text(msg, encoding="utf-8")
        return True, f"경고 (비차단): {'; '.join(issues)}"

    return True, f"파라미터 동기화 OK ({len(new_keys)}개 키 확인)"


async def _run_local_fixer_validation(
    changed_files: list[str],
    run_dir: Path,
    fixer_cycle: int,
    stop_event: asyncio.Event | None = None,
    notify: NotifyFn = None,
) -> tuple[bool, str]:
    """Codex 수정 직후 빠른 로컬 검증. 실패 시 False와 사유 반환."""
    python_changed = [path for path in changed_files if path.endswith(".py")]
    if not python_changed:
        return True, "검증 생략: Python 변경 없음"

    validation_log = run_dir / f"fixer_cycle{fixer_cycle}_validation.log"
    log_chunks: list[str] = []

    def _append_log(title: str, body: str) -> None:
        cleaned = str(body or "").strip()
        header = f"## {title}"
        log_chunks.append(header if not cleaned else f"{header}\n{cleaned}")

    if notify:
        await notify(f"{DAILY_TAG}[PROGRAMMER] 🧪 로컬 검증 중... (py_compile + 빠른 회귀 테스트)")

    syntax_errors = _syntax_check_target_files(python_changed)
    if syntax_errors:
        message = "문법 오류: " + " | ".join(syntax_errors)
        _append_log("syntax_check", message)
        try:
            validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
        except Exception:
            pass
        return False, message

    py_compile_cmd = [_resolve_python_cmd(), "-m", "py_compile"] + python_changed
    rc, output = await _stream_subprocess(
        py_compile_cmd,
        stop_event=stop_event,
        timeout_sec=180,
    )
    _append_log("py_compile", output or "(출력 없음)")
    if rc == -1:
        try:
            validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
        except Exception:
            pass
        return False, "로컬 검증 중단됨"
    if rc == -2:
        try:
            validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
        except Exception:
            pass
        return False, "py_compile 타임아웃"
    if rc != 0:
        try:
            validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
        except Exception:
            pass
        preview = (output or "")[-500:] if output else "(출력 없음)"
        return False, f"py_compile 실패: {preview}"

    covered_modules = {
        "src/novel_writer/prose_generator.py",
        "src/novel_writer/scene_distiller.py",
        "src/novel_writer/director.py",
        "generate_chapter.py",
    }
    if covered_modules & set(changed_files):
        test_cmd = [_resolve_python_cmd(), "-m", "unittest", "tests.test_reader_feedback_guards"]
        rc, output = await _stream_subprocess(
            test_cmd,
            stop_event=stop_event,
            timeout_sec=180,
        )
        _append_log("tests.test_reader_feedback_guards", output or "(출력 없음)")
        if rc == -1:
            try:
                validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
            except Exception:
                pass
            return False, "회귀 테스트 중단됨"
        if rc == -2:
            try:
                validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
            except Exception:
                pass
            return False, "회귀 테스트 타임아웃"
        if rc != 0:
            try:
                validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
            except Exception:
                pass
            preview = (output or "")[-700:] if output else "(출력 없음)"
            return False, f"빠른 회귀 테스트 실패: {preview}"

    try:
        validation_log.write_text("\n\n".join(log_chunks), encoding="utf-8")
    except Exception:
        pass

    # Parameter sync check
    sync_ok, sync_reason = await asyncio.to_thread(
        _run_param_sync_check, changed_files, run_dir
    )
    if notify and "경고" in sync_reason:
        await notify(f"{DAILY_TAG}[PROGRAMMER] ⚠️ 파라미터 동기화 경고: {sync_reason}")

    return True, f"로컬 검증 통과 ({validation_log.name})"


async def _run_prompt_smoke_test(
    changed_files: list[str],
    run_dir: Path,
    notify=None,
) -> tuple[bool, str]:
    """Quick smoke test: if prose_generator or scene_distiller changed, verify
    the module imports cleanly and key classes instantiate without errors."""
    smoke_targets = {
        "src/novel_writer/prose_generator.py": "from src.novel_writer.prose_generator import ProseGenerator; print('OK')",
        "src/novel_writer/scene_distiller.py": "from src.novel_writer.scene_distiller import SceneDistiller; print('OK')",
        "src/novel_writer/director.py": "from src.novel_writer.director import DirectorAI; print('OK')",
        "src/novel_writer/orchestrator.py": "from src.novel_writer.orchestrator import SimulationOrchestrator; print('OK')",
        "tools/inline_optimizer.py": "from tools.inline_optimizer import run_inline_optimize, update_rl_policy, run_mini_reoptimize, log_cycle_score, param_factor_analysis; print('OK')",
    }

    failures = []
    for rel_path in changed_files:
        smoke_cmd = smoke_targets.get(rel_path)
        if not smoke_cmd:
            continue
        try:
            proc = await asyncio.create_subprocess_exec(
                _resolve_python_cmd(), "-c", smoke_cmd,
                cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()[:200]
                failures.append(f"{rel_path}: {err}")
        except asyncio.TimeoutError:
            failures.append(f"{rel_path}: smoke test timeout")
        except Exception as exc:
            failures.append(f"{rel_path}: {exc}")

    if failures:
        msg = "Smoke test 실패:\n" + "\n".join(f"  - {f}" for f in failures)
        (run_dir / "smoke_test_failures.txt").write_text(msg, encoding="utf-8")
        return False, msg

    tested = [f for f in changed_files if f in smoke_targets]
    return True, f"Smoke test OK ({len(tested)}개 모듈)"


def _build_gpt_code_review_prompt(
    backup_dir: Path,
    summary: str,
    changed_files: list[str] | None = None,
    validation_summary: str = "",
) -> str:
    """Codex가 수정한 diff를 기반으로 GPT 코드리뷰 프롬프트 생성."""
    diff_blocks = []
    review_paths = changed_files or FIXER_TARGET_FILES
    normalized_paths = [path for path in review_paths if path in FIXER_TARGET_FILES]
    for rel_path in normalized_paths:
        src = REPO_ROOT / rel_path
        backup = backup_dir / Path(rel_path).name
        if not (src.exists() and backup.exists()):
            continue
        old_text = backup.read_text(encoding="utf-8", errors="replace")
        new_text = src.read_text(encoding="utf-8", errors="replace")
        if old_text == new_text:
            continue
        diff_lines = list(difflib.unified_diff(
            old_text.splitlines(), new_text.splitlines(),
            fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}", n=3,
        ))
        if diff_lines:
            # diff 최대 120줄로 제한
            diff_blocks.append(f"### {rel_path}\n```diff\n" + "\n".join(diff_lines[:120]) + "\n```")

    if not diff_blocks:
        return ""

    changed_block = "\n".join(f"- {path}" for path in normalized_paths) if normalized_paths else "- (변경 파일 없음)"
    return (
        "다음 코드 변경사항을 검토하라. 소설 생성 파이프라인의 품질을 개선하는 변경인지 판단하라.\n\n"
        "Codex 수정 요약:\n" + summary[:500] + "\n\n"
        "이번 사이클 실제 변경 파일:\n" + changed_block + "\n\n"
        + ("로컬 검증 결과:\n" + validation_summary[:400] + "\n\n" if validation_summary else "")
        + "\n\n".join(diff_blocks[:3]) +  # 최대 3개 파일
        "\n\n---\n"
        "검토 순서:\n"
        "1. 먼저 diff에 나타난 변경 부분이 리뷰 이슈를 직접 해결하는지 본다.\n"
        "2. 그 다음 이 해결책이 실제 실행 시 오류 없이 동작할 가능성이 높은지 본다.\n"
        "3. 변경된 함수의 인접 문맥에서 문법/논리 회귀가 없는지 본다.\n"
        "4. 파일 전체를 재작성하려는 시각이 아니라, 바뀐 부분이 맞는지 우선 판단한다.\n\n"
        "JSON으로만 응답하라 (다른 텍스트 금지):\n"
        '{"verdict": "approve" or "reject", "reason": "한국어로 한 문장"}\n\n'
        "approve 기준: 변경이 리뷰 이슈를 직접 개선하고, 로컬 검증과 인접 문맥상 명백한 오류 위험이 낮을 때\n"
        "reject 기준: 문제 해결이 불충분하거나, 문법 오류/런타임 위험/기존 핵심 기능 훼손 가능성이 있을 때"
    )


async def _run_gpt_code_review(
    backup_dir: Path,
    summary: str,
    run_dir: Path,
    fixer_cycle: int,
    changed_files: list[str] | None = None,
    validation_summary: str = "",
    cost_tracker: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Codex 수정 후 GPT Agent 코드리뷰.
    (approved: bool, reason: str) 반환.
    문법 오류 발견 시 즉시 reject.
    GPT 분석 실패 시 approve로 fallback (파이프라인 중단 방지).
    """
    # 1단계: 문법 체크 (빠른 로컬 검사)
    syntax_errors = _syntax_check_target_files(changed_files)
    if syntax_errors:
        return False, "문법 오류: " + " | ".join(syntax_errors)

    # 2단계: GPT diff 리뷰
    prompt = _build_gpt_code_review_prompt(
        backup_dir,
        summary,
        changed_files=changed_files,
        validation_summary=validation_summary,
    )
    if not prompt:
        return True, "변경된 파일 없음"

    try:
        llm = LLMClient(
            model="gpt-4o-mini",
            premium_model="gpt-4o",
            budget_usd=0.5,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        raw = await asyncio.to_thread(
            llm.chat,
            [{"role": "user", "content": prompt}],
            use_premium=False,  # gpt-4o-mini로 충분
            purpose="code_review",
            max_tokens=200,
        )
        _record_budget_usage(cost_tracker, metrics, llm.budget_summary(), cost_key="code_review")
        cleaned = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`")
        result = json.loads(cleaned)
        verdict = result.get("verdict", "approve")
        reason = result.get("reason", "")

        # 리뷰 결과 저장
        review_out = run_dir / f"code_review_cycle{fixer_cycle}.json"
        review_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        return verdict == "approve", reason
    except Exception as exc:
        # GPT 실패 시 approve fallback (파이프라인 중단 방지)
        return True, f"GPT 코드리뷰 실패 ({exc}) — approve fallback"


def _detect_changed_target_files(backup_dir: Path) -> list[str]:
    changed: list[str] = []
    for rel_path in FIXER_TARGET_FILES:
        current = REPO_ROOT / rel_path
        backup = backup_dir / current.name
        if not backup.exists():
            continue
        if not current.exists():
            changed.append(rel_path)
            continue
        try:
            if current.read_bytes() != backup.read_bytes():
                changed.append(rel_path)
        except Exception:
            changed.append(rel_path)
    return changed


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


async def _run_codex_fixer(
    prompt: str,
    run_dir: Path,
    fixer_cycle: int,
    set_process: ProcessFn = None,
    notify: NotifyFn = None,
    stop_event: asyncio.Event | None = None,
    codex_model: str | None = None,
) -> tuple[bool, str]:
    """Codex CLI로 코드를 직접 수정. 성공 여부와 요약 반환."""
    if notify:
        model_label = f" [{codex_model}]" if codex_model else ""
        await notify(f"{DAILY_TAG}[FIXER] 🔧 Codex 수정 시작 (사이클 {fixer_cycle}){model_label}")
    summary_path = run_dir / f"fixer_cycle{fixer_cycle}_summary.md"
    output_log_path = run_dir / f"fixer_cycle{fixer_cycle}_stdout.log"
    cmd = _resolve_codex_exec_cmd(codex_model=codex_model)
    if not cmd:
        if notify:
            await notify(f"{DAILY_TAG}[FIXER] ℹ️ Codex CLI 없음 → GPT API로 대체")
        gpt_model = "gpt-4.1-mini" if (codex_model and "mini" in codex_model) else "gpt-4.1"
        return await _run_gpt_fixer(
            prompt, run_dir, fixer_cycle,
            set_process=set_process, notify=notify, stop_event=stop_event,
            model=gpt_model,
        )
    cmd += [
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        "-o", str(summary_path),
        prompt,
    ]

    _buffer: list[str] = []

    async def _on_fixer_line(line: str) -> None:
        friendly = _friendly_fixer_line(line)
        if not friendly:
            return
        _buffer.append(friendly)
        if len(_buffer) >= 4:
            if notify:
                await notify(f"{DAILY_TAG}[FIXER] ⚙️\n" + "\n".join(_buffer))
            _buffer.clear()

    async def _on_fixer_heartbeat(elapsed: int) -> None:
        mins = elapsed // 60
        if notify:
            await notify(f"{DAILY_TAG}[FIXER] ⏳ Codex 수정 중... ({mins}분 경과)")

    async with _get_codex_fixer_lock():
        rc, output = await _stream_subprocess(
            cmd,
            on_line=_on_fixer_line,
            stop_event=stop_event,
            timeout_sec=1200,
            on_heartbeat=_on_fixer_heartbeat,
            heartbeat_sec=60,
            on_process_started=(lambda pid: set_process("codex_fixer", pid, " ".join(cmd))) if set_process else None,
            on_process_ended=(lambda: set_process(None, None, None)) if set_process else None,
        )

    # 남은 버퍼 전송
    if _buffer and notify:
        await notify(f"{DAILY_TAG}[FIXER] ⚙️\n```\n{chr(10).join(_buffer)}\n```")

    if output:
        try:
            output_log_path.write_text(output, encoding="utf-8")
        except Exception:
            pass

    if rc == -1:
        if notify:
            await notify(f"{DAILY_TAG}[FIXER] ❌ Codex 수정 실패 — 중단됨")
        return False, "Codex fixer 중단됨"
    if rc == -2:
        if notify:
            await notify(f"{DAILY_TAG}[FIXER] ❌ Codex 수정 타임아웃 (20분)")
        return False, f"Codex 타임아웃 (rc=-2)\n로그: {output_log_path.name}"
    if rc != 0:
        msg = f"Codex 실패 (rc={rc})\n{output[-800:]}"
        if notify:
            await notify(f"{DAILY_TAG}[FIXER] ❌ Codex 수정 실패 (rc={rc})")
        return False, msg

    summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else output[-1000:]
    if notify:
        await notify(f"{DAILY_TAG}[FIXER] ✅ Codex 수정 완료 (사이클 {fixer_cycle})")
    return True, summary


async def _run_gpt_fixer(
    prompt: str,
    run_dir: Path,
    fixer_cycle: int,
    set_process: ProcessFn = None,
    notify: NotifyFn = None,
    stop_event: asyncio.Event | None = None,
    model: str = "gpt-4.1",
) -> tuple[bool, str]:
    """GPT API를 사용해 코드를 직접 수정 (Codex CLI 대체). 성공 여부와 요약 반환."""
    from openai import OpenAI

    if notify:
        await notify(f"{DAILY_TAG}[FIXER] 🔧 GPT Fixer 수정 시작 (사이클 {fixer_cycle}) [{model}]")

    summary_path = run_dir / f"fixer_cycle{fixer_cycle}_summary.md"
    output_log_path = run_dir / f"fixer_cycle{fixer_cycle}_stdout.log"

    # Read all target files
    file_contents: dict[str, str] = {}
    for rel_path in FIXER_TARGET_FILES:
        abs_path = REPO_ROOT / rel_path
        if abs_path.exists():
            try:
                file_contents[rel_path] = abs_path.read_text(encoding="utf-8")
            except Exception:
                pass

    if not file_contents:
        return False, "수정 대상 파일을 읽을 수 없음"

    files_block = "\n\n".join(
        f"### {path}\n```\n{content}\n```"
        for path, content in file_contents.items()
    )

    system_msg = (
        "당신은 소설 생성 파이프라인의 코드 개선 전문가입니다.\n"
        "주어진 지시사항과 파일 내용을 분석하고, 수정이 필요한 파일만 수정하여 JSON으로 반환하세요.\n\n"
        "반환 형식 (순수 JSON만, 다른 텍스트 없음):\n"
        "{\n"
        '  "files": [\n'
        '    {"path": "파일경로", "content": "수정된 전체 파일 내용"},\n'
        '    ...\n'
        '  ],\n'
        '  "summary": "수정 내용 한국어 요약 (어떤 파일의 어떤 부분을 왜 수정했는지)"\n'
        "}\n\n"
        "규칙:\n"
        "- 수정한 파일만 files 배열에 포함 (변경 없는 파일은 제외)\n"
        "- path는 반드시 제공된 파일 목록 중에서만 선택\n"
        "- content는 해당 파일의 완전한 내용 (부분이 아닌 전체 파일)\n"
        "- 수정하지 않는 파일은 files에 넣지 말 것"
    )

    user_msg = f"## 수정 지시사항\n\n{prompt}\n\n## 현재 파일 내용\n\n{files_block}"

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    try:
        if notify:
            await notify(f"{DAILY_TAG}[FIXER] ⏳ GPT API 호출 중... (모델: {model})")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=16000,
            ),
        )

        if stop_event and stop_event.is_set():
            return False, "GPT Fixer 중단됨"

        raw = response.choices[0].message.content or ""
        try:
            output_log_path.write_text(raw, encoding="utf-8")
        except Exception:
            pass

        data = json.loads(raw)
        files_to_write: list[dict] = data.get("files", [])
        summary: str = data.get("summary", "")

        if not files_to_write:
            return False, "GPT가 수정할 파일을 반환하지 않음"

        modified_paths: list[str] = []
        for file_entry in files_to_write:
            rel_path = file_entry.get("path", "")
            content = file_entry.get("content", "")
            if not rel_path or not content:
                continue
            # 허용된 경로만 쓰기 허용
            if rel_path not in FIXER_TARGET_FILES:
                continue
            abs_path = REPO_ROOT / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
            modified_paths.append(rel_path)

        if not modified_paths:
            return False, "GPT 응답에서 유효한 파일 수정 없음"

        # py_compile 검증
        compile_errors: list[str] = []
        for rel_path in modified_paths:
            if not rel_path.endswith(".py"):
                continue
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "py_compile", str(REPO_ROOT / rel_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            if proc.returncode != 0:
                compile_errors.append(f"{rel_path}: {out.decode('utf-8', errors='replace').strip()}")

        if compile_errors:
            err_text = "\n".join(compile_errors)
            if notify:
                await notify(f"{DAILY_TAG}[FIXER] ⚠️ 컴파일 오류 — 파일 롤백:\n```\n{err_text[:500]}\n```")
            return False, f"컴파일 오류:\n{err_text}"

        try:
            summary_path.write_text(summary, encoding="utf-8")
        except Exception:
            pass

        if notify:
            files_label = ", ".join(modified_paths)
            await notify(f"{DAILY_TAG}[FIXER] ✅ GPT 수정 완료 (사이클 {fixer_cycle}) — {files_label}")
        return True, summary

    except Exception as exc:
        if notify:
            await notify(f"{DAILY_TAG}[FIXER] ❌ GPT Fixer 오류: {exc}")
        return False, f"GPT Fixer 오류: {exc}"


async def step_auto_improve_loop(
    episode_key: str,
    run_dir: Path,
    chapter_path: Path,
    target_words: int,
    budget: float,
    protagonist: str,
    guardian_briefing_path: Path | None,
    notify: NotifyFn,
    upload: UploadFn,
    set_status: StatusFn,
    stop_event: asyncio.Event | None,
    set_process: ProcessFn = None,
    review_tier: str = "premium",
    cost_tracker: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
    max_cycles: int = AUTO_IMPROVE_MAX_CYCLES,
    score_threshold: int = AUTO_IMPROVE_SCORE_THRESHOLD,
    daily_cycle: int = 1,
    manager_period: int = MANAGER_PERIOD,
    outer_max_cycles: int | None = None,
) -> Path:
    """
    배치 파라미터 탐색 + Codex 코드 픽스 루프.

    Outer loop (최대 AUTO_OUTER_MAX_CYCLES = 3회):
      Phase A: 25-trial 배치 파라미터 탐색 (5 groups × 5 병렬, SQLite 영속 study)
               → best params → rl_policy.json 업데이트
      Phase B: AI review + param_factor_analysis
               → 점수 통과 시 종료
      Phase C: Codex fix inner loop (최대 AUTO_INNER_MAX_CYCLES = 3회)
               → Manager (리뷰 + factor analysis) → Codex → Regen → 점수 체크
               → 통과 시 종료, 3회 소진 시 outer loop 재시작 (수정된 코드로 재탐색)

    총 최대 챕터 생성: outer 3 × (25 trials + 3 regen) = 84회
    """
    current_chapter = chapter_path

    # ── Load episode context ──────────────────────────────────────────────────
    _opt_ctx: dict | None = None
    try:
        from tools.inline_optimizer import run_mini_reoptimize, log_cycle_score, param_factor_analysis, CYCLE_SCORE_LOG
        from src.novel_writer.rl_policy import load_policy
        from src.novel_writer.config_loader import load_episode, load_characters

        ep_file = resolve_episode_file(episode_key)
        _ep_cfg = load_episode(str(ep_file))
        _ep_id = str(_ep_cfg.get("id") or ep_file.stem).strip()
        _char_profiles = load_characters(str(REPO_ROOT / "config" / "characters.yaml"))
        _reader_fb: dict | None = None
        try:
            _fb_cands = sorted(run_dir.glob("**/*review*.json"), key=lambda p: p.stat().st_mtime)
            if _fb_cands:
                _reader_fb = json.loads(_fb_cands[-1].read_text(encoding="utf-8"))
        except Exception:
            pass
        _opt_ctx = {
            "episode_id": _ep_id,
            "episode_config": _ep_cfg,
            "protagonist_id": protagonist,
            "protagonist_name": protagonist,
            "character_profiles": _char_profiles,
            "reader_feedback": _reader_fb,
            "base_model": "gpt-4o-mini",
            "premium_model": "gpt-4o-mini",
        }
    except Exception as _ctx_exc:
        logger.warning("[AUTO] context load failed: %s — param optimization disabled", _ctx_exc)

    _outer_max = max(1, int(outer_max_cycles or AUTO_OUTER_MAX_CYCLES))
    _inner_max = AUTO_INNER_MAX_CYCLES
    _param_analysis_report = ""
    avg = 0.0
    review_json: dict = {}
    _global_fixer_cycle = 0   # for backup dir naming across outer cycles
    _quality_focus: dict | None = None
    _ALL_SCORE_KEYS = [
        "thrill_score_10", "style_score_10", "causality_score_10",
        "character_score_10", "scene_function_score_10",
    ]

    # ── Pre-run briefing ──────────────────────────────────────────────────────
    if notify:
        _n_groups = (AUTO_BATCH_TRIALS + AUTO_BATCH_GROUP_SIZE - 1) // AUTO_BATCH_GROUP_SIZE
        _max_gens = _outer_max * (AUTO_BATCH_TRIALS + _inner_max)
        _plan_lines = [
            f"{DAILY_TAG}[AUTO] 📋 학습 계획 브리핑",
            f"",
            f"🎯 목표 점수: **{score_threshold}/10** | Outer cycle: **{_outer_max}회** | 목표 달성 시 조기 종료",
            f"",
        ]
        for _oi in range(1, _outer_max + 1):
            _plan_lines.append(
                f"**Outer {_oi}/{_outer_max}**"
            )
            _plan_lines.append(
                f"  Phase A — {AUTO_BATCH_TRIALS} trials "
                f"({_n_groups}그룹 × {AUTO_BATCH_GROUP_SIZE}병렬) 파라미터 탐색"
            )
            _plan_lines.append(
                f"  Phase B — AI 리뷰 **1회** + Factor Analysis → 점수 ≥ {score_threshold} 시 종료"
            )
            _plan_lines.append(
                f"  Phase C — GPT Fixer 최대 {_inner_max}회 (재생성마다 AI 리뷰 1회)"
            )
            _plan_lines.append("")
        _total_reviews = _outer_max * (1 + _inner_max) + 1
        _plan_lines.append(
            f"총 최대 챕터 생성: `{_outer_max} × ({AUTO_BATCH_TRIALS} + {_inner_max}) = {_max_gens}회`"
        )
        _plan_lines.append(
            f"총 AI 리뷰: 최대 `{_outer_max} × (1 + {_inner_max}) + 1 = {_total_reviews}회`"
        )
        await notify("\n".join(_plan_lines))

    for outer_cycle in range(1, _outer_max + 1):
        if stop_event and stop_event.is_set():
            break

        if notify:
            await notify(
                f"{DAILY_TAG}[AUTO] 🚀 Outer {outer_cycle}/{_outer_max} 시작 — "
                f"Phase A: {AUTO_BATCH_TRIALS} trials 배치 파라미터 탐색"
            )
        if set_status:
            set_status(f"AUTO outer {outer_cycle}/{_outer_max} — 파라미터 탐색 중...")

        # ── Phase A: 배치 파라미터 탐색 ─────────────────────────────────────
        _batch_subtrial_data: list[dict] = []
        if _opt_ctx:
            try:
                _guardian_text: str | None = None
                if guardian_briefing_path and guardian_briefing_path.exists():
                    try:
                        _guardian_text = guardian_briefing_path.read_text(encoding="utf-8")
                    except Exception:
                        pass
                _cur_policy = await asyncio.to_thread(load_policy)

                _batch_best_path, _batch_best_params, _batch_best_score, _batch_subtrial_data = (
                    await run_mini_reoptimize(
                        episode_id=_opt_ctx["episode_id"],
                        episode_config=_opt_ctx["episode_config"],
                        run_dir=run_dir,
                        protagonist_id=_opt_ctx["protagonist_id"],
                        protagonist_name=_opt_ctx["protagonist_name"],
                        target_words=target_words,
                        budget=budget,
                        character_profiles=_opt_ctx["character_profiles"],
                        reader_feedback=_opt_ctx["reader_feedback"],
                        guardian_briefing=_guardian_text,
                        current_params=dict(_cur_policy),
                        base_model=_opt_ctx["base_model"],
                        premium_model=_opt_ctx["premium_model"],
                        notify_fn=notify,
                        quality_focus=_quality_focus,
                        cycle_idx=outer_cycle,
                        n_trials=AUTO_BATCH_TRIALS,
                        group_size=AUTO_BATCH_GROUP_SIZE,
                    )
                )
                if _batch_best_path and _batch_best_path.exists():
                    current_chapter = _batch_best_path
                    if notify:
                        await notify(
                            f"{DAILY_TAG}[AUTO] ✅ Phase A 완료 — "
                            f"best score {_batch_best_score:.2f}, 챕터 업데이트"
                        )
            except Exception as _batch_exc:
                logger.warning("[AUTO] Phase A batch failed (outer %d): %s", outer_cycle, _batch_exc)
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ⚠️ Phase A 실패: {_batch_exc}")

        # ── Phase B: AI 리뷰 + factor analysis ──────────────────────────────
        if set_status:
            set_status(f"AUTO outer {outer_cycle}/{_outer_max} — AI 리뷰 중...")

        chapter_text = current_chapter.read_text(encoding="utf-8", errors="replace")
        try:
            _rev_llm = LLMClient(
                model=_llm_review_model_for_tier(review_tier),
                premium_model="gpt-4o",
                budget_usd=2.0,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            _story_ctx = await asyncio.to_thread(_load_story_context_for_review)
            _review_raw = await asyncio.to_thread(
                _rev_llm.chat,
                [{"role": "user", "content": _build_ai_reviewer_prompt(chapter_text, _story_ctx)}],
                use_premium=_use_premium_review_tier(review_tier),
                purpose="auto_improve_reviewer",
                max_tokens=1400,
            )
            _cleaned = re.sub(r"```(?:json)?\n?", "", _review_raw).strip().rstrip("`")
            review_json = json.loads(_cleaned)
            _rev_budget = _rev_llm.budget_summary()
        except Exception as _rev_exc:
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] ⚠️ AI 리뷰 실패 ({_rev_exc}), outer loop 종료")
            break

        thrill    = int(review_json.get("thrill_score_10", 0))
        style     = int(review_json.get("style_score_10", 0))
        causality = int(review_json.get("causality_score_10", 0))
        character = int(review_json.get("character_score_10", 0))
        scene_fn  = int(review_json.get("scene_function_score_10", 0))
        avg = sum(int(review_json.get(k, 0)) for k in _ALL_SCORE_KEYS) / len(_ALL_SCORE_KEYS)
        verdict = review_json.get("one_line_verdict", "")

        _outer_review_path = run_dir / f"auto_review_cycle{outer_cycle}.json"
        _outer_review_path.write_text(
            json.dumps(review_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if cost_tracker is not None:
            cost_tracker["auto_review"] = cost_tracker.get("auto_review", 0.0) + float(
                _rev_budget.get("spent_usd", 0.0)
            )
        _accumulate_usage_totals(metrics, _rev_budget)

        _review_mood = (
            "🏆" if avg >= score_threshold else
            "🌟" if avg >= 7.5 else
            "🎯" if avg >= 6.5 else
            "📉"
        )
        if notify:
            await notify(
                f"{DAILY_TAG}[AUTO] 📊 Phase B 리뷰 결과 (outer {outer_cycle}) {_review_mood}\n"
                f"긴장감: {thrill}/10 | 문체: {style}/10 | 인과성: {causality}/10 | "
                f"캐릭터: {character}/10 | 씬기능: {scene_fn}/10\n"
                f"**평균: {avg:.1f}/10** | 목표: {score_threshold:.1f}/10\n"
                f"한줄평: {verdict}"
            )

        # factor analysis — 25 trials 데이터가 바로 쌓여 있으므로 즉시 유효
        _ai_review_scores = {
            "thrill": thrill, "style": style, "causality": causality,
            "character": character, "scene_fn": scene_fn, "avg": round(avg, 2),
        }
        if _opt_ctx:
            try:
                _logged_policy = await asyncio.to_thread(load_policy)
                await asyncio.to_thread(
                    log_cycle_score,
                    _opt_ctx["episode_id"], outer_cycle,
                    dict(_logged_policy), _ai_review_scores, _batch_subtrial_data,
                    None, dict(cost_tracker),
                )
            except Exception as _log_exc:
                logger.warning("[AUTO] log_cycle_score failed: %s", _log_exc)
            try:
                _param_analysis_report = await asyncio.to_thread(param_factor_analysis)
                if notify and _param_analysis_report and len(_param_analysis_report) > 50:
                    await notify(
                        f"{DAILY_TAG}[AUTO] 📊 Factor Analysis (outer {outer_cycle}):\n"
                        f"{_param_analysis_report[:1000]}"
                    )
            except Exception as _fa_exc:
                logger.warning("[AUTO] param_factor_analysis failed: %s", _fa_exc)

        if avg >= score_threshold:
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] ✅ 품질 통과 (평균 {avg:.1f} ≥ {score_threshold}) "
                    f"— outer {outer_cycle}에서 완료"
                )
            break

        if outer_cycle == _outer_max:
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] ⚠️ 최대 outer 사이클({_outer_max}) 도달 "
                    f"(평균 {avg:.1f}) — 현재 버전으로 진행"
                )
            break

        # ── Phase C: Codex fix inner loop ───────────────────────────────────
        if notify:
            await notify(
                f"{DAILY_TAG}[AUTO] 🔧 Phase C 시작 — "
                f"Codex fix (최대 {_inner_max}회)"
            )

        _quality_focus = {
            "thrill": thrill, "style": style, "causality": causality,
            "character": character, "scene_function": scene_fn,
        }

        for inner_cycle in range(1, _inner_max + 1):
            if stop_event and stop_event.is_set():
                break

            _global_fixer_cycle += 1
            _cost_snapshot = copy.deepcopy(cost_tracker or {})
            _cycle_label = f"outer {outer_cycle} / inner {inner_cycle}"

            if set_status:
                set_status(f"AUTO {_cycle_label} — 매니저 분석 중...")

            # Manager
            try:
                _manager_instructions = await asyncio.wait_for(
                    run_manager_agent(
                        episode_key=episode_key,
                        run_dir=run_dir,
                        current_review=review_json,
                        daily_cycle=daily_cycle,
                        fixer_cycle=_global_fixer_cycle,
                        notify=notify,
                        review_tier=review_tier,
                        manager_period=manager_period,
                        cost_tracker=cost_tracker,
                        metrics=metrics,
                        param_analysis_report=_param_analysis_report,
                    ),
                    timeout=1800.0,
                )
            except asyncio.TimeoutError:
                _manager_instructions = None
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ⚠️ Manager 30분 타임아웃")

            # Codex fix
            if set_status:
                set_status(f"AUTO {_cycle_label} — Codex 수정 중...")
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] 🔧 Codex Fixer 실행 ({_cycle_label})")

            backup_dir = await asyncio.to_thread(_backup_target_files, run_dir, _global_fixer_cycle)

            fixer_prompt = _build_codex_fixer_prompt(review_json, manager_instructions=_manager_instructions)
            ok, summary = await _run_codex_fixer(
                fixer_prompt, run_dir, _global_fixer_cycle,
                set_process=set_process, notify=notify,
                stop_event=stop_event, codex_model=_codex_model_for_tier(review_tier),
            )
            if not ok:
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ❌ Codex 실패 ({_cycle_label}): {summary}")
                break

            changed_files = await asyncio.to_thread(_detect_changed_target_files, backup_dir)
            if notify:
                _chg = "\n".join(f"- {p}" for p in changed_files) if changed_files else "- 없음"
                await notify(f"{DAILY_TAG}[AUTO] ✅ 코드 수정 완료:\n{summary[:400]}\n변경: {_chg}")

            # Validation + smoke test + code review
            val_ok, val_reason = await _run_local_fixer_validation(
                changed_files=changed_files, run_dir=run_dir,
                fixer_cycle=_global_fixer_cycle, stop_event=stop_event, notify=notify,
            )
            if not val_ok:
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ⏪ 검증 실패 → 롤백: {val_reason}")
                await asyncio.to_thread(_rollback_from_backup, backup_dir)
                if cost_tracker is not None:
                    cost_tracker.update(_cost_snapshot)
                break

            smoke_ok, smoke_reason = await _run_prompt_smoke_test(changed_files, run_dir, notify)
            if not smoke_ok:
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] 🔥 Smoke test 실패 → 롤백: {smoke_reason}")
                await asyncio.to_thread(_rollback_from_backup, backup_dir)
                if cost_tracker is not None:
                    cost_tracker.update(_cost_snapshot)
                break

            code_ok, code_reason = await _run_gpt_code_review(
                backup_dir=backup_dir, summary=summary, run_dir=run_dir,
                fixer_cycle=_global_fixer_cycle, changed_files=changed_files,
                validation_summary=val_reason, cost_tracker=cost_tracker, metrics=metrics,
            )
            if not code_ok:
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ⏪ 코드리뷰 reject → 롤백: {code_reason}")
                await asyncio.to_thread(_rollback_from_backup, backup_dir)
                if cost_tracker is not None:
                    cost_tracker.update(_cost_snapshot)
                break

            await _git_commit_fixer_changes(_global_fixer_cycle, episode_key, summary)

            # Re-simulate if needed
            if any(p in SIMULATION_RELEVANT_FIXER_FILES for p in changed_files):
                sim_ok = await step_simulator(
                    episode_key=episode_key, run_dir=run_dir, cycle=daily_cycle,
                    budget=budget, notify=notify, set_status=set_status,
                    stop_event=stop_event, set_process=set_process,
                    cost_tracker=cost_tracker, metrics=metrics,
                    auto_cycle_index=_global_fixer_cycle,
                    auto_max_cycles=_outer_max * _inner_max,
                    guardian_briefing_path=guardian_briefing_path,
                )
                if not sim_ok:
                    if notify:
                        await notify(f"{DAILY_TAG}[AUTO] ⚠️ 재시뮬레이션 실패")
                    break

            # Regen with best params (rl_policy.json already updated by Phase A)
            if set_status:
                set_status(f"AUTO {_cycle_label} — 챕터 재생성 중...")
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] 📖 best params로 챕터 재생성")

            cached_scenes: Path | None = None
            if changed_files and set(changed_files).issubset(SCENE_CACHE_SAFE_FIXER_FILES):
                _sc = run_dir / f"{resolve_episode_file(episode_key).stem}_scenes.json"
                if _sc.exists():
                    cached_scenes = _sc

            _regen_tracker: dict[str, float] = {"chapter": 0.0}
            new_chapter = await step_chapter_gen(
                episode_key, run_dir, daily_cycle, target_words, budget, protagonist,
                notify=notify, upload=upload, set_status=set_status,
                stop_event=stop_event, set_process=set_process,
                cost_tracker=_regen_tracker, metrics=metrics,
                auto_cycle_index=_global_fixer_cycle,
                auto_max_cycles=_outer_max * _inner_max,
                upload_version_label=f"outer{outer_cycle}_inner{inner_cycle}",
                precomputed_scenes_path=cached_scenes,
                guardian_briefing_path=guardian_briefing_path,
                review_tier=review_tier,
            )
            if cost_tracker is not None:
                cost_tracker["auto_chapter"] = (
                    cost_tracker.get("auto_chapter", 0.0)
                    + float(_regen_tracker.get("chapter", 0.0))
                )
            if new_chapter is None:
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ⚠️ 재생성 실패")
                break

            # Score check — rollback if significantly worse
            new_text = new_chapter.read_text(encoding="utf-8", errors="replace")
            _chk_json = review_json   # fallback: keep current review if check fails
            try:
                _chk_llm = LLMClient(
                    model=_llm_review_model_for_tier(review_tier),
                    premium_model=_llm_premium_model_for_tier(review_tier),
                    budget_usd=2.0,
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                )
                _chk_ctx = await asyncio.to_thread(_load_story_context_for_review)
                _chk_raw = await asyncio.to_thread(
                    _chk_llm.chat,
                    [{"role": "user", "content": _build_ai_reviewer_prompt(new_text, _chk_ctx)}],
                    use_premium=_use_premium_review_tier(review_tier),
                    purpose="regen_score_check",
                    max_tokens=1400,
                )
                _chk_json = json.loads(re.sub(r"```(?:json)?\n?", "", _chk_raw).strip().rstrip("`"))
                new_avg = sum(int(_chk_json.get(k, 0)) for k in _ALL_SCORE_KEYS) / len(_ALL_SCORE_KEYS)
                _record_budget_usage(cost_tracker, metrics, _chk_llm.budget_summary(), cost_key="regen_check")
            except Exception as _chk_exc:
                logger.warning("[AUTO] regen score check failed: %s", _chk_exc)
                new_avg = avg

            if new_avg < avg - 0.5:
                if notify:
                    await notify(
                        f"{DAILY_TAG}[AUTO] ⏪ 재생성 후 점수 하락 "
                        f"({avg:.1f} → {new_avg:.1f}) → 롤백"
                    )
                await asyncio.to_thread(_rollback_from_backup, backup_dir)
                if cost_tracker is not None:
                    cost_tracker.update(_cost_snapshot)
                break

            current_chapter = new_chapter
            avg = new_avg
            review_json = _chk_json   # update review for next manager cycle
            if notify:
                wc = len(new_text.split())
                await notify(
                    f"{DAILY_TAG}[AUTO] 📝 재생성 완료 ({wc}단어, 점수 {avg:.1f}/10)"
                )

            if avg >= score_threshold:
                if notify:
                    await notify(
                        f"{DAILY_TAG}[AUTO] ✅ 품질 통과 (평균 {avg:.1f} ≥ {score_threshold}) "
                        f"— {_cycle_label}에서 완료"
                    )
                break  # break inner loop

        # inner loop 완료 — 점수 통과 시 outer loop도 종료
        if avg >= score_threshold:
            break

        # ── Fix E: YAML auto-feedback (if still below threshold, patch episode YAML) ──
        if avg < score_threshold and outer_cycle < _outer_max:
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] 📝 Fix E — YAML 자동 피드백 시작 "
                    f"(avg={avg:.1f} < {score_threshold})"
                )
            try:
                _yaml_feedback = _build_yaml_auto_feedback(review_json)
                _yaml_ok, _yaml_summary = await _run_story_fixer(
                    episode_key=episode_key,
                    user_feedback=_yaml_feedback,
                    run_dir=run_dir,
                    fixer_cycle=_global_fixer_cycle,
                    set_process=set_process,
                    codex_model=_codex_model_for_tier(review_tier),
                )
                if _yaml_ok:
                    if notify:
                        await notify(
                            f"{DAILY_TAG}[AUTO] ✅ YAML 수정 완료:\n{_yaml_summary[:400]}"
                        )
                    # Re-simulate so the next outer cycle's Phase A sees the new YAML
                    _yaml_sim_ok = await step_simulator(
                        episode_key=episode_key, run_dir=run_dir, cycle=daily_cycle,
                        budget=budget, notify=notify, set_status=set_status,
                        stop_event=stop_event, set_process=set_process,
                        cost_tracker=cost_tracker, metrics=metrics,
                        auto_cycle_index=_global_fixer_cycle,
                        auto_max_cycles=_outer_max * _inner_max,
                        guardian_briefing_path=guardian_briefing_path,
                    )
                    if not _yaml_sim_ok and notify:
                        await notify(f"{DAILY_TAG}[AUTO] ⚠️ YAML 수정 후 재시뮬레이션 실패 — 다음 outer cycle 진행")
                else:
                    if notify:
                        await notify(f"{DAILY_TAG}[AUTO] ⚠️ YAML 수정 실패: {_yaml_summary[:200]}")
            except Exception as _yaml_exc:
                logger.warning("[AUTO] Fix E YAML feedback failed: %s", _yaml_exc)
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ⚠️ YAML 자동 피드백 오류: {_yaml_exc}")

        if notify and outer_cycle < _outer_max:
            await notify(
                f"{DAILY_TAG}[AUTO] 🔄 Phase C {_inner_max}회 소진 (avg {avg:.1f}) "
                f"— outer {outer_cycle + 1}/{_outer_max}로 수정된 코드로 재탐색"
            )

    return current_chapter

# ── User choice helpers ───────────────────────────────────────────────────────

def _parse_user_choice(text: str) -> str:
    """'1/코드', '2/스토리', '3/최적화', '4/그만두기' 중 하나를 반환."""
    t = text.strip().lower()
    if re.search(r"^1\b|코드|code|\.py|fixer", t):
        return "code"
    if re.search(r"^2\b|스토리|story|에피소드|config|yaml|야믈|캐릭터|플롯", t):
        return "story"
    if re.search(r"^3\b|최적화|optimize|optim|더\s*돌|추가.*사이클|사이클.*추가", t):
        return "optimize"
    if re.search(r"^4\b|그만|stop|끝|종료|다음|next|승인|ok|good|pass", t):
        return "next"
    return "other"


def _parse_extra_outer_cycles(text: str) -> int | None:
    """사용자 입력에서 추가 outer cycle 수를 파싱. 숫자 없으면 None."""
    m = re.search(r"\b([1-9][0-9]?)\b", text.strip())
    return int(m.group(1)) if m else None


def _fix_unclosed_yaml_quotes(text: str) -> str:
    """줄 끝에 닫히지 않은 이중따옴표를 닫아주는 최소 포맷 수정."""
    fixed: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n\r")
        # 이스케이프되지 않은 " 개수가 홀수이고, 값 시작에 " 가 있으면 닫기
        unescaped = stripped.count('"') - stripped.count('\\"')
        if unescaped % 2 == 1 and re.search(r':\s*"[^"]*$', stripped):
            eol = line[len(stripped):]
            stripped = stripped + '"'
            line = stripped + eol
        fixed.append(line)
    return "".join(fixed)


def _validate_and_fix_episode_yamls(episode_dir: Path) -> dict:
    """
    config/episodes/ 의 ep*.yaml 파일을 전수 검사한다.
    - 파싱 오류 파일은 _fix_unclosed_yaml_quotes 로 최소 포맷 수정 후 재검증.
    - 내용(content) 변경 없이 포맷(따옴표 누락)만 수정.
    반환: {"ok": int, "fixed": [str], "errors": [str]}
    """
    import yaml as _yaml

    ok_names: list[str] = []
    fixed_names: list[str] = []
    error_names: list[str] = []

    for f in sorted(episode_dir.glob("ep*.yaml")):
        raw = f.read_text(encoding="utf-8")
        try:
            data = _yaml.safe_load(raw)
            if data and "episode" in data:
                ok_names.append(f.name)
            else:
                error_names.append(f"{f.name}: `episode:` 키 없음")
        except _yaml.YAMLError:
            patched = _fix_unclosed_yaml_quotes(raw)
            if patched != raw:
                try:
                    data2 = _yaml.safe_load(patched)
                    if data2 and "episode" in data2:
                        f.write_text(patched, encoding="utf-8")
                        fixed_names.append(f.name)
                        continue
                except _yaml.YAMLError:
                    pass
            error_names.append(f"{f.name}: YAML 파싱 오류 (수동 수정 필요)")

    return {"ok": len(ok_names), "fixed": fixed_names, "errors": error_names}


def _build_yaml_auto_feedback(review_json: dict) -> str:
    """Fix E: Build an auto-feedback string for the YAML story fixer from AI review scores.

    Translates numeric quality scores into actionable Korean narrative directives.
    Used to automatically patch the episode YAML when code fixes alone can't
    raise quality past the threshold.
    """
    lines: list[str] = []
    score_map = {
        "thrill":        ("긴장감/스릴", "장면에 극적 갈등과 압박을 추가하라. 위기감이 더 명확히 드러나도록 beat를 조정하라."),
        "style":         ("문체/스타일", "장면 분위기 묘사(atmosphere)와 캐릭터 내면 반응을 더 구체적으로 적어라."),
        "causality":     ("인과관계", "사건 순서와 동기를 명확히 하라. 각 scene_beat에 원인-결과 연결을 강화하라."),
        "character":     ("캐릭터", "주인공의 감정선 변화와 다른 캐릭터와의 관계 변화를 scene_beats에 명시하라."),
        "scene_fn":      ("장면 기능", "각 scene_beat의 목적(정보 전달/감정 상승/반전)을 더 명확히 구분하라."),
    }
    for key, (label, directive) in score_map.items():
        score = float(review_json.get(key, 10))
        if score < 6.5:
            lines.append(f"[{label} 점수={score:.1f}/10] {directive}")
    if not lines:
        # All scores are decent — push for atmosphere depth
        lines.append("전반적 품질 향상을 위해 핵심 장면의 감각적 세부 묘사와 긴장감 곡선을 강화하라.")
    return "\n".join(lines)


def _build_story_fixer_prompt(episode_key: str, user_feedback: str) -> str:
    ep_file = resolve_episode_file(episode_key)
    return (
        f"사용자 피드백을 바탕으로 에피소드 config를 수정하라.\n\n"
        f"수정 대상 파일: {ep_file}\n\n"
        f"사용자 피드백:\n{user_feedback}\n\n"
        "규칙:\n"
        "1. 해당 에피소드 YAML 파일만 수정하라\n"
        "2. 다른 에피소드 파일과 .py 파일은 절대 건드리지 마라\n"
        "3. scene_beats, characters, key_events, atmosphere 위주로 반영하라\n"
        "4. 수정 내용을 한국어로 요약하라 (어떤 항목을 왜 바꿨는지)\n"
        "5. 확인 질문 없이 바로 수정하라"
    )


async def _run_gpt_story_fixer(
    episode_key: str,
    user_feedback: str,
    run_dir: Path,
    fixer_cycle: int,
    model: str = "gpt-4.1",
) -> tuple[bool, str]:
    """GPT API를 사용해 에피소드 YAML 수정 (Codex CLI 대체). 성공 여부와 요약 반환."""
    from openai import OpenAI

    ep_file = resolve_episode_file(episode_key)
    try:
        yaml_content = ep_file.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"에피소드 파일 읽기 실패: {exc}"

    summary_path = run_dir / f"story_fixer_cycle{fixer_cycle}_summary.md"

    system_msg = (
        "당신은 소설 에피소드 구성 전문가입니다.\n"
        "사용자 피드백을 바탕으로 에피소드 YAML을 수정하고 JSON으로 반환하세요.\n\n"
        "반환 형식 (순수 JSON만):\n"
        "{\n"
        '  "content": "수정된 전체 YAML 파일 내용",\n'
        '  "summary": "수정 내용 한국어 요약 (어떤 항목을 왜 바꿨는지)"\n'
        "}\n\n"
        "규칙:\n"
        "- scene_beats, characters, key_events, atmosphere 위주로 반영\n"
        "- YAML 구조와 들여쓰기를 유지\n"
        "- 다른 파일은 절대 건드리지 말 것"
    )
    user_msg = (
        f"사용자 피드백:\n{user_feedback}\n\n"
        f"현재 에피소드 파일 ({ep_file.name}):\n```yaml\n{yaml_content}\n```"
    )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=8000,
            ),
        )
        data = json.loads(response.choices[0].message.content or "{}")
        new_content = data.get("content", "")
        summary = data.get("summary", "")

        if not new_content:
            return False, "GPT가 수정된 YAML을 반환하지 않음"

        ep_file.write_text(new_content, encoding="utf-8")
        try:
            summary_path.write_text(summary, encoding="utf-8")
        except Exception:
            pass
        return True, summary

    except Exception as exc:
        return False, f"GPT Story Fixer 오류: {exc}"


async def _run_story_fixer(
    episode_key: str,
    user_feedback: str,
    run_dir: Path,
    fixer_cycle: int,
    set_process: ProcessFn = None,
    codex_model: str | None = None,
) -> tuple[bool, str]:
    """Codex로 에피소드 YAML 수정. Codex 없으면 GPT API로 대체. 성공 여부와 요약 반환."""
    summary_path = run_dir / f"story_fixer_cycle{fixer_cycle}_summary.md"
    prompt = _build_story_fixer_prompt(episode_key, user_feedback)
    cmd = _resolve_codex_exec_cmd(codex_model=codex_model)
    if not cmd:
        gpt_model = "gpt-4.1-mini" if (codex_model and "mini" in codex_model) else "gpt-4.1"
        return await _run_gpt_story_fixer(
            episode_key, user_feedback, run_dir, fixer_cycle, model=gpt_model,
        )
    cmd += [
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        "-o", str(summary_path),
        prompt,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return False, "Codex 실행 파일을 찾을 수 없음"
    if set_process:
        set_process("codex_story_fixer", proc.pid, " ".join(cmd))
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1200)
    except asyncio.TimeoutError:
        proc.terminate()
        await proc.wait()
        return False, "Story fixer 타임아웃 (20분)"
    finally:
        if set_process:
            set_process(None, None, None)
    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    if proc.returncode != 0:
        return False, f"Codex 실패 (rc={proc.returncode})\n{output[-800:]}"
    summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else output[-1000:]
    return True, summary


async def _git_commit_story_fix(episode_key: str, summary: str) -> tuple[bool, str]:
    """스토리 YAML 수정 후 git commit."""
    ep_file = resolve_episode_file(episode_key)
    proc_add = await asyncio.create_subprocess_exec(
        "git", "add", str(ep_file), cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    await proc_add.wait()
    proc_diff = await asyncio.create_subprocess_exec(
        "git", "diff", "--cached", "--quiet", cwd=str(REPO_ROOT),
    )
    await proc_diff.wait()
    if proc_diff.returncode == 0:
        return False, "변경된 내용 없음"
    commit_msg = (
        f"auto: story config update — {episode_key}\n\n"
        f"{summary[:200].replace(chr(10), ' ')}\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    proc_commit = await asyncio.create_subprocess_exec(
        "git", "commit", "-m", commit_msg, cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc_commit.communicate()
    out = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
    if proc_commit.returncode != 0:
        return False, f"git commit 실패: {out}"
    return True, out


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
            "이제 읽어보시고 자유롭게 피드백 남겨주세요.\n"
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
    review_tier: str = "premium",
    feedback_queue: asyncio.Queue | None = None,
    feedback_timeout_hours: float = 24.0,
    notify: NotifyFn = None,
    upload: UploadFn = None,
    no_discord: bool = False,
    stop_event: asyncio.Event | None = None,
    set_status: StatusFn = None,
    set_process: ProcessFn = None,
    set_metrics: MetricsFn = None,
    on_start_wait: Callable[[], None] | None = None,
    on_end_wait: Callable[[], None] | None = None,
    reset_emotions: bool = False,
    outer_max_cycles: int | None = None,
) -> dict[str, Any]:
    review_tier = str(review_tier or "premium").strip()
    if not review_tier:
        review_tier = "premium"
    load_project_env(REPO_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve episode key (support bare numbers)
    ep_file = resolve_episode_file(episode_key)
    episode_key = ep_file.stem  # normalise to full key e.g. "ep01_academic_presentation"

    reset_summary = _prepare_episode_restart_state(episode_key)

    cycle = _get_cycle_number(episode_key)
    run_dir = _allocate_daily_output_dir(episode_key)
    pipeline_start = time.monotonic()
    cost_tracker = {
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
    time_tracker: dict[str, float] = {}
    if set_metrics:
        set_metrics(dict(cost_tracker))

    async def _stop_and_return(step_label: str, step_name: str) -> dict[str, Any]:
        if set_status:
            set_status(f"중단됨 ({step_label})")
        await _notify_stop_usage_summary(
            notify,
            pipeline_start,
            cost_tracker,
            step_label=step_label,
        )
        return {"success": False, "step": step_name, "cycle": cycle, "stopped": True}

    if set_status:
        set_status(f"시작 — {episode_key} 사이클 {cycle}")
    if reset_summary.get("changed"):
        archived_episode_ids = reset_summary.get("db", {}).get("episode_ids", [])
        removed_story_keys = reset_summary.get("story_state", {}).get("removed_episode_keys", [])
        if notify:
            await notify(f"{DAILY_TAG}[RESET] ♻️ `{episode_key}` 재생성을 위해 기존 데이터를 백업했습니다.")
            thread_lines = [
                f"백업 위치: `{Path(reset_summary['backup_dir']).relative_to(REPO_ROOT)}`",
            ]
            if archived_episode_ids:
                thread_lines.append(
                    "DB 아카이브: " + ", ".join(f"`{episode_id}`" for episode_id in archived_episode_ids)
                )
            if removed_story_keys:
                thread_lines.append(
                    "story_state 정리: " + ", ".join(f"`{key}`" for key in removed_story_keys)
                )
            await notify(f"{DAILY_TAG}[RESET] 🗂️ " + "\n".join(thread_lines))
    if notify:
        await notify(f"{DAILY_TAG}[START] 🎬 `{episode_key}` 파이프라인 시작 (사이클 {cycle})")
        await notify(
            f"{DAILY_TAG}[START] run: `{run_dir.relative_to(REPO_ROOT)}`\n"
            f"리뷰 등급: `{review_tier}`\n"
            f"outer cycles: `{int(outer_max_cycles or AUTO_OUTER_MAX_CYCLES)}`"
        )

    # ── Step 1: Config Guardian ──
    _t0 = time.monotonic()
    ok1, guardian_briefing_path = await step_guardian(
        episode_key, run_dir, cycle, notify, upload, set_status, stop_event, review_tier,
        cost_tracker=cost_tracker,
        metrics=cost_tracker,
    )
    time_tracker["guardian"] = time.monotonic() - _t0
    if not ok1:
        if stop_event and stop_event.is_set():
            return await _stop_and_return("Guardian 단계", "guardian")
        if set_status:
            set_status("중단됨 (Guardian 단계)")
        return {"success": False, "step": "guardian", "cycle": cycle}

    # ── Step 2: Simulator ──
    _t0 = time.monotonic()
    ok2 = await step_simulator(episode_key, run_dir, cycle, budget, notify, set_status, stop_event,
                               set_process=set_process,
                               cost_tracker=cost_tracker,
                               metrics=cost_tracker,
                               guardian_briefing_path=guardian_briefing_path,
                               reset_emotions=reset_emotions)
    if set_metrics:
        set_metrics(dict(cost_tracker))
    if not ok2:
            if stop_event and stop_event.is_set():
                return await _stop_and_return("Simulator 단계", "simulator")
            if set_status:
                set_status("중단됨 (Simulator 단계)")
            return {"success": False, "step": "simulator", "cycle": cycle}
    time_tracker["simulator"] = time.monotonic() - _t0
    if notify:
        _sim_min = int(time_tracker["simulator"] // 60)
        _sim_sec = int(time_tracker["simulator"] % 60)
        await notify(f"{DAILY_TAG}[SIM] ⏱️ 소요 시간: {_sim_min}분 {_sim_sec:02d}초")

    # ── Step 3: Chapter generation ──
    _t0 = time.monotonic()
    # Generate one baseline chapter; Phase A (25-trial batch) will supersede it
    chapter_path = await step_chapter_gen(
            episode_key, run_dir, cycle, target_words, budget, protagonist,
            notify, upload, set_status, stop_event,
            set_process=set_process,
            cost_tracker=cost_tracker,
            metrics=cost_tracker,
            guardian_briefing_path=guardian_briefing_path,
            review_tier=review_tier,
        )
    if set_metrics:
        set_metrics(dict(cost_tracker))
    if chapter_path is None:
            if stop_event and stop_event.is_set():
                return await _stop_and_return("Chapter Gen 단계", "chapter_gen")
            if set_status:
                set_status("중단됨 (Chapter Gen 단계)")
            return {"success": False, "step": "chapter_gen", "cycle": cycle}
    time_tracker["chapter_gen"] = time.monotonic() - _t0
    if notify:
        _ch_min = int(time_tracker["chapter_gen"] // 60)
        _ch_sec = int(time_tracker["chapter_gen"] % 60)
        await notify(f"{DAILY_TAG}[CHAPTER] ⏱️ 소요 시간: {_ch_min}분 {_ch_sec:02d}초")

    # ── Step 3.5: AI 자동 개선 루프 (AI 리뷰 → Codex Fixer → 챕터 재생성) ──
    if notify:
        await notify(
            f"{DAILY_TAG}[AUTO] 🚀 AI 자동 개선 루프 시작 "
            f"(outer 최대 {outer_max_cycles or AUTO_OUTER_MAX_CYCLES}회, 목표 {AUTO_IMPROVE_SCORE_THRESHOLD}/10)"
        )
    _t0 = time.monotonic()
    chapter_path = await step_auto_improve_loop(
        episode_key, run_dir, chapter_path, target_words, budget, protagonist,
        guardian_briefing_path=guardian_briefing_path,
        notify=notify,
        upload=upload,
        set_status=set_status,
        stop_event=stop_event,
        set_process=set_process,
        review_tier=review_tier,
        cost_tracker=cost_tracker,
        metrics=cost_tracker,
        daily_cycle=cycle,
        outer_max_cycles=outer_max_cycles,
    )
    time_tracker["auto_improve"] = time.monotonic() - _t0
    if set_metrics:
        set_metrics(dict(cost_tracker))

    # ── Save quality review scores → data/quality_review_latest.json + policy log ──
    try:
        from tools.inline_optimizer import update_policy_log_quality_scores, POLICY_SCORE_LOG as _PSL
        _review_files = sorted(run_dir.glob("auto_review_cycle*.json"), key=lambda p: p.stat().st_mtime)
        if _review_files:
            _final_review = json.loads(_review_files[-1].read_text(encoding="utf-8"))
            _q_scores = {
                "thrill":         _final_review.get("thrill_score_10", 0),
                "style":          _final_review.get("style_score_10", 0),
                "causality":      _final_review.get("causality_score_10", 0),
                "character":      _final_review.get("character_score_10", 0),
                "scene_function": _final_review.get("scene_function_score_10", 0),
            }
            _qrl_path = REPO_ROOT / "data" / "quality_review_latest.json"
            _qrl_path.write_text(
                json.dumps({"episode_id": episode_key, "date": str(date.today()), "scores": _q_scores},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            update_policy_log_quality_scores(_q_scores, _PSL)
            logger.info("[QUALITY_FEEDBACK] Saved quality scores for next episode: %s", _q_scores)
    except Exception as exc:
        logger.warning("[QUALITY_FEEDBACK] Failed to save quality scores: %s", exc)

    # ── Step 4: Final quality review → user ──
    _t0 = time.monotonic()
    scorecard = await step_quality_review(
        episode_key, chapter_path, run_dir, cycle, notify, upload, set_status, stop_event,
        review_tier=review_tier,
        cost_tracker=cost_tracker,
        metrics=cost_tracker,
    )
    time_tracker["quality_review"] = time.monotonic() - _t0
    if set_metrics:
        set_metrics(dict(cost_tracker))
    if scorecard is None:
        if stop_event and stop_event.is_set():
            return await _stop_and_return("Final Review 단계", "quality_review")
        if set_status:
            set_status("중단됨 (Final Review 단계)")
        return {"success": False, "step": "quality_review", "cycle": cycle}

    _increment_cycle(episode_key)

    # ── 품질 차트 생성 + Discord 전송 ──
    chart_path = await asyncio.to_thread(
        _generate_quality_chart, episode_key, run_dir, cost_tracker, time_tracker, review_tier
    )
    if chart_path and upload:
        try:
            await upload(chart_path, f"📊 [{episode_key}] 사이클 {cycle} 품질 리포트")
        except Exception:
            pass
    elif chart_path and notify:
        try:
            await notify(f"{DAILY_TAG}[REPORT] 📊 품질 리포트 저장 → `{chart_path.relative_to(REPO_ROOT)}`")
        except Exception:
            pass

    if no_discord or feedback_queue is None:
        if set_status:
            set_status("완료 (no-discord)")
        if notify:
            _elapsed_line, _token_line, _cost_line = _build_final_usage_lines(pipeline_start, cost_tracker)
            await notify(
                f"{DAILY_TAG}[DONE] ✅ 파이프라인 완료 (no-discord 모드)\n"
                f"- chapter: `{chapter_path.relative_to(REPO_ROOT)}`\n"
                f"{_elapsed_line}\n"
                f"{_token_line}\n"
                f"{_cost_line}"
            )
        return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": None, "feedback": None}

    # ── Step 5: 선택지 메뉴 루프 ─────────────────────────────────────────────
    # 1=코드수정, 2=스토리수정, 3=최적화 추가, 4=그만두기
    # 1/2 완료 후 다시 이 메뉴로 복귀. 4 선택 시 종료.
    raw_feedback: str | None = None
    choice: str = "other"

    async def _show_choice_menu() -> None:
        if notify:
            await notify(
                f"{DAILY_TAG}[CHOICE] 📋 **개선 방향을 선택해주세요.**\n\n"
                "**1️⃣ 코드 수정** — Codex가 소설 생성 .py 파일을 자동 수정 후 챕터 재생성\n"
                "**2️⃣ 스토리 수정** — 에피소드 config YAML을 Codex로 직접 수정\n"
                "**3️⃣ 최적화 추가** — 파라미터 탐색 + Codex fix를 N회 더 실행\n"
                "**4️⃣ 그만두기** — 현재 챕터로 마무리\n\n"
                "번호 + 구체적인 의견을 같이 적으면 더 잘 반영됩니다.\n"
                "예: `1 대사가 너무 딱딱해` / `2 수민이 너무 수동적으로 나와`"
            )
        if set_status:
            set_status("선택 대기 중 (1=코드 / 2=스토리 / 3=최적화 / 4=종료)")

    await _show_choice_menu()

    while True:
        if on_start_wait:
            on_start_wait()

        raw_feedback = await wait_for_feedback(
            feedback_queue, feedback_timeout_hours, notify, stop_event,
        )

        if on_end_wait:
            on_end_wait()

        if raw_feedback is None:
            if stop_event and stop_event.is_set():
                return await _stop_and_return("피드백 대기", "feedback_wait")
            if set_status:
                set_status("완료 (응답 없음)")
            if notify:
                _elapsed_line, _token_line, _cost_line = _build_final_usage_lines(pipeline_start, cost_tracker)
                await notify(
                    f"{DAILY_TAG}[DONE] ℹ️ 응답이 없어 여기서 마무리했습니다.\n"
                    "다시 실행하려면 `!novel-daily <번호>`\n"
                    f"{_elapsed_line}\n{_token_line}\n{_cost_line}"
                )
            return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": None, "feedback": None}

        choice = _parse_user_choice(raw_feedback)

        # ── Step 5a: 코드 수정 ──
        if choice == "code":
            if set_status:
                set_status("코드 수정 중 (Codex)...")
            if notify:
                await notify(f"{DAILY_TAG}[CHOICE] 1️⃣ 코드 수정 선택 — Codex Fixer 실행 중...")

            backup_dir = await asyncio.to_thread(_backup_target_files, run_dir, cycle)
            if notify:
                await notify(f"{DAILY_TAG}[CHOICE] 💾 이전 버전 백업 → `{backup_dir.name}/`")

            user_issues = raw_feedback.strip()
            fixer_prompt = (
                f"사용자 피드백: {user_issues}\n\n" + _build_codex_fixer_prompt({
                    "what_felt_boring_or_hard": [user_issues],
                    "style_tips": [],
                    "reader_comment": user_issues,
                    "thrill_score_10": "?",
                    "style_score_10": "?",
                })
            )
            ok, summary = await _run_codex_fixer(
                fixer_prompt, run_dir, cycle,
                set_process=set_process, notify=notify, stop_event=stop_event,
                codex_model=_codex_model_for_tier(review_tier),
            )
            if ok:
                if notify:
                    await notify(f"{DAILY_TAG}[CHOICE] ✅ 코드 수정 완료:\n{summary}")
                committed, _ = await _git_commit_fixer_changes(cycle, episode_key, summary)
                if committed and notify:
                    await notify(f"{DAILY_TAG}[CHOICE] 📦 git commit 완료")
                if notify:
                    await notify(f"{DAILY_TAG}[CHOICE] 📖 수정된 코드로 챕터 재생성 중...")
                new_chapter = await step_chapter_gen(
                    episode_key, run_dir, cycle, target_words, budget, protagonist,
                    notify=notify, upload=upload, set_status=set_status,
                    stop_event=stop_event, set_process=set_process, cost_tracker=cost_tracker,
                    metrics=cost_tracker,
                    upload_version_label=f"choice_code_cycle{cycle}",
                    guardian_briefing_path=guardian_briefing_path,
                    review_tier=review_tier,
                )
                if new_chapter:
                    chapter_path = new_chapter
                    if set_metrics:
                        set_metrics(dict(cost_tracker))
            else:
                if notify:
                    await notify(f"{DAILY_TAG}[CHOICE] ❌ 코드 수정 실패: {summary}")
            await _show_choice_menu()
            continue

        # ── Step 5b: 스토리 수정 ──
        elif choice == "story":
            if set_status:
                set_status("스토리 수정 중 (Codex)...")
            if notify:
                await notify(f"{DAILY_TAG}[CHOICE] 2️⃣ 스토리 수정 선택 — 에피소드 YAML 수정 중...")

            ok, summary = await _run_story_fixer(
                episode_key,
                raw_feedback,
                run_dir,
                cycle,
                set_process=set_process,
                codex_model=_codex_model_for_tier(review_tier),
            )
            if ok:
                if notify:
                    await notify(f"{DAILY_TAG}[CHOICE] ✅ 스토리 수정 완료:\n{summary}")

                # ── YAML 검수 단계 ──
                ep_dir = REPO_ROOT / "config" / "episodes"
                if notify:
                    await notify(f"{DAILY_TAG}[FIXER] 🔍 YAML 검수 시작 — 에피소드 파일 전수 검사 중...")
                if set_status:
                    set_status("YAML 검수 중...")
                yaml_result = await asyncio.to_thread(_validate_and_fix_episode_yamls, ep_dir)
                if yaml_result["fixed"]:
                    fixed_list = ", ".join(yaml_result["fixed"])
                    if notify:
                        await notify(
                            f"{DAILY_TAG}[FIXER] 🔍 포맷 수정 완료: {fixed_list}\n"
                            f"(따옴표 누락 자동 수정 — 내용 변경 없음)"
                        )
                if yaml_result["errors"]:
                    err_list = "\n".join(f"  • {e}" for e in yaml_result["errors"])
                    if notify:
                        await notify(
                            f"{DAILY_TAG}[FIXER] ⚠️ YAML 검수 — 수동 수정 필요:\n{err_list}"
                        )
                else:
                    if notify:
                        await notify(
                            f"{DAILY_TAG}[FIXER] ✅ YAML 검수 완료 "
                            f"({yaml_result['ok']}개 정상"
                            + (f", {len(yaml_result['fixed'])}개 포맷 수정됨" if yaml_result["fixed"] else "")
                            + ")"
                        )

                committed, _ = await _git_commit_story_fix(episode_key, summary)
                if committed and notify:
                    await notify(f"{DAILY_TAG}[CHOICE] 📦 git commit 완료 — 다음 `!novel-daily`에 반영됩니다")
            else:
                if notify:
                    await notify(f"{DAILY_TAG}[CHOICE] ❌ 스토리 수정 실패: {summary}")
            await _show_choice_menu()
            continue

        # ── Step 5c: 최적화 추가 ──
        elif choice == "optimize":
            if notify:
                await notify(
                    f"{DAILY_TAG}[CHOICE] 3️⃣ 최적화 추가 — 몇 회 더 돌릴까요?\n\n"
                    f"숫자만 입력하거나 `3 2` 처럼 번호 뒤에 붙여도 됩니다.\n"
                    f"예: `2` → outer 2회 추가 실행\n\n"
                    f"(현재 목표 {score_threshold}/10, Phase A {AUTO_BATCH_TRIALS} trials × N회 + Codex fix)"
                )
            if set_status:
                set_status("추가 최적화 횟수 대기 중...")

            if on_start_wait:
                on_start_wait()
            _extra_raw = await wait_for_feedback(feedback_queue, feedback_timeout_hours, notify, stop_event)
            if on_end_wait:
                on_end_wait()

            _extra_cycles = _parse_extra_outer_cycles(_extra_raw or "") if _extra_raw else None
            if _extra_cycles is None:
                _extra_cycles = AUTO_OUTER_MAX_CYCLES  # default if no number given
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] 🚀 AI 자동 개선 루프 시작 "
                    f"(outer 최대 {_extra_cycles}회, 목표 {score_threshold}/10)"
                )
            chapter_path = await step_auto_improve_loop(
                episode_key, run_dir, chapter_path, target_words, budget, protagonist,
                guardian_briefing_path=guardian_briefing_path,
                notify=notify, upload=upload, set_status=set_status,
                stop_event=stop_event, set_process=set_process,
                review_tier=review_tier,
                cost_tracker=cost_tracker, metrics=cost_tracker,
                daily_cycle=cycle,
                outer_max_cycles=_extra_cycles,
            )
            if set_metrics:
                set_metrics(dict(cost_tracker))
            await _show_choice_menu()
            continue

        # ── Step 5d: 그만두기(4/next) / 기타 → 루프 탈출, Step 6으로 ──
        else:
            break  # while 루프 종료

    # ── Step 6: 피드백 파싱 + story_state 업데이트 ──
    if set_status:
        set_status("피드백 분석 중...")
    llm = LLMClient(
        model="gpt-4o-mini", premium_model=_llm_premium_model_for_tier(review_tier), budget_usd=1.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    with ep_file.open(encoding="utf-8") as f:
        episode_data = (yaml.safe_load(f) or {}).get("episode", {})

    parsed = parse_feedback_with_llm(raw_feedback, episode_key, llm)
    _record_budget_usage(cost_tracker, cost_tracker, llm.budget_summary(), cost_key="feedback_parse")
    if set_metrics:
        set_metrics(dict(cost_tracker))
    if choice == "next":
        parsed["approved_next_episode"] = True
    elif choice in ("code", "story", "optimize"):
        parsed["approved_next_episode"] = False
    update_story_state(STORY_STATE_PATH, episode_key, episode_data, parsed)

    approved = parsed.get("approved_next_episode", False)
    if set_status:
        set_status(f"완료 — {'승인됨' if approved else '재시도 예정'}")
    if notify:
        choice_label = {"code": "코드 수정", "story": "스토리 수정", "next": "그만두기", "optimize": "추가 최적화", "other": "피드백 저장"}.get(choice, "완료")
        _elapsed_line, _token_line, _cost_line = _build_final_usage_lines(pipeline_start, cost_tracker)
        if approved:
            await notify(
                f"{DAILY_TAG}[DONE] ✅ {choice_label} 완료. 다음 에피소드로 이동하려면: `!novel-daily <번호>`\n"
                f"{_elapsed_line}\n"
                f"{_token_line}\n"
                f"{_cost_line}"
            )
        else:
            issues = parsed.get("specific_issues", [])
            issue_str = "\n".join(f"  - {i}" for i in issues) if issues else "  (코멘트 참조)"
            await notify(
                f"{DAILY_TAG}[DONE] 📝 {choice_label} 완료. 같은 화 재시도: `!novel-daily {episode_key}`\n"
                f"개선 포인트:\n{issue_str}\n"
                f"{_elapsed_line}\n"
                f"{_token_line}\n"
                f"{_cost_line}"
            )

    return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": approved, "feedback": parsed, "choice": choice}


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Pipeline")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--target-words", type=int, default=3500)
    parser.add_argument("--budget", type=float, default=4.0)
    parser.add_argument("--protagonist", default="kim_sumin")
    parser.add_argument("--outer-cycles", type=int, default=AUTO_OUTER_MAX_CYCLES)
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    async def _run():
        result = await run_daily_pipeline(
            episode_key=args.episode,
            target_words=args.target_words,
            budget=args.budget,
            protagonist=args.protagonist,
            no_discord=args.no_discord,
            outer_max_cycles=args.outer_cycles,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    main()
