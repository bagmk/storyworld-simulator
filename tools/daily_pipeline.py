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
from datetime import datetime
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

# Auto-improve loop settings
AUTO_IMPROVE_MAX_CYCLES = 20      # Codex fixer 최대 반복 횟수
AUTO_IMPROVE_SCORE_THRESHOLD = 8.5  # thrill+style 평균 이 점수 이상이면 통과 (10점 만점)
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
    """mini 티어이면 gpt-5.1-codex, 그 외엔 None (config.toml 기본값 사용)."""
    return "gpt-5.1-codex-mini" if _use_mini_review_tier(review_tier) else None


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
    - 하단: LLM 비용 파이 차트
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
    has_model_row = has_model_times or has_model_tokens
    n_rows = (
        (1 if has_scores else 0)
        + (1 if has_costs else 0)
        + (1 if has_times else 0)
        + (1 if has_model_row else 0)
    )
    if n_rows == 0:
        return None

    fig = plt.figure(figsize=(12 if has_model_times and has_model_tokens else 9, 4 * n_rows))
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

    if has_costs:
        ax = fig.add_subplot(grid[ax_idx]); ax_idx += 1
        labels_nz = [l for l, _ in cost_vals_nonzero]
        vals_nz = [v for _, v in cost_vals_nonzero]
        total_cost = sum(vals_nz)
        cat_colors = {"시뮬레이션": "#3498db", "챕터생성": "#e74c3c", "리뷰": "#2ecc71"}
        colors = [cat_colors.get(l, "#9b59b6") for l in labels_nz]
        wedges, texts, autotexts = ax.pie(
            vals_nz, labels=labels_nz, autopct="%1.1f%%",
            colors=colors, startangle=140, pctdistance=0.75,
        )
        for t in texts:
            t.set_fontsize(10)
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(f"LLM 비용 구성 (총 ${total_cost:.4f}, Codex CLI 제외)", fontsize=11)

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

    if has_model_row:
        model_cols = (1 if has_model_times else 0) + (1 if has_model_tokens else 0)
        model_grid = grid[ax_idx].subgridspec(1, model_cols)
        model_ax_idx = 0
        model_colors = {
            "gpt-4o-mini":  "#3498db",
            "gpt-4.1-mini": "#1abc9c",
            "gpt-4o":       "#e74c3c",
            "gpt-5-mini":   "#9b59b6",
            "Codex CLI":    "#2ecc71",
        }
        if has_model_times:
            ax = fig.add_subplot(model_grid[0, model_ax_idx]); model_ax_idx += 1
            m_labels = [m for m, _ in model_time_nonzero]
            m_vals   = [t for _, t in model_time_nonzero]
            total_model_sec = sum(m_vals)
            m_colors = [model_colors.get(m, "#f39c12") for m in m_labels]
            wedges, texts, autotexts = ax.pie(
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
            ax = fig.add_subplot(model_grid[0, model_ax_idx]); model_ax_idx += 1
            tok_labels = [m for m, _ in model_token_nonzero]
            tok_vals = [t for _, t in model_token_nonzero]
            total_model_tokens = sum(tok_vals)
            tok_colors = [model_colors.get(m, "#f39c12") for m in tok_labels]
            wedges, texts, autotexts = ax.pie(
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

        briefing_path = run_dir / "guardian_gpt_report.txt"
        briefing_path.write_text(gpt_report, encoding="utf-8")

        if notify:
            await notify(f"{DAILY_TAG}[GUARDIAN] 🧠 GPT 분석 리포트:\n{gpt_report}")
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
        "python3", "-u", "simulate.py",
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
        "python3", "-u", "generate_chapter.py",
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

    cmd = [
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        prompt,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
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

    cmd = [
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        prompt,
    ]

    if notify:
        await notify(
            f"{DAILY_TAG}[ERROR-FIX] 🔧 `{failed_step}` 오류 감지 — Codex 자동 진단 시작..."
        )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
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
    return (
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
        "11. 검증이 실패하면 그대로 끝내지 말고, 실패 원인을 반영해 수정한 뒤 요약에 검증 결과를 함께 남겨라"
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

    return (
        f"당신은 소설 생성 AI의 수석 매니저다. 목표는 챕터 품질 점수를 8.5 이상으로 끌어올리는 것이다.\n"
        f"현재 상황: 일일 사이클 {daily_cycle}, 픽서 내부 사이클 {fixer_cycle}, {depth_label}\n\n"
        f"## {current_block}\n"
        f"{trend_block}"
        f"{stagnation_block}"
        f"{ineffective_block}"
        f"\n## 시작점 힌트 (휴리스틱)\n{heuristic_hints}\n"
        f"{history_block}"
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
    )
    # Factor Analysis 결과를 프롬프트 앞에 추가
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

    py_compile_cmd = ["python3", "-m", "py_compile"] + python_changed
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
        test_cmd = ["python3", "-m", "unittest", "tests.test_reader_feedback_guards"]
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
    return True, f"로컬 검증 통과 ({validation_log.name})"


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
    cmd = ["codex", "exec"]
    if codex_model:
        cmd += ["-m", codex_model]
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
) -> Path:
    """
    AI 리뷰 → Codex Fixer → 챕터 재생성 루프.
    점수 통과 또는 max_cycles 도달 시 최종 chapter_path 반환.
    """
    current_chapter = chapter_path

    for fixer_cycle in range(1, max_cycles + 1):
        if stop_event and stop_event.is_set():
            break

        # 롤백 시 비용 복원을 위해 사이클 시작 시점 스냅샷
        _cost_snapshot = copy.deepcopy(cost_tracker or {})

        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — 리뷰 중...")
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 🔄 AI 자동 개선 루프 {fixer_cycle}/{max_cycles} 시작")

        # ── AI 리뷰 ──
        chapter_text = current_chapter.read_text(encoding="utf-8", errors="replace")
        try:
            llm = LLMClient(
                model=_llm_review_model_for_tier(review_tier),
                premium_model="gpt-4o",
                budget_usd=2.0,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            _story_ctx = await asyncio.to_thread(_load_story_context_for_review)
            review_raw = await asyncio.to_thread(
                llm.chat,
                [{"role": "user", "content": _build_ai_reviewer_prompt(chapter_text, _story_ctx)}],
                use_premium=_use_premium_review_tier(review_tier),
                purpose="auto_improve_reviewer",
                max_tokens=1400,
            )
            cleaned = re.sub(r"```(?:json)?\n?", "", review_raw).strip().rstrip("`")
            review_json = json.loads(cleaned)
            review_budget = llm.budget_summary()
        except Exception as exc:
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] ⚠️ 리뷰 실패 ({exc}), 루프 종료")
            break

        _ALL_SCORE_KEYS = [
            "thrill_score_10", "style_score_10", "causality_score_10",
            "character_score_10", "scene_function_score_10",
        ]
        thrill   = int(review_json.get("thrill_score_10", 0))
        style    = int(review_json.get("style_score_10", 0))
        causality = int(review_json.get("causality_score_10", 0))
        character = int(review_json.get("character_score_10", 0))
        scene_fn  = int(review_json.get("scene_function_score_10", 0))
        # 항상 5개 기준으로 avg 계산 — 일부 키가 없으면 0으로 처리
        avg = sum(int(review_json.get(k, 0)) for k in _ALL_SCORE_KEYS) / len(_ALL_SCORE_KEYS)
        verdict = review_json.get("one_line_verdict", "")

        # 리뷰 저장 (codex 경로는 _run_codex_review에서 이미 저장하므로 GPT 경로만 덮어쓰기)
        review_path = run_dir / f"auto_review_cycle{fixer_cycle}.json"
        review_path.write_text(json.dumps(review_json, ensure_ascii=False, indent=2), encoding="utf-8")
        if cost_tracker is not None:
            cost_tracker["auto_review"] = cost_tracker.get("auto_review", 0.0) + float(review_budget.get("spent_usd", 0.0))
        _accumulate_usage_totals(metrics, review_budget)

        if notify:
            summary_budget_line = _format_budget_line("AI 리뷰 비용", review_budget)
            _review_mood = (
                "🏆" if avg >= score_threshold else
                "🌟" if avg >= 7.5 else
                "🎯" if avg >= 6.5 else
                "🧪" if avg >= 5.5 else
                "📉"
            )
            await notify(
                f"{DAILY_TAG}[AUTO] 📊 AI 리뷰 결과 (사이클 {fixer_cycle}) {_review_mood}\n"
                f"긴장감: {thrill}/10 | 문체: {style}/10 | 인과성: {causality}/10 | "
                f"캐릭터: {character}/10 | 씬기능: {scene_fn}/10\n"
                f"**평균: {avg:.1f}/10** | 목표: {score_threshold:.1f}/10"
            )
            await notify(
                f"{DAILY_TAG}[AUTO] 🧾 AI 리뷰 상세 (사이클 {fixer_cycle})\n"
                f"한줄평: {verdict}\n"
                f"좋았던 점:\n- " + "\n- ".join(review_json.get("what_felt_good", [])) + "\n"
                f"지루하거나 어려웠던 점:\n- " + "\n- ".join(review_json.get("what_felt_boring_or_hard", [])) + "\n"
                f"개선 팁:\n- " + "\n- ".join(review_json.get("style_tips", [])) + "\n"
                f"독자 코멘트: {review_json.get('reader_comment', '')}\n"
                f"{summary_budget_line}"
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

        # ── Manager Agent (novel-loop 구조 이식) ──
        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — 매니저 분석 중...")
        manager_instructions = await run_manager_agent(
            episode_key=episode_key,
            run_dir=run_dir,
            current_review=review_json,
            daily_cycle=daily_cycle,
            fixer_cycle=fixer_cycle,
            notify=notify,
            review_tier=review_tier,
            manager_period=manager_period,
            cost_tracker=cost_tracker,
            metrics=metrics,
        )

        # ── Codex Fixer ──
        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — Codex 코드 수정 중...")
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 🔧 Codex Fixer 실행 중... (코드 자동 수정)")

        # 수정 전 백업
        backup_dir = await asyncio.to_thread(_backup_target_files, run_dir, fixer_cycle)
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 💾 이전 버전 백업 완료 → `{backup_dir.name}/`")

        fixer_prompt = _build_codex_fixer_prompt(review_json, manager_instructions=manager_instructions)
        # 리뷰 진단 요약 → Reviewer 봇 스레드로 전송
        if notify:
            _t = review_json.get("thrill_score_10", "?")
            _s = review_json.get("style_score_10", "?")
            _c = review_json.get("causality_score_10", "?")
            _ch = review_json.get("character_score_10", "?")
            _sf = review_json.get("scene_function_score_10", "?")
            _issues_txt = "\n".join(f"- {i}" for i in review_json.get("what_felt_boring_or_hard", []))
            _tips_txt   = "\n".join(f"- {t}" for t in review_json.get("style_tips", []))
            await notify(
                f"{DAILY_TAG}[AUTO] 📋 Codex 수정 진단 (사이클 {fixer_cycle})\n"
                f"점수: 긴장감={_t}/10 | 문체={_s}/10 | 인과성={_c}/10 | 캐릭터={_ch}/10 | 씬기능={_sf}/10\n\n"
                f"**문제점:**\n{_issues_txt}\n\n"
                f"**개선 팁:**\n{_tips_txt}"
            )
        ok, summary = await _run_codex_fixer(
            fixer_prompt,
            run_dir,
            fixer_cycle,
            set_process=set_process,
            notify=notify,
            stop_event=stop_event,
            codex_model=_codex_model_for_tier(review_tier),
        )

        if not ok:
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] ❌ Codex Fixer 실패: {summary}")
            break

        changed_files = await asyncio.to_thread(_detect_changed_target_files, backup_dir)
        if notify:
            changed_lines = "\n".join(f"- {path}" for path in changed_files) if changed_files else "- (감지된 변경 파일 없음)"
            await notify(
                f"{DAILY_TAG}[AUTO] ✅ 코드 수정 완료:\n{summary[:600]}\n\n"
                f"이번 사이클 변경 파일:\n{changed_lines}"
            )

        if notify:
            await notify(
                f"{DAILY_TAG}[PROGRAMMER] 🧪 코드 검수 시작 (사이클 {fixer_cycle})\n"
                f"검수 대상:\n{changed_lines}"
            )

        validation_ok, validation_reason = await _run_local_fixer_validation(
            changed_files=changed_files,
            run_dir=run_dir,
            fixer_cycle=fixer_cycle,
            stop_event=stop_event,
            notify=notify,
        )
        if not validation_ok:
            if notify:
                await notify(
                    f"{DAILY_TAG}[PROGRAMMER] ⏪ 로컬 검증 실패 → 롤백\n"
                    f"사유: {validation_reason}"
                )
            restored = await asyncio.to_thread(_rollback_from_backup, backup_dir)
            if cost_tracker is not None:
                cost_tracker.update(_cost_snapshot)
            if notify:
                await notify(f"{DAILY_TAG}[PROGRAMMER] ↩️ 코드 롤백 완료: {', '.join(restored)}")
            break
        if notify:
            await notify(f"{DAILY_TAG}[PROGRAMMER] ✅ 로컬 검증 통과: {validation_reason}")

        # ── GPT 코드리뷰 + 롤백 ──
        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — Programmer 코드 검수 중...")
        if notify:
            await notify(f"{DAILY_TAG}[PROGRAMMER] 🕵️ 코드리뷰 중...")

        code_approved, review_reason = await _run_gpt_code_review(
            backup_dir=backup_dir,
            summary=summary,
            run_dir=run_dir,
            fixer_cycle=fixer_cycle,
            changed_files=changed_files,
            validation_summary=validation_reason,
            cost_tracker=cost_tracker,
            metrics=metrics,
        )

        if not code_approved:
            if notify:
                await notify(
                    f"{DAILY_TAG}[PROGRAMMER] ⏪ 코드리뷰 reject → 롤백\n"
                    f"사유: {review_reason}"
                )
            restored = await asyncio.to_thread(_rollback_from_backup, backup_dir)
            if cost_tracker is not None:
                cost_tracker.update(_cost_snapshot)
            if notify:
                await notify(f"{DAILY_TAG}[PROGRAMMER] ↩️ 롤백 완료: {', '.join(restored)}")
            break  # 롤백 후 루프 종료, 현재 챕터 유지

        if notify:
            await notify(f"{DAILY_TAG}[PROGRAMMER] ✅ 코드리뷰 통과: {review_reason}")

        # 수정 후 자동 git commit
        committed, commit_msg = await _git_commit_fixer_changes(fixer_cycle, episode_key, summary)
        if notify:
            if committed:
                await notify(f"{DAILY_TAG}[AUTO] 📦 git commit 완료 (사이클 {fixer_cycle}): `{commit_msg[:120]}`")
            else:
                await notify(f"{DAILY_TAG}[AUTO] ℹ️ git commit 스킵: {commit_msg}")

        if any(path in SIMULATION_RELEVANT_FIXER_FILES for path in changed_files):
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] 🔁 시뮬레이션 관련 코드 변경 감지 "
                    f"({fixer_cycle}/{max_cycles}) — 시뮬레이션부터 다시 검증합니다."
                )
            sim_ok = await step_simulator(
                episode_key=episode_key,
                run_dir=run_dir,
                cycle=daily_cycle,
                budget=budget,
                notify=notify,
                set_status=set_status,
                stop_event=stop_event,
                set_process=set_process,
                cost_tracker=cost_tracker,
                metrics=metrics,
                auto_cycle_index=fixer_cycle,
                auto_max_cycles=max_cycles,
                guardian_briefing_path=guardian_briefing_path,
            )
            if not sim_ok:
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ⚠️ 재시뮬레이션 실패 — AUTO 루프 종료")
                break

        cached_scenes_path: Path | None = None
        if changed_files and set(changed_files).issubset(SCENE_CACHE_SAFE_FIXER_FILES):
            candidate = run_dir / f"{resolve_episode_file(episode_key).stem}_scenes.json"
            if candidate.exists():
                cached_scenes_path = candidate
                if notify:
                    await notify(
                        f"{DAILY_TAG}[AUTO] ⚡ scene distill 캐시 재사용 — "
                        f"`{candidate.name}`로 장면 압축 단계를 건너뜁니다."
                    )

        # ── 챕터 재생성 ──
        if set_status:
            set_status(f"AI 개선 루프 {fixer_cycle}/{max_cycles} — 챕터 재생성 중...")
        if notify:
            await notify(f"{DAILY_TAG}[AUTO] 📖 수정된 코드로 챕터를 다시 생성합니다.")

        # auto_chapter 비용 격리 추적 — 재생성 전용 임시 tracker 사용
        _regen_cost_tracker: dict[str, float] = {"chapter": 0.0}
        new_chapter = await step_chapter_gen(
            episode_key, run_dir, daily_cycle, target_words, budget, protagonist,
            notify=notify,
            upload=upload,
            set_status=set_status,
            stop_event=stop_event,
            set_process=set_process,
            cost_tracker=_regen_cost_tracker,
            metrics=metrics,
            auto_cycle_index=fixer_cycle,
            auto_max_cycles=max_cycles,
            upload_version_label=f"auto_cycle{fixer_cycle}",
            precomputed_scenes_path=cached_scenes_path,
            guardian_briefing_path=guardian_briefing_path,
        )
        if cost_tracker is not None:
            cost_tracker["auto_chapter"] = (
                cost_tracker.get("auto_chapter", 0.0)
                + float(_regen_cost_tracker.get("chapter", 0.0))
            )
        if new_chapter is None:
            if notify:
                await notify(
                    f"{DAILY_TAG}[AUTO] ⚠️ 챕터 재생성 실패 또는 중단 — AUTO 루프 종료"
                )
            break

        if new_chapter:
            # ── 재생성 후 점수 비교 → 하락 시 롤백 ──
            new_chapter_text = new_chapter.read_text(encoding="utf-8", errors="replace")
            try:
                llm_check = LLMClient(
                    model="gpt-4o-mini", premium_model=_llm_premium_model_for_tier(review_tier), budget_usd=1.0,
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                )
                _story_ctx_check = await asyncio.to_thread(_load_story_context_for_review)
                check_raw = await asyncio.to_thread(
                    llm_check.chat,
                    [{"role": "user", "content": _build_ai_reviewer_prompt(new_chapter_text, _story_ctx_check)}],
                    use_premium=False,
                    purpose="regen_score_check",
                    max_tokens=1000,
                )
                check_cleaned = re.sub(r"```(?:json)?\n?", "", check_raw).strip().rstrip("`")
                check_json = json.loads(check_cleaned)
                _ck_keys = ["thrill_score_10", "style_score_10", "causality_score_10",
                            "character_score_10", "scene_function_score_10"]
                new_avg = sum(int(check_json.get(k, 0)) for k in _ck_keys) / len(_ck_keys)
                _record_budget_usage(cost_tracker, metrics, llm_check.budget_summary(), cost_key="regen_check")
            except Exception as _check_exc:
                if "llm_check" in locals():
                    _record_budget_usage(cost_tracker, metrics, llm_check.budget_summary(), cost_key="regen_check")
                if notify:
                    await notify(
                        f"{DAILY_TAG}[AUTO] ⚠️ 재생성 점수 체크 실패 ({type(_check_exc).__name__}: {_check_exc}) "
                        "— 이전 점수 유지, 롤백 생략"
                    )
                new_avg = avg  # 체크 실패 시 현재 점수로 간주 (롤백 방지)

            if new_avg < avg - 0.5:
                # 점수 하락 → 코드 롤백 + 이전 챕터 유지
                if notify:
                    await notify(
                        f"{DAILY_TAG}[AUTO] ⏪ 재생성 후 점수 하락 ({avg:.1f} → {new_avg:.1f}) → 롤백"
                    )
                restored = await asyncio.to_thread(_rollback_from_backup, backup_dir)
                if cost_tracker is not None:
                    cost_tracker.update(_cost_snapshot)
                if notify:
                    await notify(f"{DAILY_TAG}[AUTO] ↩️ 코드 롤백 완료: {', '.join(restored)}")
                break
            else:
                current_chapter = new_chapter
                avg = new_avg  # 다음 사이클 비교를 위해 업데이트
                if notify:
                    wc = len(new_chapter_text.split())
                    await notify(f"{DAILY_TAG}[AUTO] 📝 재생성 완료 ({wc}단어, 점수 {avg:.1f}/10)")
        else:
            if notify:
                await notify(f"{DAILY_TAG}[AUTO] ⚠️ 챕터 재생성 실패 — 이전 버전 유지")
            break

    return current_chapter


# ── User choice helpers ───────────────────────────────────────────────────────

def _parse_user_choice(text: str) -> str:
    """'1/코드', '2/스토리', '3/다음' 중 하나를 반환."""
    t = text.strip().lower()
    if re.search(r"^1\b|코드|code|\.py|fixer", t):
        return "code"
    if re.search(r"^2\b|스토리|story|에피소드|config|yaml|야믈|캐릭터|플롯", t):
        return "story"
    if re.search(r"^3\b|다음|next|승인|좋아|ok|good|pass", t):
        return "next"
    return "other"


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


async def _run_story_fixer(
    episode_key: str,
    user_feedback: str,
    run_dir: Path,
    fixer_cycle: int,
    set_process: ProcessFn = None,
    codex_model: str | None = None,
) -> tuple[bool, str]:
    """Codex로 에피소드 YAML 수정. 성공 여부와 요약 반환."""
    summary_path = run_dir / f"story_fixer_cycle{fixer_cycle}_summary.md"
    prompt = _build_story_fixer_prompt(episode_key, user_feedback)
    cmd = ["codex", "exec"]
    if codex_model:
        cmd += ["-m", codex_model]
    cmd += [
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd", str(REPO_ROOT),
        "-o", str(summary_path),
        prompt,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
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
) -> dict[str, Any]:
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
            f"리뷰 등급: `{review_tier}`"
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
        if set_status:
            set_status("중단됨 (Guardian 단계)")
        return {"success": False, "step": "guardian", "cycle": cycle}

    # ── Step 2: Simulator ──
    _t0 = time.monotonic()
    ok2 = await step_simulator(episode_key, run_dir, cycle, budget, notify, set_status, stop_event,
                               set_process=set_process,
                               cost_tracker=cost_tracker,
                               metrics=cost_tracker,
                               auto_cycle_index=0,
                               auto_max_cycles=AUTO_IMPROVE_MAX_CYCLES,
                               guardian_briefing_path=guardian_briefing_path,
                               reset_emotions=reset_emotions)
    if set_metrics:
        set_metrics(dict(cost_tracker))
    if not ok2:
        # 오류 발생 시 Codex 자동 진단 후 1회 재시도
        sim_log = run_dir / "simulator_error.log"
        sim_error_output = sim_log.read_text(encoding="utf-8", errors="replace") if sim_log.exists() else "(로그 없음)"
        fixed = await _run_codex_error_fixer(sim_error_output, "simulator", run_dir, notify, set_process)
        if fixed:
            if notify:
                await notify(f"{DAILY_TAG}[SIM] 🔄 Codex 수정 후 시뮬레이션 재시도...")
            ok2 = await step_simulator(episode_key, run_dir, cycle, budget, notify, set_status, stop_event,
                                       set_process=set_process,
                                       cost_tracker=cost_tracker,
                                       metrics=cost_tracker,
                                       auto_cycle_index=0,
                                       auto_max_cycles=AUTO_IMPROVE_MAX_CYCLES,
                                       guardian_briefing_path=guardian_briefing_path,
                                       reset_emotions=reset_emotions)
            if set_metrics:
                set_metrics(dict(cost_tracker))
        if not ok2:
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
    chapter_path = await step_chapter_gen(
        episode_key, run_dir, cycle, target_words, budget, protagonist,
        notify, upload, set_status, stop_event,
        set_process=set_process,
        cost_tracker=cost_tracker,
        metrics=cost_tracker,
        auto_cycle_index=0,
        auto_max_cycles=AUTO_IMPROVE_MAX_CYCLES,
        guardian_briefing_path=guardian_briefing_path,
    )
    if set_metrics:
        set_metrics(dict(cost_tracker))
    if chapter_path is None:
        # 오류 발생 시 Codex 자동 진단 후 1회 재시도
        chap_log = run_dir / "chapter_gen_error.log"
        chap_error_output = chap_log.read_text(encoding="utf-8", errors="replace") if chap_log.exists() else "(로그 없음)"
        fixed = await _run_codex_error_fixer(chap_error_output, "chapter_gen", run_dir, notify, set_process)
        if fixed:
            if notify:
                await notify(f"{DAILY_TAG}[CHAPTER] 🔄 Codex 수정 후 챕터 생성 재시도...")
            chapter_path = await step_chapter_gen(
                episode_key, run_dir, cycle, target_words, budget, protagonist,
                notify, upload, set_status, stop_event,
                set_process=set_process,
                cost_tracker=cost_tracker,
                metrics=cost_tracker,
                auto_cycle_index=0,
                auto_max_cycles=AUTO_IMPROVE_MAX_CYCLES,
                guardian_briefing_path=guardian_briefing_path,
            )
            if set_metrics:
                set_metrics(dict(cost_tracker))
        if chapter_path is None:
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
            f"(최대 {AUTO_IMPROVE_MAX_CYCLES}사이클, 목표 평균 {AUTO_IMPROVE_SCORE_THRESHOLD}/10)"
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
    )
    time_tracker["auto_improve"] = time.monotonic() - _t0
    if set_metrics:
        set_metrics(dict(cost_tracker))

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

    _increment_cycle(episode_key)

    total_elapsed = time.monotonic() - pipeline_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

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

    _prompt_tok = int(cost_tracker.get("prompt_tokens", 0))
    _comp_tok = int(cost_tracker.get("completion_tokens", 0))
    _token_line = f"🪙 토큰: 입력 {_prompt_tok:,} + 출력 {_comp_tok:,} = 총 {_prompt_tok + _comp_tok:,}"

    if no_discord or feedback_queue is None:
        if set_status:
            set_status("완료 (no-discord)")
        if notify:
            await notify(
                f"{DAILY_TAG}[DONE] ✅ 파이프라인 완료 (no-discord 모드)\n"
                f"- chapter: `{chapter_path.relative_to(REPO_ROOT)}`\n"
                f"⏱️ 총 소요 시간: {total_min}분 {total_sec:02d}초\n"
                f"{_token_line}\n"
                f"{_total_cost_line(cost_tracker)}"
            )
        return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": None, "feedback": None}

    # ── Step 5: 선택지 메뉴 ──
    if notify:
        await notify(
            f"{DAILY_TAG}[CHOICE] 📋 **개선 방향을 선택해주세요.**\n\n"
            "**1️⃣ 코드 수정** — Codex가 소설 생성 .py 파일을 자동 수정 후 챕터 재생성\n"
            "**2️⃣ 스토리 수정** — 에피소드 config YAML을 Codex로 직접 수정\n"
            "**3️⃣ 다음으로** — 이대로 승인하고 다음 화로\n\n"
            "번호 + 구체적인 의견을 같이 적으면 더 잘 반영됩니다.\n"
            "예: `1 대사가 너무 딱딱해` / `2 수민이 너무 수동적으로 나와`"
        )
    if set_status:
        set_status("선택 대기 중 (1=코드 / 2=스토리 / 3=다음)")

    if on_start_wait:
        on_start_wait()

    raw_feedback = await wait_for_feedback(
        feedback_queue, feedback_timeout_hours, notify, stop_event,
    )

    if on_end_wait:
        on_end_wait()

    if raw_feedback is None:
        if set_status:
            set_status("완료 (응답 없음)")
        if notify:
            await notify(
                f"{DAILY_TAG}[DONE] ℹ️ 응답이 없어 여기서 마무리했습니다.\n"
                "다시 실행하려면 `!novel-daily <번호>`\n"
                f"⏱️ 총 소요 시간: {total_min}분 {total_sec:02d}초\n"
                f"{_token_line}\n"
                f"{_total_cost_line(cost_tracker)}"
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

        # 사용자 피드백을 반영한 fixer prompt
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
            fixer_prompt,
            run_dir,
            cycle,
            set_process=set_process,
            notify=notify,
            stop_event=stop_event,
            codex_model=_codex_model_for_tier(review_tier),
        )
        if ok:
            if notify:
                await notify(f"{DAILY_TAG}[CHOICE] ✅ 코드 수정 완료:\n{summary}")
            committed, _ = await _git_commit_fixer_changes(cycle, episode_key, summary)
            if committed and notify:
                await notify(f"{DAILY_TAG}[CHOICE] 📦 git commit 완료")
            # 챕터 재생성
            if notify:
                await notify(f"{DAILY_TAG}[CHOICE] 📖 수정된 코드로 챕터 재생성 중...")
            new_chapter = await step_chapter_gen(
                episode_key, run_dir, cycle, target_words, budget, protagonist,
                notify=notify, upload=upload, set_status=set_status,
                stop_event=stop_event, set_process=set_process, cost_tracker=cost_tracker,
                upload_version_label=f"choice_code_cycle{cycle}",
                guardian_briefing_path=guardian_briefing_path,
            )
            if new_chapter:
                chapter_path = new_chapter
                if set_metrics:
                    set_metrics(dict(cost_tracker))
        else:
            if notify:
                await notify(f"{DAILY_TAG}[CHOICE] ❌ 코드 수정 실패: {summary}")

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

    # ── Step 5c: 다음으로 / 기타 피드백 ──
    else:
        pass  # 아래 Step 6에서 처리

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
    elif choice in ("code", "story"):
        parsed["approved_next_episode"] = False
    update_story_state(STORY_STATE_PATH, episode_key, episode_data, parsed)

    approved = parsed.get("approved_next_episode", False)
    if set_status:
        set_status(f"완료 — {'승인됨' if approved else '재시도 예정'}")
    if notify:
        choice_label = {"code": "코드 수정", "story": "스토리 수정", "next": "다음으로", "other": "피드백 저장"}.get(choice, "완료")
        if approved:
            await notify(
                f"{DAILY_TAG}[DONE] ✅ {choice_label} 완료. 다음 에피소드로 이동하려면: `!novel-daily <번호>`\n"
                f"⏱️ 총 소요 시간: {total_min}분 {total_sec:02d}초\n"
                f"{_token_line}\n"
                f"{_total_cost_line(cost_tracker)}"
            )
        else:
            issues = parsed.get("specific_issues", [])
            issue_str = "\n".join(f"  - {i}" for i in issues) if issues else "  (코멘트 참조)"
            await notify(
                f"{DAILY_TAG}[DONE] 📝 {choice_label} 완료. 같은 화 재시도: `!novel-daily {episode_key}`\n"
                f"개선 포인트:\n{issue_str}\n"
                f"⏱️ 총 소요 시간: {total_min}분 {total_sec:02d}초\n"
                f"{_token_line}\n"
                f"{_total_cost_line(cost_tracker)}"
            )

    return {"success": True, "cycle": cycle, "chapter_path": str(chapter_path), "approved": approved, "feedback": parsed, "choice": choice}


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
