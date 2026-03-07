#!/usr/bin/env python3
"""
Build an ep01-ep10 benchmark comparison notebook and summary tables.

Inputs:
- rule-based quality reports for generated and Good_example
- coherence reports for generated and Good_example
- LLM review report (both groups)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _ep_num_from_key(s: str) -> int:
    import re

    m = re.search(r"ep(\d+)", s or "")
    return int(m.group(1)) if m else 9999


def _quality_index(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in report.get("episodes", []):
        key = str(row.get("episode") or row.get("episode_key") or "")
        n = _ep_num_from_key(key)
        if n == 9999:
            continue
        out[n] = row
    return out


def _llm_index(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("episodes", []):
        n = row.get("episode_number")
        g = str(row.get("source_group", ""))
        if isinstance(n, int) and g:
            out[(n, g)] = row
    return out


def _coh_summary(report: dict[str, Any], label: str) -> list[tuple[str, Any]]:
    ft = report.get("full_text_coherence_benchmark", {}) or {}
    gcv = report.get("global_character_voice", {}) or {}
    return [
        ("label", label),
        ("series_coherence_score", report.get("series_coherence_score")),
        ("series_coherence_score_with_full_text", report.get("series_coherence_score_with_full_text")),
        ("full_text_coherence_score", ft.get("score")),
        ("global_character_voice_score", gcv.get("score")),
        ("protagonist_presence_rate", report.get("protagonist_presence_rate")),
    ]


def build_summary_rows(
    quality_generated: dict[str, Any],
    quality_good: dict[str, Any],
    llm_review: dict[str, Any],
) -> list[dict[str, Any]]:
    qg = _quality_index(quality_generated)
    qx = _quality_index(quality_good)
    li = _llm_index(llm_review)
    rows: list[dict[str, Any]] = []
    for n in range(1, 11):
        gen_q = qg.get(n, {})
        good_q = qx.get(n, {})
        gen_l = li.get((n, "generated"), {})
        good_l = li.get((n, "good_example"), {})
        rows.append(
            {
                "episode": f"ep{n:02d}",
                "generated_quality": gen_q.get("overall_score"),
                "good_quality": good_q.get("overall_score"),
                "quality_delta_vs_good": _delta(gen_q.get("overall_score"), good_q.get("overall_score")),
                "generated_llm_score": gen_l.get("overall_score"),
                "good_llm_score": good_l.get("overall_score"),
                "llm_delta_vs_good": _delta(gen_l.get("overall_score"), good_l.get("overall_score")),
                "generated_words": _nested(gen_q, "basic_stats", "word_count"),
                "good_words": _nested(good_q, "basic_stats", "word_count"),
                "generated_top_fix": _first_fix(gen_l),
            }
        )
    return rows


def _nested(row: dict[str, Any], key1: str, key2: str) -> Any:
    obj = row.get(key1, {})
    if isinstance(obj, dict):
        return obj.get(key2)
    return None


def _first_fix(llm_row: dict[str, Any]) -> str:
    fixes = llm_row.get("priority_fixes", [])
    if not isinstance(fixes, list) or not fixes:
        return ""
    first = fixes[0] if isinstance(fixes[0], dict) else {}
    return str(first.get("recommendation", "")).strip()


def _delta(a: Any, b: Any) -> float | None:
    try:
        return round(float(a) - float(b), 3)
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    coh_generated: dict[str, Any],
    coh_good: dict[str, Any],
    llm_review: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# ep01-ep10 Benchmark Comparison")
    lines.append("")
    lines.append("## Series-Level Summary")
    lines.append("")
    lines.append("| Metric | Generated | Good Example | Delta |")
    lines.append("|---|---:|---:|---:|")
    gen_series = {k: v for k, v in _coh_summary(coh_generated, "generated")}
    good_series = {k: v for k, v in _coh_summary(coh_good, "good_example")}
    for k in [
        "series_coherence_score",
        "series_coherence_score_with_full_text",
        "full_text_coherence_score",
        "global_character_voice_score",
        "protagonist_presence_rate",
    ]:
        gv = gen_series.get(k)
        xv = good_series.get(k)
        dv = _delta(gv, xv)
        lines.append(f"| {k} | {_fmt(gv)} | {_fmt(xv)} | {_fmt(dv)} |")

    llm_sum = llm_review.get("summary_by_group", {}) or {}
    lines.append("")
    lines.append("| Metric | Generated | Good Example | Delta |")
    lines.append("|---|---:|---:|---:|")
    gl = (llm_sum.get("generated") or {}).get("avg_overall_score")
    xl = (llm_sum.get("good_example") or {}).get("avg_overall_score")
    lines.append(f"| llm_review_avg_overall_score | {_fmt(gl)} | {_fmt(xl)} | {_fmt(_delta(gl, xl))} |")

    lines.append("")
    lines.append("## Episode Table (1-10)")
    lines.append("")
    lines.append("| Episode | Gen Quality | Good Quality | ΔQ | Gen LLM | Good LLM | ΔLLM | Top Fix (Generated) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['episode']} | {_fmt(r['generated_quality'])} | {_fmt(r['good_quality'])} | "
            f"{_fmt(r['quality_delta_vs_good'])} | {_fmt(r['generated_llm_score'])} | "
            f"{_fmt(r['good_llm_score'])} | {_fmt(r['llm_delta_vs_good'])} | "
            f"{_md_escape(r['generated_top_fix'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_escape(s: Any) -> str:
    return str(s or "").replace("|", "/")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return ""
    return str(v)


def build_notebook(
    out_path: Path,
    quality_generated_path: str,
    quality_good_path: str,
    coherence_generated_path: str,
    coherence_good_path: str,
    llm_review_path: str,
    summary_csv_path: str,
) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ep01-ep10 Benchmark Comparison Notebook\n",
                "\n",
                "생성 챕터(`output/`)와 `examples/Good_example`를 규칙 기반 품질/시리즈 일관성/LLM 리뷰 기준으로 비교합니다.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "\n",
                f"QUALITY_GENERATED = Path({quality_generated_path!r})\n",
                f"QUALITY_GOOD = Path({quality_good_path!r})\n",
                f"COH_GENERATED = Path({coherence_generated_path!r})\n",
                f"COH_GOOD = Path({coherence_good_path!r})\n",
                f"LLM_REVIEW = Path({llm_review_path!r})\n",
                f"SUMMARY_CSV = Path({summary_csv_path!r})\n",
                "\n",
                "def loadj(p):\n",
                "    return json.loads(p.read_text(encoding='utf-8'))\n",
                "\n",
                "qg = loadj(QUALITY_GENERATED)\n",
                "qx = loadj(QUALITY_GOOD)\n",
                "cg = loadj(COH_GENERATED)\n",
                "cx = loadj(COH_GOOD)\n",
                "lr = loadj(LLM_REVIEW)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "summary_df = pd.read_csv(SUMMARY_CSV)\n",
                "summary_df\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "series_rows = [\n",
                "    {\n",
                "      'metric': 'series_coherence_score',\n",
                "      'generated': cg.get('series_coherence_score'),\n",
                "      'good_example': cx.get('series_coherence_score'),\n",
                "    },\n",
                "    {\n",
                "      'metric': 'series_coherence_score_with_full_text',\n",
                "      'generated': cg.get('series_coherence_score_with_full_text'),\n",
                "      'good_example': cx.get('series_coherence_score_with_full_text'),\n",
                "    },\n",
                "    {\n",
                "      'metric': 'full_text_coherence_score',\n",
                "      'generated': (cg.get('full_text_coherence_benchmark') or {}).get('score'),\n",
                "      'good_example': (cx.get('full_text_coherence_benchmark') or {}).get('score'),\n",
                "    },\n",
                "    {\n",
                "      'metric': 'llm_review_avg_overall_score',\n",
                "      'generated': ((lr.get('summary_by_group') or {}).get('generated') or {}).get('avg_overall_score'),\n",
                "      'good_example': ((lr.get('summary_by_group') or {}).get('good_example') or {}).get('avg_overall_score'),\n",
                "    },\n",
                "]\n",
                "series_df = pd.DataFrame(series_rows)\n",
                "series_df['delta'] = series_df['generated'] - series_df['good_example']\n",
                "series_df\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "llm_rows = pd.DataFrame(lr.get('episodes', []))\n",
                "llm_rows[['episode_number','episode_id','source_group','overall_score','issues','priority_fixes']].sort_values(['episode_number','source_group'])\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generated chapters only: high-priority rewrite suggestions\n",
                "gen = llm_rows[llm_rows['source_group']=='generated'].copy()\n",
                "def top_fix(fixes):\n",
                "    if isinstance(fixes, list) and fixes:\n",
                "        f = fixes[0] if isinstance(fixes[0], dict) else {}\n",
                "        return f.get('recommendation', '')\n",
                "    return ''\n",
                "gen['top_fix'] = gen['priority_fixes'].apply(top_fix)\n",
                "gen[['episode_number','episode_id','overall_score','top_fix']].sort_values('episode_number')\n",
            ],
        },
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build benchmark comparison notebook and summary table")
    p.add_argument("--quality-generated", required=True)
    p.add_argument("--quality-good", required=True)
    p.add_argument("--coherence-generated", required=True)
    p.add_argument("--coherence-good", required=True)
    p.add_argument("--llm-review", required=True)
    p.add_argument("--notebook-out", default="notebooks/ep01_ep10_benchmark_comparison.ipynb")
    p.add_argument("--summary-csv-out", default="reports/ep01_ep10_benchmark_summary.csv")
    p.add_argument("--summary-md-out", default="reports/ep01_ep10_benchmark_summary.md")
    args = p.parse_args()

    qg = _load_json(args.quality_generated)
    qx = _load_json(args.quality_good)
    cg = _load_json(args.coherence_generated)
    cx = _load_json(args.coherence_good)
    lr = _load_json(args.llm_review)

    rows = build_summary_rows(qg, qx, lr)

    csv_path = Path(args.summary_csv_out)
    md_path = Path(args.summary_md_out)
    nb_path = Path(args.notebook_out)

    write_csv(csv_path, rows)
    write_markdown(md_path, rows, cg, cx, lr)
    build_notebook(
        out_path=nb_path,
        quality_generated_path=args.quality_generated,
        quality_good_path=args.quality_good,
        coherence_generated_path=args.coherence_generated,
        coherence_good_path=args.coherence_good,
        llm_review_path=args.llm_review,
        summary_csv_path=args.summary_csv_out,
    )

    print(f"Summary CSV: {csv_path}")
    print(f"Summary MD:  {md_path}")
    print(f"Notebook:    {nb_path}")


if __name__ == "__main__":
    main()
