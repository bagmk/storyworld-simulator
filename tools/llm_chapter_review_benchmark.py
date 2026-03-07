#!/usr/bin/env python3
"""
LLM-based chapter review benchmark.

Adds a human-like evaluative metric on top of rule-based quality metrics by asking an
LLM to score each chapter and recommend concrete fixes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.novel_writer.env_loader import load_project_env
from src.novel_writer.llm_client import LLMClient


SYSTEM_PROMPT = """You are a strict fiction quality reviewer for serialized thriller chapters.
Return ONLY valid JSON matching the requested schema.
Focus on:
- narrative clarity and scene continuity
- character consistency and motivation readability
- tension / pacing balance
- prose vividness (concrete detail vs abstraction)
- dialogue usefulness
- continuity hooks for a serialized story
Scores must be 0.0 to 1.0.
Recommendations must be actionable rewrite suggestions.
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        return {}
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def _chapter_excerpt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.35) :]
    return f"{head}\n\n...[TRUNCATED]...\n\n{tail}"


def _episode_meta(config_path: Path) -> tuple[str, int | None]:
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    ep = data.get("episode", data) if isinstance(data, dict) else {}
    episode_id = str(ep.get("id", config_path.stem)).strip() if isinstance(ep, dict) else config_path.stem
    m = re.search(r"ep(\d+)", config_path.stem)
    episode_num = int(m.group(1)) if m else None
    return episode_id, episode_num


def _find_generated_chapter(out_dir: Path, episode_id: str, ep_num: int | None) -> Path | None:
    candidates: list[Path] = []
    patterns = [f"{episode_id}_chapter.md"]
    if ep_num is not None:
        patterns.extend([f"ep{ep_num:02d}_*_chapter.md", f"ep{ep_num:02d}_*.md"])
    for pat in patterns:
        candidates.extend(sorted(out_dir.glob(pat)))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.name), p.stat().st_mtime), reverse=True)
    return candidates[0]


def _find_good_example(good_dir: Path, ep_num: int | None) -> Path | None:
    if ep_num is None:
        return None
    p = good_dir / f"ep{ep_num}.md"
    return p if p.exists() else None


def review_one(
    llm: LLMClient,
    chapter_text: str,
    episode_id: str,
    episode_num: int | None,
    source_group: str,
    max_chars: int,
) -> dict[str, Any]:
    excerpt = _chapter_excerpt(chapter_text, max_chars=max_chars)
    prompt = {
        "task": "Evaluate a serialized novel chapter and recommend revisions",
        "episode_id": episode_id,
        "episode_number": episode_num,
        "source_group": source_group,
        "output_schema": {
            "overall_score": "float 0..1",
            "scores": {
                "clarity": "float 0..1",
                "character_consistency": "float 0..1",
                "pacing_tension": "float 0..1",
                "prose_vividness": "float 0..1",
                "dialogue_effectiveness": "float 0..1",
                "serial_continuity_hook": "float 0..1",
            },
            "strengths": ["short strings"],
            "issues": ["short strings"],
            "priority_fixes": [
                {
                    "category": "one of clarity/character/pacing/prose/dialogue/continuity",
                    "problem": "short string",
                    "recommendation": "actionable rewrite direction",
                    "impact": "high/medium/low",
                }
            ],
            "one_paragraph_summary": "short paragraph",
        },
        "chapter_text": excerpt,
    }
    raw = llm.chat(
        messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        system=SYSTEM_PROMPT,
        use_premium=False,
        purpose="llm_chapter_review",
        temperature=0.2,
        max_tokens=1200,
    )
    parsed = _extract_json(raw)
    return parsed | {"_raw_response": raw}


def _score(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, v))


def _normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    scores = review.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    norm_scores = {}
    for k in [
        "clarity",
        "character_consistency",
        "pacing_tension",
        "prose_vividness",
        "dialogue_effectiveness",
        "serial_continuity_hook",
    ]:
        norm_scores[k] = _score(scores.get(k))

    overall = _score(review.get("overall_score"))
    if overall is None:
        vals = [v for v in norm_scores.values() if isinstance(v, float)]
        overall = round(sum(vals) / len(vals), 3) if vals else 0.0

    def _as_list(name: str) -> list[str]:
        v = review.get(name, [])
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]

    priority_fixes = review.get("priority_fixes", [])
    if not isinstance(priority_fixes, list):
        priority_fixes = []
    cleaned_fixes: list[dict[str, str]] = []
    for row in priority_fixes[:8]:
        if not isinstance(row, dict):
            continue
        cleaned_fixes.append(
            {
                "category": str(row.get("category", "")).strip(),
                "problem": str(row.get("problem", "")).strip(),
                "recommendation": str(row.get("recommendation", "")).strip(),
                "impact": str(row.get("impact", "")).strip(),
            }
        )

    return {
        "overall_score": round(overall, 3),
        "scores": {k: (round(v, 3) if isinstance(v, float) else None) for k, v in norm_scores.items()},
        "strengths": _as_list("strengths")[:8],
        "issues": _as_list("issues")[:8],
        "priority_fixes": cleaned_fixes,
        "one_paragraph_summary": str(review.get("one_paragraph_summary", "")).strip(),
        "raw_response": review.get("_raw_response", ""),
    }


def build_targets(episodes: list[Path], mode: str, output_dir: Path, good_examples_dir: Path) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for cfg in sorted(episodes):
        eid, ep_num = _episode_meta(cfg)
        if mode in {"generated", "both"}:
            p = _find_generated_chapter(output_dir, eid, ep_num)
            if p:
                targets.append(
                    {
                        "episode_id": eid,
                        "episode_number": ep_num,
                        "source_group": "generated",
                        "chapter_path": p,
                        "episode_config": str(cfg),
                    }
                )
        if mode in {"good_example", "both"}:
            p = _find_good_example(good_examples_dir, ep_num)
            if p:
                targets.append(
                    {
                        "episode_id": eid,
                        "episode_number": ep_num,
                        "source_group": "good_example",
                        "chapter_path": p,
                        "episode_config": str(cfg),
                    }
                )
    return targets


def main() -> None:
    load_project_env()

    p = argparse.ArgumentParser(description="LLM chapter review benchmark")
    p.add_argument("--episodes", nargs="+", required=True, help="Episode YAML config paths")
    p.add_argument("--mode", choices=["generated", "good_example", "both"], default="both")
    p.add_argument("--output-dir", default="output", help="Generated chapters directory")
    p.add_argument("--good-examples-dir", default="examples/Good_example", help="Reference chapter directory")
    p.add_argument("--report-out", required=True, help="JSON output path")
    p.add_argument("--model", default="gpt-4o-mini", help="LLM model for review")
    p.add_argument("--premium-model", default="gpt-5-mini", help="Premium model (unused unless changed)")
    p.add_argument("--budget", type=float, default=6.0, help="Total review budget")
    p.add_argument("--max-chars", type=int, default=14000, help="Max chapter chars sent per review")
    args = p.parse_args()

    targets = build_targets(
        episodes=[Path(x) for x in args.episodes],
        mode=args.mode,
        output_dir=Path(args.output_dir),
        good_examples_dir=Path(args.good_examples_dir),
    )
    if not targets:
        raise SystemExit("No chapter files found to review.")

    llm = LLMClient(model=args.model, premium_model=args.premium_model, budget_usd=args.budget)

    rows: list[dict[str, Any]] = []
    for idx, t in enumerate(targets, start=1):
        chapter_path = Path(t["chapter_path"])
        print(f"[{idx}/{len(targets)}] review {t['source_group']} {t['episode_id']} <- {chapter_path}")
        text = _read_text(chapter_path)
        raw_review = review_one(
            llm=llm,
            chapter_text=text,
            episode_id=t["episode_id"],
            episode_num=t["episode_number"],
            source_group=t["source_group"],
            max_chars=args.max_chars,
        )
        norm = _normalize_review(raw_review)
        rows.append(
            {
                **t,
                "chapter_path": str(chapter_path),
                **norm,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["source_group"], []).append(r)

    summary_by_group: dict[str, dict[str, Any]] = {}
    for group, items in grouped.items():
        scores = [float(x["overall_score"]) for x in items if x.get("overall_score") is not None]
        summary_by_group[group] = {
            "count": len(items),
            "avg_overall_score": round(sum(scores) / len(scores), 3) if scores else None,
        }

    report = {
        "benchmark": "llm_chapter_review",
        "generated_at_utc": _utc_now(),
        "config": {
            "mode": args.mode,
            "output_dir": args.output_dir,
            "good_examples_dir": args.good_examples_dir,
            "model": args.model,
            "budget": args.budget,
            "max_chars": args.max_chars,
        },
        "summary_by_group": summary_by_group,
        "episodes": rows,
        "llm_budget_summary": llm.budget_summary(),
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {out}")
    for group, s in summary_by_group.items():
        print(f"  {group}: n={s['count']} avg={s['avg_overall_score']}")


if __name__ == "__main__":
    main()
