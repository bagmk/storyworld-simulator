#!/usr/bin/env python3
"""
Repeat generation/evaluation until POV+timeline benchmark reaches 100%.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from quality_analyzer import analyze_file


def run(cmd: list[str], env: dict) -> None:
    proc = subprocess.run(cmd, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-config", required=True)
    parser.add_argument("--protagonist", default="kim_sumin")
    parser.add_argument("--words", type=int, default=1400)
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--target", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    output_dir = root / "output"
    runs_md = output_dir / "rl_runs"
    runs_json = root / "reports" / "rl_runs"
    runs_md.mkdir(parents=True, exist_ok=True)
    runs_json.mkdir(parents=True, exist_ok=True)

    env = dict(**__import__("os").environ)
    best = None
    best_round = 0

    for i in range(1, args.max_rounds + 1):
        print(f"\n=== ROUND {i}/{args.max_rounds} ===", flush=True)
        before = {
            p.resolve(): p.stat().st_mtime for p in output_dir.glob("*_chapter.md")
        }
        run(
            [
                sys.executable,
                str(root / "generate_chapter.py"),
                "--episode",
                args.episode_id,
                "--episode-config",
                args.episode_config,
                "--protagonist",
                args.protagonist,
                "--style",
                "third_person_close",
                "--words",
                str(args.words),
                "--scenes",
                str(args.scenes),
                "--output",
                str(output_dir),
            ],
            env=env,
        )

        # Prefer chapter generated/updated in this round.
        changed = [
            p for p in output_dir.glob("*_chapter.md")
            if p.resolve() not in before or p.stat().st_mtime > before[p.resolve()] + 0.5
        ]
        chapters = sorted(changed, key=lambda p: p.stat().st_mtime, reverse=True)
        if not chapters:
            chapters = sorted(output_dir.glob("*_chapter.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not chapters:
            raise RuntimeError("No chapter file found")
        chapter_path = chapters[0]

        result = analyze_file(chapter_path)
        pov = float(result["pov_consistency"]["score"])
        tl = float(result["timeline_coherence"]["score"])
        overall = float(result["overall_score"])

        round_md = runs_md / f"{args.episode_id}_round_{i}.md"
        round_json = runs_json / f"{args.episode_id}_round_{i}.json"
        shutil.copy2(chapter_path, round_md)
        round_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"overall={overall:.3f} pov={pov:.2f} timeline={tl:.2f}", flush=True)

        if best is None or (pov + tl, overall) > (best["pov"] + best["tl"], best["overall"]):
            best = {"pov": pov, "tl": tl, "overall": overall}
            best_round = i
            shutil.copy2(round_md, output_dir / f"{args.episode_id}_best_100pct_loop.md")
            shutil.copy2(round_json, root / f"reports/{args.episode_id}_best_100pct_loop.json")

        if pov >= args.target and tl >= args.target:
            shutil.copy2(round_md, output_dir / f"{args.episode_id}_100pct.md")
            shutil.copy2(round_json, root / f"reports/{args.episode_id}_100pct.json")
            print(f"REACHED_100 round={i}", flush=True)
            return

    print(
        f"NOT_REACHED_100 max_rounds={args.max_rounds} "
        f"best_round={best_round} best_pov={best['pov']:.2f} best_timeline={best['tl']:.2f} best_overall={best['overall']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
