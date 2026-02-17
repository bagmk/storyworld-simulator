#!/usr/bin/env python3
"""\
Safely scale `max_turns` values in all episode YAML files.

This is analogous to `update_thresholds.py`, but it targets lines like:
    max_turns: 86
and rewrites them to a scaled integer value (default: 1/2).

Design goals:
- Regex-based replacement so comments/formatting are preserved.
- Writes ONLY if content changed.
- Supports --dry-run.

Usage:
  python scripts/update_max_turns_quarter.py --dry-run
  python scripts/update_max_turns_quarter.py --episodes-dir config/episodes

Optional knobs:
  --factor 0.5            Scale factor (default: 0.5)
  --rounding floor|ceil|round
  --min-turns 1            Lower bound after scaling
"""

import argparse
import math
import re
from pathlib import Path


def _apply_rounding(x: float, mode: str) -> int:
    if mode == "floor":
        return int(math.floor(x))
    if mode == "ceil":
        return int(math.ceil(x))
    if mode == "round":
        # Python uses bankers rounding; acceptable for positive ints.
        return int(round(x))
    raise ValueError(f"Unknown rounding mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale max_turns in episode YAMLs")
    parser.add_argument(
        "--episodes-dir",
        default="config/episodes",
        help="Directory containing episode YAML files (default: config/episodes)",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="Scale factor applied to max_turns (default: 0.5)",
    )
    parser.add_argument(
        "--rounding",
        choices=["floor", "ceil", "round"],
        default="round",
        help="How to convert scaled value to int (default: round)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=1,
        help="Minimum allowed max_turns after scaling (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )

    args = parser.parse_args()

    if args.factor <= 0:
        raise SystemExit("--factor must be > 0")
    if args.min_turns < 1:
        raise SystemExit("--min-turns must be >= 1")

    episodes_dir = Path(args.episodes_dir)
    files = sorted(episodes_dir.glob("ep*.yaml"))

    if not files:
        print(f"No episode files found in {episodes_dir}")
        return

    total_changes = 0

    for filepath in files:
        content = filepath.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  SKIP (empty): {filepath.name}")
            continue

        changes_in_file = 0

        def replacer(match: re.Match) -> str:
            nonlocal changes_in_file
            prefix = match.group(1)
            old_val = int(match.group(2))
            scaled = old_val * args.factor
            new_val = max(args.min_turns, _apply_rounding(scaled, args.rounding))
            if new_val != old_val:
                changes_in_file += 1
                return f"{prefix}{new_val}"
            return match.group(0)

        new_content = re.sub(r"(max_turns:\s*)(\d+)", replacer, content)

        if new_content != content:
            total_changes += changes_in_file
            if args.dry_run:
                print(f"  WOULD CHANGE: {filepath.name} ({changes_in_file} max_turns)")
                # Show before/after (first occurrence is typical)
                old_vals = re.findall(r"max_turns:\s*(\d+)", content)
                new_vals = re.findall(r"max_turns:\s*(\d+)", new_content)
                for ov, nv in zip(old_vals, new_vals):
                    if ov != nv:
                        print(f"    {ov} → {nv}")
            else:
                filepath.write_text(new_content, encoding="utf-8")
                print(f"  UPDATED: {filepath.name} ({changes_in_file} max_turns)")
        else:
            print(f"  OK (no change): {filepath.name}")

    action = "Would change" if args.dry_run else "Changed"
    print(f"\n{action} {total_changes} max_turns values across {len(files)} files")


if __name__ == "__main__":
    main()
