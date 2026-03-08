"""
Utilities for loading and shaping reader review markdown feedback.
"""

from __future__ import annotations

from pathlib import Path
import re


def load_reader_review(path: str) -> dict:
    """
    Load a markdown reader review and extract actionable fields.
    Returns an empty dict if parsing yields no usable signal.
    """
    review_path = Path(path).expanduser()
    text = review_path.read_text(encoding="utf-8")
    parsed = parse_reader_review_markdown(text)
    if not parsed:
        return {}
    parsed["source_path"] = str(review_path)
    return parsed


def resolve_reader_review_path(
    explicit_path: str = "",
    episode_id: str = "",
    output_dir: str = "output",
) -> Path | None:
    """
    Resolve which reader review markdown should be used for a run.
    Priority:
    1) Explicit --reader-review-md path
    2) Latest cycle review for episode under output_dir
    3) Latest review-like markdown under output_dir
    """
    if explicit_path and explicit_path.strip():
        p = Path(explicit_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Reader review markdown not found: {p}")
        return p

    base = Path(output_dir).expanduser()
    if not base.exists():
        return None

    candidates: list[Path] = []
    if episode_id:
        candidates.extend(base.rglob(f"{episode_id}_cycle*_review.md"))
        candidates.extend(base.rglob(f"{episode_id}_review.md"))
        candidates.extend(base.rglob(f"{episode_id}*_review.md"))
    if not candidates:
        candidates.extend(base.rglob("*_cycle*_review.md"))
    if not candidates:
        candidates.extend(base.rglob("*_review.md"))
    if not candidates:
        return None

    candidates = list(dict.fromkeys(candidates))

    def _extract_run_hint(path: Path, base_dir: Path) -> tuple[int, int]:
        """
        Extract run hierarchy hints from paths like output/YYYYMMDD/NNN/file.md.
        Returns (-1, -1) when unavailable.
        """
        try:
            rel = path.resolve().relative_to(base_dir.resolve())
            parts = list(rel.parts)
        except Exception:
            parts = list(path.parts)
        for i in range(len(parts) - 2):
            date_part = parts[i]
            run_part = parts[i + 1]
            if re.fullmatch(r"\d{8}", date_part) and re.fullmatch(r"\d{3}", run_part):
                return int(date_part), int(run_part)
        return -1, -1

    def _sort_key(path: Path) -> tuple[int, int, int, float]:
        m = re.search(r"_cycle(\d+)_review\.md$", path.name)
        cycle = int(m.group(1)) if m else -1
        run_date, run_index = _extract_run_hint(path, base)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        # Prefer newer run folders first (output/YYYYMMDD/NNN), then cycle inside that run.
        # This avoids stale high-cycle files from an older run overriding current run feedback.
        # Fall back to mtime when run/cycle hints are unavailable.
        return run_date, run_index, cycle, mtime

    return max(candidates, key=_sort_key)


def parse_reader_review_markdown(text: str) -> dict:
    """
    Extract commonly used sections from a markdown review report.
    """
    if not text or not text.strip():
        return {}

    section_map = {
        "좋았던 점": "what_felt_good",
        "지루하거나 읽기 어려웠던 점": "what_felt_boring_or_hard",
        "문체 개선 팁": "style_tips",
    }
    out = {
        "what_felt_good": [],
        "what_felt_boring_or_hard": [],
        "style_tips": [],
        "one_line_verdict": "",
        "reader_comment": "",
    }

    current_key = ""
    in_comment = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            in_comment = line.startswith("## 독자 코멘트")
            current_key = ""
            if in_comment:
                continue
            title = line[3:].strip()
            for marker, key in section_map.items():
                if marker in title:
                    current_key = key
                    break
            continue

        if line.startswith("- 한 줄 평:"):
            out["one_line_verdict"] = line.split(":", 1)[1].strip()
            continue

        if in_comment:
            if out["reader_comment"]:
                out["reader_comment"] += " " + line.lstrip("- ").strip()
            else:
                out["reader_comment"] = line.lstrip("- ").strip()
            continue

        if current_key and line.startswith("- "):
            item = line[2:].strip()
            if item:
                out[current_key].append(item)

    cleaned = {
        k: v for k, v in out.items()
        if (isinstance(v, list) and v) or (isinstance(v, str) and v.strip())
    }
    return cleaned


def build_feedback_prompt_block(review: dict, max_items: int = 3) -> str:
    """
    Convert parsed review fields into a compact prompt-ready guidance block.
    """
    if not isinstance(review, dict) or not review:
        return ""

    lines: list[str] = []
    weak = _select_priority_items(review.get("what_felt_boring_or_hard", []) or [], max_items)
    tips = _select_priority_items(review.get("style_tips", []) or [], max_items)
    if weak:
        lines.append("Reader pain points to reduce:")
        for item in weak:
            lines.append(f"- {item}")
    if tips:
        lines.append("Reader-requested style adjustments:")
        for item in tips:
            lines.append(f"- {item}")
    verdict = str(review.get("one_line_verdict", "")).strip()
    if verdict:
        lines.append(f"One-line verdict: {verdict}")
    comment = str(review.get("reader_comment", "")).strip()
    if comment:
        lines.append(f"Reader comment: {comment[:240]}")
    return "\n".join(lines).strip()


def _select_priority_items(items: list[str], max_items: int) -> list[str]:
    """
    Select diverse high-value feedback points instead of taking only the first N.
    This avoids dropping late-but-critical notes (e.g., speaker clarity).
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = str(raw or "").strip()
        if not item:
            continue
        fp = _fingerprint(item)
        if fp in seen:
            continue
        seen.add(fp)
        cleaned.append(item)

    if not cleaned or max_items <= 0:
        return []

    scored = []
    for idx, item in enumerate(cleaned):
        scored.append((_priority_score(item), idx, item))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [item for _, _, item in scored[:max_items]]


def _priority_score(text: str) -> int:
    t = str(text or "")
    score = 1
    patterns = [
        (r"기술|용어|약자|전문|jargon|acronym", 4),
        (r"반복|중복|늘어지|같은|유의어", 4),
        (r"긴 문단|문단|호흡|리듬|속도", 4),
        (r"누가 말|화자|대화 전환|헷갈|speaker|dialogue", 5),
        (r"증거|목록|요약|흐름", 3),
    ]
    for pattern, weight in patterns:
        if re.search(pattern, t, flags=re.IGNORECASE):
            score += weight
    return score


def _fingerprint(text: str) -> str:
    norm = re.sub(r"[^0-9a-z가-힣\s]", " ", str(text or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm
