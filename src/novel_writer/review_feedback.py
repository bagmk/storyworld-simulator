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


def ensure_repetition_watch_terms(review: dict) -> dict:
    """
    Ensure repetition_watch_terms exists when feedback clearly reports repetition.
    """
    if not isinstance(review, dict) or not review:
        return review if isinstance(review, dict) else {}

    existing = review.get("repetition_watch_terms", []) or []
    if existing:
        return review

    signals: list[str] = []
    for key in ("what_felt_boring_or_hard", "style_tips"):
        vals = review.get(key, []) or []
        if isinstance(vals, list):
            signals.extend(str(v or "") for v in vals)
        else:
            signals.append(str(vals or ""))
    signals.append(str(review.get("reader_comment", "") or ""))

    corpus = " ".join(s for s in signals if s).lower()
    if not any(k in corpus for k in ("반복", "중복", "같은 표현", "유의어", "늘어지")):
        return review

    inferred = _extract_repetition_watch_terms(signals)
    if inferred:
        review["repetition_watch_terms"] = inferred
    return review


def ensure_jargon_watch_terms(review: dict) -> dict:
    """
    Ensure jargon_watch_terms exists when feedback reports technical-term overload.
    """
    if not isinstance(review, dict) or not review:
        return review if isinstance(review, dict) else {}

    existing = review.get("jargon_watch_terms", []) or []
    if existing:
        return review

    signals: list[str] = []
    for key in ("what_felt_boring_or_hard", "style_tips"):
        vals = review.get(key, []) or []
        if isinstance(vals, list):
            signals.extend(str(v or "") for v in vals)
        else:
            signals.append(str(vals or ""))
    signals.append(str(review.get("reader_comment", "") or ""))

    corpus = " ".join(s for s in signals if s).lower()
    if not any(k in corpus for k in ("기술", "용어", "약어", "약자", "전문", "jargon", "acronym")):
        return review

    inferred = _extract_jargon_watch_terms(signals)
    if inferred:
        review["jargon_watch_terms"] = inferred
    return review


def resolve_reader_review_path(
    explicit_path: str = "",
    episode_id: str = "",
    output_dir: str = "output",
    prefer_run_id: str = "",
) -> Path | None:
    """
    Resolve which reader review markdown should be used for a run.
    Priority:
    1) Explicit --reader-review-md path
    2) Latest cycle review/manager report for episode under output_dir
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

    episode_key = ""
    m = re.match(r"^(ep\d{2})[_-]?.*$", str(episode_id or "").strip(), flags=re.IGNORECASE)
    if m:
        episode_key = m.group(1).lower()

    def _collect_candidates(search_base: Path) -> list[Path]:
        matches: list[Path] = []
        if episode_id:
            matches.extend(search_base.rglob(f"{episode_id}_cycle*_review.md"))
            matches.extend(search_base.rglob(f"{episode_id}_cycle*_manager.md"))
            matches.extend(search_base.rglob(f"{episode_id}_review.md"))
            matches.extend(search_base.rglob(f"{episode_id}_manager.md"))
            matches.extend(search_base.rglob(f"{episode_id}*_review.md"))
            matches.extend(search_base.rglob(f"{episode_id}*_manager.md"))
        if episode_key:
            matches.extend(search_base.rglob(f"{episode_key}_*_cycle*_review.md"))
            matches.extend(search_base.rglob(f"{episode_key}_*_cycle*_manager.md"))
            matches.extend(search_base.rglob(f"{episode_key}_*_review.md"))
            matches.extend(search_base.rglob(f"{episode_key}_*_manager.md"))
        if not matches:
            matches.extend(search_base.rglob("*_cycle*_review.md"))
            matches.extend(search_base.rglob("*_cycle*_manager.md"))
        if not matches:
            matches.extend(search_base.rglob("*_review.md"))
            matches.extend(search_base.rglob("*_manager.md"))
        return list(dict.fromkeys(matches))

    run_hint = str(prefer_run_id or "").strip()
    if run_hint:
        run_ids = [run_hint]
        if run_hint.isdigit():
            run_ids.append(run_hint.zfill(3))
        run_ids = list(dict.fromkeys(run_ids))
        run_dirs: list[Path] = []
        for rid in run_ids:
            local_dir = base / rid
            if local_dir.is_dir():
                run_dirs.append(local_dir)
            for p in base.glob(f"*/{rid}"):
                if p.is_dir():
                    run_dirs.append(p)
        if run_dirs:
            run_dirs = list(dict.fromkeys(run_dirs))

            def _run_dir_sort_key(path: Path) -> tuple[int, float]:
                parent = path.parent.name
                run_date = int(parent) if re.fullmatch(r"\d{8}", parent) else -1
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                return run_date, mtime

            preferred_run_dir = max(run_dirs, key=_run_dir_sort_key)
            scoped = _collect_candidates(preferred_run_dir)
            if scoped:
                candidates = scoped
            else:
                candidates = _collect_candidates(base)
        else:
            candidates = _collect_candidates(base)
    else:
        candidates = _collect_candidates(base)

    if not candidates:
        return None

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
        m = re.search(r"_cycle(\d+)_(?:review|manager)\.md$", path.name)
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

    section_markers = [
        ("what_felt_good", ("좋았던 점",)),
        ("what_felt_boring_or_hard", ("지루하거나 읽기 어려웠던 점", "cross-branch issues")),
        ("fixer_priority_actions", ("fixer priority actions", "priority actions")),
        ("style_tips", ("문체 개선 팁",)),
        ("score_strategy", ("score strategy",)),
        ("periodic_diagnosis", ("periodic diagnosis",)),
    ]
    out = {
        "what_felt_good": [],
        "what_felt_boring_or_hard": [],
        "fixer_priority_actions": [],
        "style_tips": [],
        "score_strategy": [],
        "periodic_diagnosis": [],
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
            title_lower = title.lower()
            for key, markers in section_markers:
                if any(marker.lower() in title_lower for marker in markers):
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

    merged_style_tips: list[str] = []
    for key in ("fixer_priority_actions", "style_tips", "score_strategy", "periodic_diagnosis"):
        vals = out.get(key, [])
        if isinstance(vals, list):
            merged_style_tips.extend(v for v in vals if str(v or "").strip())
    if merged_style_tips:
        out["style_tips"] = merged_style_tips

    cleaned = {
        k: v for k, v in out.items()
        if (isinstance(v, list) and v) or (isinstance(v, str) and v.strip())
    }
    repetition_terms = _extract_repetition_watch_terms(
        (out.get("what_felt_boring_or_hard", []) or [])
        + (out.get("fixer_priority_actions", []) or [])
        + (out.get("style_tips", []) or [])
        + (out.get("score_strategy", []) or [])
        + (out.get("periodic_diagnosis", []) or [])
    )
    if repetition_terms:
        cleaned["repetition_watch_terms"] = repetition_terms
    jargon_terms = _extract_jargon_watch_terms(
        (out.get("what_felt_boring_or_hard", []) or [])
        + (out.get("fixer_priority_actions", []) or [])
        + (out.get("style_tips", []) or [])
        + (out.get("score_strategy", []) or [])
        + (out.get("periodic_diagnosis", []) or [])
        + [out.get("reader_comment", "")]
    )
    if jargon_terms:
        cleaned["jargon_watch_terms"] = jargon_terms
    style_constraints = _extract_style_constraints(
        (out.get("what_felt_boring_or_hard", []) or [])
        + (out.get("fixer_priority_actions", []) or [])
        + (out.get("style_tips", []) or [])
        + (out.get("score_strategy", []) or [])
        + (out.get("periodic_diagnosis", []) or [])
        + [out.get("reader_comment", "")]
    )
    if style_constraints:
        cleaned["style_constraints"] = style_constraints
    return cleaned


def build_feedback_prompt_block(review: dict, max_items: int = 3) -> str:
    """
    Convert parsed review fields into a compact prompt-ready guidance block.
    """
    if not isinstance(review, dict) or not review:
        return ""

    lines: list[str] = []
    weak = _select_priority_items(review.get("what_felt_boring_or_hard", []) or [], max_items)
    fixer_actions = _select_priority_items(review.get("fixer_priority_actions", []) or [], max_items)
    style_tips = _select_priority_items(review.get("style_tips", []) or [], max_items)
    score_strategy = _select_priority_items(review.get("score_strategy", []) or [], max_items)
    periodic_diagnosis = _select_priority_items(review.get("periodic_diagnosis", []) or [], max_items)

    tips: list[str] = []
    for bucket in (fixer_actions, style_tips, score_strategy, periodic_diagnosis):
        for item in bucket:
            if item in tips:
                continue
            tips.append(item)
            if len(tips) >= max_items:
                break
        if len(tips) >= max_items:
            break

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


def count_feedback_term_occurrences(text: str, term: str) -> int:
    """
    Count term occurrences robustly across punctuation/spacing variants.
    This helps detect repeats like "그러면, 정확히" vs "그러면 정확히".
    """
    base = str(text or "").lower()
    token = str(term or "").strip().lower()
    if not base or not token:
        return 0

    norm_base = re.sub(r"\s+", " ", base).strip()
    norm_token = re.sub(r"\s+", " ", token).strip()
    compact_base = re.sub(r"\s+", "", base)
    compact_token = re.sub(r"\s+", "", token)
    alnum_base = re.sub(r"[^0-9a-z가-힣]+", "", base)
    alnum_token = re.sub(r"[^0-9a-z가-힣]+", "", token)
    punct_norm_base = re.sub(r"[^0-9a-z가-힣]+", " ", base).strip()
    punct_norm_token = re.sub(r"[^0-9a-z가-힣]+", " ", token).strip()

    counts = [
        norm_base.count(norm_token) if norm_token else 0,
        compact_base.count(compact_token) if compact_token else 0,
        alnum_base.count(alnum_token) if alnum_token else 0,
        punct_norm_base.count(punct_norm_token) if punct_norm_token else 0,
    ]
    return max(counts)


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
        (r"기술|용어|약자|약어|전문|jargon|acronym", 4),
        (r"처음 등장|첫 등장|첫 언급|초기 언급|첫 노출", 4),
        (r"괄호|정의|풀어쓰기|plain-language|plain language", 4),
        (r"비유|예시|일상", 3),
        (r"반복|중복|늘어지|같은|유의어", 4),
        (r"긴 문단|문단|호흡|리듬|속도|단조|단조롭|비슷한 리듬|같은 리듬", 4),
        (r"쪼개|분할|줄바꿈|목록|나열|단문", 4),
        (r"누가 말|화자|대화 전환|헷갈|speaker|dialogue", 5),
        (r"행동 태그|행동문|addressee|화자 식별", 5),
        (r"인물|역할|구분|호칭|이름", 5),
        (r"심리|내면|설명적|감정선|표정|행동|보여주", 4),
        (r"장면 전환|전환|복도|발표장|흐름", 3),
        (r"증거|목록|목록처럼|나열|요약|흐름", 3),
        (r"정보 전달 위주|설명 위주|감정의 고저|감정 고저|임팩트", 4),
    ]
    for pattern, weight in patterns:
        if re.search(pattern, t, flags=re.IGNORECASE):
            score += weight
    return score


def _fingerprint(text: str) -> str:
    norm = re.sub(r"[^0-9a-z가-힣\s]", " ", str(text or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _extract_repetition_watch_terms(items: list[str], max_terms: int = 10) -> list[str]:
    """
    Pull likely repeated terms from reader complaints, especially parenthetical
    term lists like "(실시간, 보상, 드리프트 등)".
    """
    if not items:
        return []

    stop_terms = {
        "등", "같은", "의미", "표현", "단어", "용어", "기술", "전문", "문장",
        "문단", "호흡", "리듬", "속도", "설명", "감정", "장면", "전환", "독자",
        "반복", "중복", "느낌", "부분", "내용", "정보", "핵심어",
        "묘사", "행동",
        "중요한", "간단한", "자연스러운", "효과적인", "신선한", "분명한",
        "부담", "흐름", "구간", "독해", "분위기", "임팩트", "감각", "비유", "감각비유", "감각 비유",
        "이미지",
    }

    candidates: list[str] = []

    def _extract_topic_starters(sample: str) -> list[str]:
        """
        Pull short Korean topic-subject starters often cited in repetition complaints
        (e.g., "수민은 ...했다" -> "수민은").
        """
        hits: list[str] = []
        for m in re.findall(r"([가-힣A-Za-z][가-힣A-Za-z0-9_]{0,11}(?:은|는|이|가))", sample):
            token = _normalize_candidate_term(m)
            if token:
                hits.append(token)
        return hits

    def _normalize_candidate_term(
        raw: str,
        *,
        max_words: int = 2,
        keep_hyphen: bool = False,
        allow_single_hangul: bool = False,
    ) -> str:
        t = str(raw or "").strip().strip("\"'“”‘’")
        t = re.sub(r"^(?:예시?|e\.g\.?)\s*[:：]?\s*", "", t, flags=re.IGNORECASE)
        if keep_hyphen:
            t = re.sub(r"[^0-9a-zA-Z가-힣\-\s]+", " ", t)
        else:
            t = re.sub(r"[^0-9a-zA-Z가-힣\s]+", " ", t)
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"^\W+|\W+$", "", t)
        t = re.sub(r"(은|는|이|가|을|를|에|에서|으로|와|과)$", "", t).strip()
        t = re.sub(r"\s*등$", "", t).strip()
        # Drop trailing meta-qualifiers so "메모 반복" becomes "메모".
        t = re.sub(r"\s+(반복|중복|과다|과잉|묘사|표현)$", "", t, flags=re.IGNORECASE).strip()
        # Drop broad hint wrappers so we keep concrete motif terms.
        t = re.sub(r"^(?:비슷한|유사한|같은)\s+", "", t, flags=re.IGNORECASE).strip()
        if re.search(r"(?:의미|유사|비슷한)\s*의\s*(?:행동|표현|묘사)$", t):
            return ""
        if not t:
            return ""
        # Drop broad compound phrases ("설명이나 감정", "A 또는 B").
        if re.search(r"[가-힣A-Za-z0-9]+(?:이나|거나)\s+[가-힣A-Za-z0-9]+", t):
            return ""
        if re.search(r"\s(?:또는|및|와|과)\s", f" {t} "):
            return ""
        low = t.lower()
        # Drop instruction-like fragments accidentally captured from guidance bullets.
        if re.search(r"(하라|해라|하라\.|해라\.|하라$|해라$|하시오|하십시오)$", low):
            return ""
        if re.search(r"(?:으로|로)\s+(?:의미|리듬|호흡|화자|핵심)", low):
            return ""
        if low in stop_terms or len(low) > 24:
            return ""
        if len(low) < 2:
            if not (allow_single_hangul and re.fullmatch(r"[가-힣]", low)):
                return ""
        if low.endswith("적인") or low.endswith("스럽다") or low.endswith("스럽게"):
            return ""
        if len(low.split()) > max(1, max_words):
            return ""
        if not re.search(r"[0-9a-z가-힣]", low):
            return ""
        if re.search(r"[.!?]$", low):
            return ""
        return t

    def _collect_tokens(chunk_text: str, *, allow_single_hangul: bool = False) -> None:
        normalized = re.sub(
            r"(?<=[0-9A-Za-z가-힣])(와|과)(?=\s+[0-9A-Za-z가-힣])",
            r" \1 ",
            str(chunk_text),
        )
        for token in re.split(r"[,/|·;:]| 및 | 와 | 과 ", normalized):
            t = _normalize_candidate_term(token, allow_single_hangul=allow_single_hangul)
            if t:
                candidates.append(t)
        # Also split Korean conjunction-linked motifs (e.g., "공기 변화나 심장 박동 묘사").
        for token in re.split(r"\s*(?:이나|나|혹은|또는|및)\s*", chunk_text):
            t = _normalize_candidate_term(token, allow_single_hangul=allow_single_hangul)
            if t:
                candidates.append(t)

    def _collect_phrase_hints(sample: str) -> None:
        """
        Collect lightweight phrase hints even when reviews do not use quotes/parentheses.
        Examples:
        - "반복되는 설명" -> "설명"
        - "비슷한 표현으로 여러 번 반복" -> "표현"
        - "기술 단어나 반복되는 설명" -> "기술 단어", "설명"
        """
        normalized = re.sub(r"\s+", " ", sample).strip()
        if not normalized:
            return

        patterns = [
            r"(?:반복(?:되는)?|중복(?:되는)?|비슷한)\s+([가-힣A-Za-z][가-힣A-Za-z0-9 ]{1,18})",
            r"([가-힣A-Za-z][가-힣A-Za-z0-9 ]{1,18})\s*(?:의|이|가|을|를|은|는)?\s*(?:반복|중복)",
            r"(?:같은|유사한)\s+([가-힣A-Za-z][가-힣A-Za-z0-9 ]{1,18})",
        ]
        for pattern in patterns:
            for m in re.findall(pattern, normalized, flags=re.IGNORECASE):
                chunk = str(m or "").strip()
                if not chunk:
                    continue
                chunk = re.sub(
                    r"\s*(?:구간|부분|문장|문단|설명|표현|어휘|용어)?\s*(?:으로|에서|가|이|은|는|을|를)?$",
                    "",
                    chunk,
                    flags=re.IGNORECASE,
                ).strip()
                t = _normalize_candidate_term(chunk)
                if t:
                    candidates.append(t)
        # Capture parenthetical example terms after "예:" even when commas are absent.
        for m in re.findall(r"예\s*[:：]\s*([가-힣A-Za-z0-9 ]{2,36})", normalized, flags=re.IGNORECASE):
            _collect_tokens(str(m))
        for token in re.findall(r"[가-힣A-Za-z]{2,12}", normalized):
            if len(token) < 3:
                continue
            if not any(k in token for k in ("용어", "약어", "표현", "문장", "대화", "말투", "어투", "톤")):
                continue
            t = _normalize_candidate_term(token)
            if t:
                candidates.append(t)

    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        low_text = text.lower()
        has_repetition_signal = any(
            k in low_text for k in ("반복", "중복", "같은 의미", "유의어", "늘어지")
        )
        quoted_terms = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,30})[\"“”'‘’]", text)
        if not has_repetition_signal:
            continue
        for q in quoted_terms:
            for starter in _extract_topic_starters(q):
                candidates.append(starter)
            t = _normalize_candidate_term(q, max_words=4, keep_hyphen=True)
            if t:
                candidates.append(t)
        for chunk in re.findall(r"\(([^)]+)\)", text):
            # Parenthetical examples often include 1-syllable motif words (e.g., "펜", "빛").
            _collect_tokens(chunk, allow_single_hangul=True)
        _collect_phrase_hints(text)

    out: list[str] = []
    seen: set[str] = set()
    for t in candidates:
        fp = _fingerprint(t)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _extract_jargon_watch_terms(items: list[str], max_terms: int = 10) -> list[str]:
    """
    Pull concrete technical terms readers found hard to parse.
    Focus on examples listed in parenthesis/quotes and mixed-script terms.
    """
    if not items:
        return []

    generic = {
        "기술", "기술 용어", "전문 용어", "용어", "약어", "약자", "설명", "지식", "과학",
        "속도감", "리듬", "문장", "문단", "이해", "독자", "표현", "내용",
        "전문 용어 예", "기술 용어 예",
    }
    candidates: list[str] = []

    def _clean(raw: str) -> str:
        t = str(raw or "").strip().strip("\"'“”‘’")
        t = re.sub(r"^(?:예시?|e\.g\.?)\s*[:：]?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"[(){}\[\]]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            return ""
        # Normalize unicode subscript digits so T₂ -> T2 for matching.
        t = t.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
        if len(t) > 32:
            return ""
        low = t.lower()
        if low in generic:
            return ""
        if "전문 용어" in low or "기술 용어" in low:
            return ""
        if not re.search(r"[A-Za-z가-힣]", t):
            return ""
        # Drop sentence-like fragments ("...바꿔 리듬을 살려라") that are guidance, not terms.
        if re.search(r"[.!?]$", t):
            return ""
        if len(t.split()) > 3:
            return ""
        if re.search(r"(?:\s|^)(?:은|는|이|가|을|를|에|에서|으로|로|도|때|땐|듯|처럼)(?:\s|$)", t):
            return ""
        if re.search(r"(?:다|요|죠|라|자)$", t):
            return ""
        # Favor concrete terms (letters + digits, multi-word compounds, or Korean domain tokens).
        has_alpha_num = bool(re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", t))
        has_compound = bool(re.search(r"\s", t))
        has_korean_domain = bool(re.search(r"(드리프트|보상|레이어|코히런스|위상|지연|프로토콜|회로|계층)", t))
        has_caps_acronym = bool(re.search(r"\b[A-Z]{2,}[A-Z0-9-]*\b", t))
        if not (has_alpha_num or has_korean_domain or has_caps_acronym):
            return ""
        if has_compound and not has_korean_domain:
            return ""
        return t

    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        low = text.lower()
        if not any(k in low for k in ("기술", "용어", "약어", "약자", "전문", "jargon", "acronym")):
            # Still allow explicit examples to be captured.
            if "예:" not in text and "e.g." not in low:
                continue

        quoted = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,40})[\"“”'‘’]", text)
        parenthetical = re.findall(r"\(([^)]+)\)", text)
        examples = re.findall(r"(?:예시?|e\.g\.?)\s*[:：]\s*([^.;\n]{2,60})", text, flags=re.IGNORECASE)
        chunks = quoted + parenthetical + examples

        for chunk in chunks:
            normalized = chunk.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
            for token in re.split(r"[,/|·;:]| 및 | 와 | 과 | 또는 | 혹은 ", normalized):
                term = _clean(token)
                if term:
                    candidates.append(term)
        # Capture common acronym-like technical terms even without quotes/parentheses.
        for token in re.findall(r"\b[A-Z]{2,}[A-Z0-9-]{0,15}\b", text):
            term = _clean(token)
            if term:
                candidates.append(term)
        # Capture lowercase/mixed English technical keywords often written without acronyms.
        for token in re.findall(r"\b[a-z][a-z0-9_-]{2,20}\b", low):
            if token in {"with", "from", "into", "then", "when", "only", "once", "line", "block"}:
                continue
            if not any(k in token for k in ("latency", "drift", "coherence", "protocol", "throughput", "quantum", "metric")):
                continue
            term = _clean(token)
            if term:
                candidates.append(term)

    out: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        key = _fingerprint(term)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def _extract_style_constraints(items: list[str]) -> dict:
    """
    Extract numeric style constraints from review prose.
    Example signals:
    - "한 문단당 1회"
    - "40~70자"
    - "짧은 문장:긴 문장 = 1:2"
    - "문단 길이 임계값(예: 3문장 초과)"
    - "정보량 많은 문장은 2문장 이하"
    - "문단당 용어 2개 이하"
    """
    if not items:
        return {}

    merged = "\n".join(str(x or "") for x in items if str(x or "").strip())
    if not merged:
        return {}

    out: dict[str, object] = {}

    # paragraph term repetition cap
    m_rep = re.search(r"한\s*문단당\s*([0-9]{1,2})\s*회", merged)
    if m_rep:
        cap = int(m_rep.group(1))
        if 1 <= cap <= 5:
            out["max_term_repeats_per_paragraph"] = cap
    m_scene_rep = re.search(
        r"장면당[^0-9\n]{0,16}?([0-9]{1,2})(?:\s*[~\-–]\s*([0-9]{1,2}))?\s*회",
        merged,
    )
    if m_scene_rep:
        lo = int(m_scene_rep.group(1))
        hi = int(m_scene_rep.group(2) or lo)
        if lo > hi:
            lo, hi = hi, lo
        if 1 <= lo <= 8 and 1 <= hi <= 8:
            out["max_term_repeats_per_scene"] = hi

    # target sentence length window by Korean characters
    m_chars = re.search(r"([0-9]{2,3})\s*[~\-–]\s*([0-9]{2,3})\s*자", merged)
    if m_chars:
        lo = int(m_chars.group(1))
        hi = int(m_chars.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if 20 <= lo <= 200 and 20 <= hi <= 260:
            out["sentence_chars_min"] = lo
            out["sentence_chars_max"] = hi

    # short:long sentence rhythm ratio (e.g., 1:2)
    m_ratio = re.search(r"짧은\s*문장\s*[:：]\s*긴\s*문장\s*=\s*([0-9]{1,2})\s*[:：]\s*([0-9]{1,2})", merged)
    if m_ratio:
        short_n = int(m_ratio.group(1))
        long_n = int(m_ratio.group(2))
        if short_n > 0 and long_n > 0:
            out["short_to_long_sentence_ratio"] = [short_n, long_n]

    # paragraph sentence cap (e.g., "3문장 초과 시 분할")
    m_par_cap = re.search(r"([0-9]{1,2})\s*문장\s*(?:초과|이상)\s*\)?\s*(?:시|이면|일 때)?\s*(?:자동\s*)?(?:분할|쪼개|분리)", merged)
    if m_par_cap:
        cap = int(m_par_cap.group(1))
        if 1 <= cap <= 8:
            out["max_sentences_per_paragraph"] = cap
    else:
        m_par_cap2 = re.search(r"문단(?:당|)\s*([0-9]{1,2})\s*문장\s*이하", merged)
        if m_par_cap2:
            cap = int(m_par_cap2.group(1))
            if 1 <= cap <= 8:
                out["max_sentences_per_paragraph"] = cap

    # dense info line cap (e.g., "정보량 많은 문장은 2문장 이하")
    m_dense = re.search(r"정보량[^0-9\n]{0,24}?([0-9]{1,2})\s*문장\s*이하", merged)
    if m_dense:
        dense_cap = int(m_dense.group(1))
        if 1 <= dense_cap <= 5:
            out["max_sentences_in_dense_info"] = dense_cap

    # jargon density cap (e.g., "문단당 용어 2개 이하")
    m_jargon = re.search(
        r"문단당[^0-9\n]{0,16}?(?:기술\s*용어|전문\s*용어|용어)[^0-9\n]{0,8}?([0-9]{1,2})\s*개\s*이하",
        merged,
    )
    if m_jargon:
        jargon_cap = int(m_jargon.group(1))
        if 1 <= jargon_cap <= 8:
            out["max_jargon_terms_per_paragraph"] = jargon_cap

    # short beat sentence char window and per-scene frequency
    m_short_chars = re.search(
        r"(?:짧은\s*단문|단문|짧은\s*문장)[^0-9\n]{0,24}?\(?\s*([0-9]{1,2})\s*[~\-–]\s*([0-9]{1,2})\s*자",
        merged,
    )
    if m_short_chars:
        lo = int(m_short_chars.group(1))
        hi = int(m_short_chars.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if 3 <= lo <= 24 and 3 <= hi <= 36:
            out["short_beat_chars_min"] = lo
            out["short_beat_chars_max"] = hi
    m_short_freq = re.search(
        r"(?:짧은\s*단문|단문|짧은\s*문장)[^0-9\n]{0,24}?장면당[^0-9\n]{0,8}?([0-9]{1,2})\s*[~\-–]\s*([0-9]{1,2})\s*회",
        merged,
    )
    if m_short_freq:
        lo = int(m_short_freq.group(1))
        hi = int(m_short_freq.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if 0 <= lo <= 10 and 0 <= hi <= 10:
            out["short_beats_per_scene_min"] = lo
            out["short_beats_per_scene_max"] = hi

    # transition line character window (e.g., "연결문 ... 10~15자")
    m_transition = re.search(
        r"(?:연결문|전환문)[^0-9\n]{0,24}?\(?\s*([0-9]{1,2})\s*[~\-–]\s*([0-9]{1,2})\s*자",
        merged,
    )
    if m_transition:
        lo = int(m_transition.group(1))
        hi = int(m_transition.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if 5 <= lo <= 40 and 5 <= hi <= 40:
            out["transition_chars_min"] = lo
            out["transition_chars_max"] = hi

    # speaker/name refresh cadence
    m_name_refresh = re.search(
        r"([0-9]{1,2})\s*문장\s*연속[^.\n]{0,40}?(?:이름\s*반복|이름\s*삽입|호칭\s*반복)",
        merged,
    )
    if m_name_refresh:
        streak = int(m_name_refresh.group(1))
        if 2 <= streak <= 8:
            out["speaker_refresh_streak"] = streak

    return out
