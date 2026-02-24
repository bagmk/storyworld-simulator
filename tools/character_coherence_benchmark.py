#!/usr/bin/env python3
"""
Benchmark coherence across episodes (default: ep01-ep05).

Measures:
- transition continuity across every adjacent pair (1->2, 2->3, 3->4, 4->5)
- global character voice coherence across the full episode set
- protagonist persistence and cast overlap stability
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import yaml


@dataclass
class EpisodeMeta:
    number: int
    config_path: Path
    episode_id: str
    clues: list[str]


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def parse_episode_meta(paths: list[Path]) -> list[EpisodeMeta]:
    metas: list[EpisodeMeta] = []
    for p in paths:
        raw = load_yaml(p)
        ep = raw.get("episode", raw)
        if not isinstance(ep, dict):
            continue
        eid = str(ep.get("id", "")).strip()
        m = re.search(r"ep(\d+)", p.stem)
        num = int(m.group(1)) if m else 0
        clues = []
        for c in ep.get("introduced_clues", []):
            if isinstance(c, dict):
                txt = str(c.get("content", "")).strip()
                if txt:
                    clues.append(txt)
            elif isinstance(c, str):
                txt = c.strip()
                if txt:
                    clues.append(txt)
        metas.append(EpisodeMeta(number=num, config_path=p, episode_id=eid, clues=clues))
    metas.sort(key=lambda x: x.number)
    return metas


def load_character_index(characters_yaml: Path) -> dict[str, set[str]]:
    data = load_yaml(characters_yaml)
    out: dict[str, set[str]] = {}
    korean_alias_overrides: dict[str, set[str]] = {
        "kim_sumin": {"수민"},
        "elena_ramirez": {"엘레나"},
        "alex_moreno": {"모레노"},
        "carlos_reyes": {"카를로스"},
        "el_patron": {"패트론", "라파엘"},
        "ben_clarke": {"벤"},
        "hyejin_kim": {"혜진"},
        "marcelo": {"마르셀로"},
    }
    for row in data.get("characters", []):
        if not isinstance(row, dict):
            continue
        cid = str(row.get("id", "")).strip()
        if not cid:
            continue
        names = {
            str(row.get("name", "")).strip().lower(),
            str(row.get("id", "")).strip().lower(),
        }
        for piece in re.split(r"[\s_]+", str(row.get("name", "")).strip().lower()):
            if len(piece) >= 3:
                names.add(piece)
        for piece in re.split(r"[\s_]+", cid.lower()):
            if len(piece) >= 3:
                names.add(piece)
        for alias in row.get("aliases", []) or []:
            names.add(str(alias).strip().lower())
        for ko_alias in korean_alias_overrides.get(cid, set()):
            names.add(ko_alias.lower())
        names = {n for n in names if n}
        out[cid] = names
    return out


def resolve_chapter_file(ep: EpisodeMeta, chapters_dir: Path, fallback_dirs: list[Path]) -> Optional[Path]:
    candidates: list[Path] = []
    patterns = [
        f"{ep.episode_id}_chapter.md",
        f"ep{ep.number:02d}_*_chapter.md",
        f"ep{ep.number:02d}_*.md",
    ]
    search_dirs = [chapters_dir, *fallback_dirs]
    for root in search_dirs:
        if not root.exists():
            continue
        for pat in patterns:
            candidates.extend(sorted(root.glob(pat)))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.name), p.stat().st_mtime), reverse=True)
    return candidates[0]


def strip_markdown_envelope(text: str) -> str:
    txt = text
    txt = re.sub(r"^\s*#.*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^\s*\*Episode:.*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^\s*---\s*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"\n\s*\*Scene structure:\*[\s\S]*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"\n\s*\*Evidence ledger:\*[\s\S]*$", "", txt, flags=re.MULTILINE)
    return txt.strip()


def extract_present_characters(text: str, char_index: dict[str, set[str]]) -> set[str]:
    low = text.lower()
    present: set[str] = set()
    for cid, aliases in char_index.items():
        if any(alias and alias in low for alias in aliases):
            present.add(cid)
    return present


def count_character_mentions(text: str, char_index: dict[str, set[str]]) -> dict[str, int]:
    low = text.lower()
    counts: dict[str, int] = {}
    for cid, aliases in char_index.items():
        n = 0
        for alias in aliases:
            if not alias:
                continue
            n += low.count(alias)
        if n > 0:
            counts[cid] = n
    return counts


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
    nb = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _extract_keywords(clues: list[str]) -> list[str]:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "were", "into",
        "about", "have", "will", "then", "when", "where", "which",
    }
    toks: list[str] = []
    for clue in clues:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[0-9]{3,}|[A-Z]{2,}", clue):
            t = tok.strip().lower()
            if t and t not in stop:
                toks.append(t)
    seen = set()
    out: list[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:20]


def clue_memory_score(prev_clues: list[str], current_text: str) -> float:
    keywords = _extract_keywords(prev_clues)
    if not keywords:
        return 0.6
    low = current_text.lower()
    hit = sum(1 for kw in keywords if kw in low)
    denom = min(8, len(keywords))
    return min(1.0, hit / max(1, denom))


def full_text_coherence_benchmark(
    metas: list[EpisodeMeta],
    chapter_text_by_eid: dict[str, str],
    char_index: dict[str, set[str]],
    protagonist_id: str,
) -> dict[str, Any]:
    """
    Whole-corpus benchmark using concatenated text (not averaging per-episode totals).
    Measures continuity signals directly from the full 1..N corpus.
    """
    ordered_texts: list[str] = []
    for ep in metas:
        txt = chapter_text_by_eid.get(ep.episode_id, "")
        if txt.strip():
            ordered_texts.append(txt.strip())
    corpus = "\n\n".join(ordered_texts).strip()
    if not corpus:
        return {
            "score": 0.0,
            "has_corpus": False,
        }

    corpus_low = corpus.lower()
    corpus_words = re.findall(r"[A-Za-z0-9_]+|[가-힣]+", corpus)
    word_count = len(corpus_words)

    # 1) Character recurrence across the whole corpus
    mention_counts = count_character_mentions(corpus, char_index)
    recurring_chars = {cid: n for cid, n in mention_counts.items() if n >= 3}
    recurring_char_score = min(1.0, len(recurring_chars) / 6.0)

    # 2) Corpus-wide clue/thread callback coverage:
    #    use clue keywords from episodes 1..N-1 and check if they recur anywhere in the corpus.
    all_prior_keywords: list[str] = []
    for ep in metas[:-1]:
        all_prior_keywords.extend(_extract_keywords(ep.clues))
    dedup_keywords: list[str] = []
    seen_kw = set()
    for kw in all_prior_keywords:
        if kw in seen_kw:
            continue
        seen_kw.add(kw)
        dedup_keywords.append(kw)
    callback_hits = sum(1 for kw in dedup_keywords[:30] if kw in corpus_low)
    callback_score = (
        min(1.0, callback_hits / max(1, min(12, len(dedup_keywords))))
        if dedup_keywords else 0.6
    )

    # 3) Whole-corpus transition clarity density
    transition_markers = re.findall(
        r"잠시 후|그 후|이후|다음 날|그날 밤|며칠 후|한편|곧이어",
        corpus
    )
    transition_density = (len(transition_markers) / max(1, word_count)) * 1000.0
    # Target range: ~1 to 8 markers per 1000 words across serialized prose
    if 1.0 <= transition_density <= 8.0:
        transition_density_score = 1.0
    elif transition_density < 1.0:
        transition_density_score = max(0.0, transition_density / 1.0)
    else:
        transition_density_score = max(0.0, 1.0 - ((transition_density - 8.0) / 8.0))

    # 4) Protagonist anchor in full corpus
    protagonist_aliases = char_index.get(protagonist_id, {protagonist_id})
    protagonist_mentions = sum(corpus_low.count(a) for a in protagonist_aliases if a)
    protagonist_density = (protagonist_mentions / max(1, word_count)) * 1000.0
    protagonist_anchor_score = min(1.0, protagonist_density / 8.0)

    score = (
        recurring_char_score * 0.35
        + callback_score * 0.30
        + transition_density_score * 0.15
        + protagonist_anchor_score * 0.20
    )
    return {
        "score": round(score, 3),
        "has_corpus": True,
        "word_count": word_count,
        "recurring_character_count": len(recurring_chars),
        "recurring_character_score": round(recurring_char_score, 3),
        "clue_callback_keyword_hits": callback_hits,
        "clue_callback_score": round(callback_score, 3),
        "transition_markers": len(transition_markers),
        "transition_density_per_1000_words": round(transition_density, 3),
        "transition_density_score": round(transition_density_score, 3),
        "protagonist_mentions": protagonist_mentions,
        "protagonist_density_per_1000_words": round(protagonist_density, 3),
        "protagonist_anchor_score": round(protagonist_anchor_score, 3),
    }


def load_emotion_rows(conn: sqlite3.Connection, episode_ids: list[str]) -> list[sqlite3.Row]:
    if not episode_ids:
        return []
    placeholders = ",".join("?" for _ in episode_ids)
    return conn.execute(
        f"""
        SELECT episode_id, turn, agent_id, emotion_type, intensity
        FROM emotions
        WHERE episode_id IN ({placeholders})
        ORDER BY episode_id, turn
        """,
        episode_ids,
    ).fetchall()


def build_emotion_vectors(
    rows: list[sqlite3.Row],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, dict[str, float]]]]:
    start: dict[str, dict[str, dict[str, float]]] = {}
    final: dict[str, dict[str, dict[str, float]]] = {}
    min_turn: dict[tuple[str, str], int] = {}
    max_turn: dict[tuple[str, str], int] = {}

    for r in rows:
        eid = str(r["episode_id"])
        agent = str(r["agent_id"])
        turn = int(r["turn"] or 0)
        emo = str(r["emotion_type"])
        val = float(r["intensity"] or 0.0)
        key = (eid, agent)

        if key not in min_turn or turn < min_turn[key]:
            min_turn[key] = turn
            start.setdefault(eid, {})[agent] = {}
        if turn == min_turn[key]:
            start.setdefault(eid, {}).setdefault(agent, {})[emo] = val

        if key not in max_turn or turn >= max_turn[key]:
            if key not in max_turn or turn > max_turn[key]:
                final.setdefault(eid, {})[agent] = {}
            max_turn[key] = turn
            final.setdefault(eid, {}).setdefault(agent, {})[emo] = val

    return start, final


def load_relationship_map(
    conn: sqlite3.Connection,
    episode_ids: list[str],
    protagonist_id: str,
) -> dict[str, dict[str, float]]:
    if not episode_ids:
        return {}
    placeholders = ",".join("?" for _ in episode_ids)
    rows = conn.execute(
        f"""
        SELECT episode_id, agent2_id, value
        FROM relationships
        WHERE episode_id IN ({placeholders})
          AND agent1_id = ?
        """,
        [*episode_ids, protagonist_id],
    ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out.setdefault(str(r["episode_id"]), {})[str(r["agent2_id"])] = float(r["value"])
    return out


def extract_character_dialogue_features(
    text: str,
    char_index: dict[str, set[str]],
) -> dict[str, dict[str, float]]:
    lines = re.findall(r'"([^"]{3,})"', text)
    feats: dict[str, list[dict[str, float]]] = {cid: [] for cid in char_index}

    for m in re.finditer(r'"([^"]{3,})"', text):
        line = m.group(1).strip()
        if not line:
            continue
        start = max(0, m.start() - 60)
        ctx = text[start:m.start()].lower()
        speaker: Optional[str] = None
        for cid, aliases in char_index.items():
            if any(a in ctx for a in aliases):
                speaker = cid
                break
        if speaker is None:
            continue

        words = re.findall(r"[A-Za-z0-9_]+|[가-힣]+", line)
        wcount = len(words)
        qrate = 1.0 if ("?" in line or "?" in ctx) else 0.0
        erate = 1.0 if "!" in line else 0.0
        tech_hits = len(re.findall(r"data|model|protocol|grant|contract|evidence|nsa|fbi|lab|quantum", line.lower()))
        feats[speaker].append({
            "avg_words": float(wcount),
            "question_rate": qrate,
            "exclaim_rate": erate,
            "tech_rate": float(tech_hits) / max(1.0, float(wcount)),
        })

    out: dict[str, dict[str, float]] = {}
    for cid, rows in feats.items():
        if not rows:
            continue
        out[cid] = {
            "avg_words": mean(r["avg_words"] for r in rows),
            "question_rate": mean(r["question_rate"] for r in rows),
            "exclaim_rate": mean(r["exclaim_rate"] for r in rows),
            "tech_rate": mean(r["tech_rate"] for r in rows),
            "line_count": float(len(rows)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark full-series character coherence")
    p.add_argument("--chapters-dir", default="output", help="Directory with generated chapter markdown files")
    p.add_argument("--fallback-chapters-dir", action="append", default=[], help="Optional fallback chapter directories")
    p.add_argument("--characters", default="config/characters.yaml", help="characters.yaml path")
    p.add_argument("--db", default="data/simulation.db", help="SQLite DB path")
    p.add_argument("--episodes", nargs="*", default=[
        "config/episodes/ep01_academic_presentation.yaml",
        "config/episodes/ep02_ben_encounter.yaml",
        "config/episodes/ep03_unexpected_visitors.yaml",
        "config/episodes/ep04_patrons_doctrine.yaml",
        "config/episodes/ep05_between_faction.yaml",
    ], help="Episode config YAML paths in intended order")
    p.add_argument("--protagonist-id", default="kim_sumin", help="Protagonist agent id")
    p.add_argument("--report-out", default="", help="Output JSON report path")
    args = p.parse_args()

    episode_paths = [Path(x) for x in args.episodes]
    metas = parse_episode_meta(episode_paths)
    if len(metas) < 2:
        raise SystemExit("Need at least 2 episodes to benchmark coherence")

    chapters_dir = Path(args.chapters_dir)
    fallback_dirs = [Path(x) for x in args.fallback_chapters_dir]
    char_index = load_character_index(Path(args.characters))

    episodes_payload: list[dict[str, Any]] = []
    chapter_text_by_eid: dict[str, str] = {}
    cast_by_eid: dict[str, set[str]] = {}
    voice_by_eid: dict[str, dict[str, dict[str, float]]] = {}
    protagonist_presence: dict[str, bool] = {}

    for ep in metas:
        chapter_path = resolve_chapter_file(ep, chapters_dir, fallback_dirs)
        if chapter_path is None:
            episodes_payload.append({
                "episode_id": ep.episode_id,
                "episode_number": ep.number,
                "chapter_found": False,
            })
            continue

        text = strip_markdown_envelope(chapter_path.read_text(encoding="utf-8"))
        chapter_text_by_eid[ep.episode_id] = text
        cast = extract_present_characters(text, char_index)
        cast_by_eid[ep.episode_id] = cast
        voice_by_eid[ep.episode_id] = extract_character_dialogue_features(text, char_index)

        protagonist_aliases = char_index.get(args.protagonist_id, {args.protagonist_id})
        low = text.lower()
        protagonist_presence[ep.episode_id] = any(a in low for a in protagonist_aliases)

        episodes_payload.append({
            "episode_id": ep.episode_id,
            "episode_number": ep.number,
            "chapter_found": True,
            "chapter_path": str(chapter_path),
            "character_mentions_count": len(cast),
            "protagonist_present": protagonist_presence[ep.episode_id],
        })

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    eids = [m.episode_id for m in metas]
    emotion_rows = load_emotion_rows(conn, eids)
    start_emotions, final_emotions = build_emotion_vectors(emotion_rows)
    rel_map = load_relationship_map(conn, eids, protagonist_id=args.protagonist_id)
    conn.close()

    transitions: list[dict[str, Any]] = []
    transition_scores: list[float] = []

    for i in range(1, len(metas)):
        prev_ep = metas[i - 1]
        cur_ep = metas[i]
        prev_id = prev_ep.episode_id
        cur_id = cur_ep.episode_id

        cast_score = _jaccard(cast_by_eid.get(prev_id, set()), cast_by_eid.get(cur_id, set()))
        clue_score = clue_memory_score(prev_ep.clues, chapter_text_by_eid.get(cur_id, ""))

        emo_agent_scores: list[float] = []
        prev_final = final_emotions.get(prev_id, {})
        cur_start = start_emotions.get(cur_id, {})
        for aid in set(prev_final) & set(cur_start):
            emo_agent_scores.append(_cosine(prev_final.get(aid, {}), cur_start.get(aid, {})))
        emotion_score = mean(emo_agent_scores) if emo_agent_scores else 0.5

        prev_rel = rel_map.get(prev_id, {})
        cur_rel = rel_map.get(cur_id, {})
        common_pairs = set(prev_rel) & set(cur_rel)
        if common_pairs:
            rel_score = mean(max(0.0, 1.0 - abs(prev_rel[k] - cur_rel[k])) for k in common_pairs)
        else:
            rel_score = 0.5

        transition_score = (
            cast_score * 0.30
            + clue_score * 0.25
            + emotion_score * 0.25
            + rel_score * 0.20
        )
        transition_scores.append(transition_score)
        transitions.append({
            "from_episode": prev_id,
            "to_episode": cur_id,
            "cast_overlap_score": round(cast_score, 3),
            "clue_memory_score": round(clue_score, 3),
            "emotion_continuity_score": round(emotion_score, 3),
            "relationship_continuity_score": round(rel_score, 3),
            "transition_score": round(transition_score, 3),
        })

    char_series_vectors: dict[str, list[dict[str, float]]] = {}
    for eid, by_char in voice_by_eid.items():
        _ = eid
        for cid, feat in by_char.items():
            if feat.get("line_count", 0.0) < 2.0:
                continue
            char_series_vectors.setdefault(cid, []).append(feat)

    char_voice_scores: list[float] = []
    char_voice_details: list[dict[str, Any]] = []
    for cid, vectors in char_series_vectors.items():
        if len(vectors) < 2:
            continue
        sims: list[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                a = {k: v for k, v in vectors[i].items() if k != "line_count"}
                b = {k: v for k, v in vectors[j].items() if k != "line_count"}
                sims.append(_cosine(a, b))
        if not sims:
            continue
        score = mean(sims)
        char_voice_scores.append(score)
        char_voice_details.append({
            "character_id": cid,
            "episodes_covered": len(vectors),
            "voice_consistency_score": round(score, 3),
        })

    global_voice_score = mean(char_voice_scores) if char_voice_scores else 0.5
    protagonist_presence_rate = (
        sum(1 for x in protagonist_presence.values() if x) / max(1, len(metas))
    )
    transition_avg = mean(transition_scores) if transition_scores else 0.0
    full_text_benchmark = full_text_coherence_benchmark(
        metas=metas,
        chapter_text_by_eid=chapter_text_by_eid,
        char_index=char_index,
        protagonist_id=args.protagonist_id,
    )
    overall = (
        transition_avg * 0.70
        + global_voice_score * 0.20
        + protagonist_presence_rate * 0.10
    )
    overall_with_full_text = (
        overall * 0.75
        + float(full_text_benchmark.get("score", 0.0)) * 0.25
    )

    report = {
        "episode_ids": eids,
        "episodes": episodes_payload,
        "transitions": transitions,
        "global_character_voice": {
            "score": round(global_voice_score, 3),
            "details": char_voice_details,
        },
        "protagonist_presence_rate": round(protagonist_presence_rate, 3),
        "series_coherence_score": round(overall, 3),
        "full_text_coherence_benchmark": full_text_benchmark,
        "series_coherence_score_with_full_text": round(overall_with_full_text, 3),
    }

    out_path: Path
    if args.report_out:
        out_path = Path(args.report_out)
    else:
        out_path = Path("reports") / "character_coherence_ep01_ep05.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Report: {out_path}")
    print(f"Series coherence: {report['series_coherence_score']:.3f}")
    print(f"Full-text coherence: {report['full_text_coherence_benchmark']['score']:.3f}")
    print(f"Series coherence (with full-text): {report['series_coherence_score_with_full_text']:.3f}")
    print(f"Protagonist presence: {report['protagonist_presence_rate']:.3f}")
    print("Transition scores:")
    for t in transitions:
        print(
            f"  {t['from_episode']} -> {t['to_episode']}: "
            f"{t['transition_score']:.3f} "
            f"(cast={t['cast_overlap_score']:.3f}, clue={t['clue_memory_score']:.3f}, "
            f"emo={t['emotion_continuity_score']:.3f}, rel={t['relationship_continuity_score']:.3f})"
        )


if __name__ == "__main__":
    main()
