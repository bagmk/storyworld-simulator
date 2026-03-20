#!/usr/bin/env python3
"""
tools/analysis/factor_analysis.py

누적 Fixer diff → OpenAI 임베딩 → PCA → Lasso 회귀 → 강한 평가 보고서 생성.

매 5사이클 Manager 심층 회고 시 자동 실행.
실행할수록 데이터가 쌓여 분석이 정밀해짐:
  - 5번째:  ~5개 포인트  → 방향성 탐색
  - 10번째: ~10개 포인트 → Lasso 계수 신뢰도 상승
  - 15번째: ~15개 포인트 → 패턴 확정
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
CACHE_PATH = DATA_DIR / "factor_embedding_cache.json"
DATASET_PATH = DATA_DIR / "factor_dataset.json"

SCORE_KEYS = [
    "thrill_score_10",
    "style_score_10",
    "causality_score_10",
    "character_score_10",
    "scene_function_score_10",
]
SCORE_LABELS = ["긴장감", "문체", "인과성", "캐릭터", "씬기능"]

FIXER_TARGET_FILES = [
    "src/novel_writer/prose_generator.py",
    "src/novel_writer/scene_distiller.py",
    "src/novel_writer/director.py",
    "src/novel_writer/orchestrator.py",
    "generate_chapter.py",
    "simulate.py",
]

MIN_DATA_POINTS = 4  # 최소 데이터 포인트 수 (이 미만이면 PCA/회귀 생략)


# ── 1. 데이터 수집 ─────────────────────────────────────────────────────────────

def _load_scores(review_path: Path) -> dict[str, float] | None:
    """review JSON에서 5개 점수를 읽는다. 없는 항목은 None으로."""
    if not review_path.exists():
        return None
    try:
        d = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    scores = {k: float(d[k]) for k in SCORE_KEYS if k in d}
    return scores if scores else None


def _extract_diff_text(backup_dir: Path, compare_dir: Path | None) -> str:
    """
    backup_dir (수정 전) ↔ compare_dir (수정 후) 간 diff의 +/- 줄만 추출.
    compare_dir=None 이면 현재 REPO_ROOT 파일과 비교.
    """
    chunks: list[str] = []
    for rel in FIXER_TARGET_FILES:
        old_path = backup_dir / rel
        new_path = (compare_dir / rel) if compare_dir else (REPO_ROOT / rel)
        if not old_path.exists():
            continue
        if not new_path.exists():
            continue
        old_lines = old_path.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = new_path.read_text(encoding="utf-8", errors="replace").splitlines()
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
        changed = [l for l in diff if (l.startswith("+") or l.startswith("-"))
                   and not l.startswith("+++") and not l.startswith("---")]
        if changed:
            chunks.append(f"# {Path(rel).name}\n" + "\n".join(changed))
    return "\n\n".join(chunks)


def collect_dataset() -> list[dict]:
    """
    output/daily/ 전체에서 (diff_text, score_before, score_after) 쌍 수집.
    연속된 backup 디렉터리가 있어야 diff를 계산할 수 있음.
    결과를 data/factor_dataset.json 에 저장 후 반환.
    """
    records: list[dict] = []
    seen_keys: set[str] = set()

    # 기존 캐시 로드 (중복 제거용 키만 사용)
    if DATASET_PATH.exists():
        try:
            existing = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            for r in existing:
                seen_keys.add(r.get("id", ""))
        except Exception:
            pass

    daily_base = OUTPUT_DIR / "daily"
    if not daily_base.exists():
        return []

    for run_dir in sorted(daily_base.glob("*/*")):
        if not run_dir.is_dir():
            continue
        # 이 실행의 모든 backup 디렉터리 수집
        backups: dict[int, Path] = {}
        for bp in run_dir.glob("backup_before_fixer_cycle*"):
            m = re.search(r"cycle(\d+)$", bp.name)
            if m:
                backups[int(m.group(1))] = bp

        if not backups:
            continue

        sorted_cycles = sorted(backups.keys())
        for i, cycle_n in enumerate(sorted_cycles):
            backup_n = backups[cycle_n]

            # 다음 backup이 있으면 그걸 "수정 후" 로 사용, 없으면 현재 파일
            if i + 1 < len(sorted_cycles):
                cycle_next = sorted_cycles[i + 1]
                compare_dir = backups[cycle_next]
            else:
                compare_dir = None  # 현재 파일과 비교

            diff_text = _extract_diff_text(backup_n, compare_dir)
            if not diff_text.strip():
                continue

            # 점수: 수정 전 = auto_review_cycleN, 수정 후 = auto_review_cycle(N+1)
            score_before = _load_scores(run_dir / f"auto_review_cycle{cycle_n}.json")
            next_cycle = (sorted_cycles[i + 1] if i + 1 < len(sorted_cycles)
                         else cycle_n + 1)
            score_after = _load_scores(run_dir / f"auto_review_cycle{next_cycle}.json")

            record_id = hashlib.md5(
                f"{run_dir}|{cycle_n}".encode()
            ).hexdigest()[:12]

            if record_id in seen_keys:
                continue

            record: dict[str, Any] = {
                "id": record_id,
                "run_dir": str(run_dir),
                "cycle": cycle_n,
                "diff_text": diff_text,
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": None,
            }

            # delta 계산 (두 점수 모두 있을 때만)
            if score_before and score_after:
                delta = {}
                for k in SCORE_KEYS:
                    if k in score_before and k in score_after:
                        delta[k] = score_after[k] - score_before[k]
                if delta:
                    record["score_delta"] = delta

            records.append(record)
            seen_keys.add(record_id)

    # 기존 데이터와 합쳐서 저장
    all_records: list[dict] = []
    if DATASET_PATH.exists():
        try:
            all_records = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing_ids = {r.get("id") for r in all_records}
    for r in records:
        if r["id"] not in existing_ids:
            # diff_text는 크므로 저장 시 hash로 대체 (임베딩 캐시에서 매칭)
            all_records.append(r)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Factor dataset: %d total records saved", len(all_records))
    return all_records


# ── 2. 임베딩 ─────────────────────────────────────────────────────────────────

def _diff_hash(diff_text: str) -> str:
    return hashlib.md5(diff_text.encode()).hexdigest()


def embed_diffs(records: list[dict], openai_client: Any) -> dict[str, list[float]]:
    """
    각 record의 diff_text를 OpenAI embedding으로 변환.
    캐시(data/factor_embedding_cache.json)를 사용해 중복 API 호출 방지.
    반환: {diff_hash: embedding_vector}
    """
    # 캐시 로드
    cache: dict[str, list[float]] = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    to_embed: list[tuple[str, str]] = []  # (hash, text)
    for r in records:
        diff_text = r.get("diff_text", "")
        if not diff_text.strip():
            continue
        h = _diff_hash(diff_text)
        if h not in cache:
            # 임베딩 입력 길이 제한 (토큰 ~ 8191)
            truncated = diff_text[:12000]
            to_embed.append((h, truncated))

    if to_embed:
        logger.info("Embedding %d new diffs via OpenAI...", len(to_embed))
        texts = [t for _, t in to_embed]
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            for (h, _), emb_data in zip(to_embed, response.data):
                cache[h] = emb_data.embedding
        except Exception as e:
            logger.error("Embedding API error: %s", e)

        # 캐시 저장
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Embedding cache updated: %d total entries", len(cache))

    return cache


# ── 3. PCA + Lasso 분석 ───────────────────────────────────────────────────────

def run_factor_model(
    records: list[dict],
    embeddings: dict[str, list[float]],
) -> dict:
    """
    PCA + Lasso로 "어떤 방향의 코드 변경이 어떤 점수를 올리는가" 분석.
    반환: {
        "n_points": int,
        "components": [{"label": str, "variance_pct": float, "top_tokens": list[str]}],
        "score_models": {score_key: {"coefficients": {pc_idx: float}, "r2": float}},
        "recommendations": [str],
        "warning": str | None,
    }
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    # delta가 있는 레코드만 사용
    usable = [
        r for r in records
        if r.get("score_delta") and r.get("diff_text", "").strip()
        and _diff_hash(r["diff_text"]) in embeddings
    ]

    result: dict[str, Any] = {
        "n_points": len(usable),
        "components": [],
        "score_models": {},
        "recommendations": [],
        "warning": None,
    }

    if len(usable) < MIN_DATA_POINTS:
        result["warning"] = (
            f"데이터 포인트 {len(usable)}개 — 최소 {MIN_DATA_POINTS}개 필요. "
            f"방향성 참고만 가능."
        )
        if not usable:
            return result

    # 임베딩 행렬 구성
    X_list = []
    Y_dict: dict[str, list[float]] = {k: [] for k in SCORE_KEYS}

    for r in usable:
        h = _diff_hash(r["diff_text"])
        X_list.append(embeddings[h])
        delta = r["score_delta"]
        for k in SCORE_KEYS:
            Y_dict[k].append(delta.get(k, float("nan")))

    X = np.array(X_list)  # (N, 1536)

    # PCA: 분산 80% 보존 또는 최대 10개 컴포넌트
    n_components = min(len(usable) - 1, 10, X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # (N, n_components)

    # 각 PC의 top 토큰(diff에서 자주 나오는 식별자) 추출
    components_info = []
    for i, (component, var_ratio) in enumerate(
        zip(pca.components_, pca.explained_variance_ratio_)
    ):
        # 해당 PC에서 가장 큰 가중치를 가진 임베딩 차원에 해당하는 diff 토큰 추출
        # 실제로는 top/bottom 레코드에서 자주 나오는 식별자를 역추적
        top_idx = np.argsort(np.abs(component))[-50:]
        # PC 값이 높은 레코드의 diff에서 자주 나오는 식별자 추출
        high_pc_records = [
            usable[j] for j in np.argsort(X_pca[:, i])[-3:]
        ]
        token_counter: dict[str, int] = {}
        for rec in high_pc_records:
            tokens = re.findall(
                r'\b[a-zA-Z_가-힣][a-zA-Z0-9_가-힣]{3,}\b',
                rec["diff_text"]
            )
            for tok in tokens:
                token_counter[tok] = token_counter.get(tok, 0) + 1
        # Python 키워드 / 일반 불용어 제거
        STOPWORDS = {
            "self", "None", "True", "False", "return", "import", "from",
            "class", "def", "elif", "else", "pass", "with", "raise",
            "isinstance", "list", "dict", "str", "int", "float", "bool",
            "constraints", "tuned", "changed", "reader_feedback",
        }
        top_tokens = [
            t for t, _ in sorted(token_counter.items(), key=lambda x: -x[1])
            if t not in STOPWORDS
        ][:6]

        components_info.append({
            "label": f"PC{i+1}",
            "variance_pct": round(var_ratio * 100, 1),
            "top_tokens": top_tokens,
        })

    result["components"] = components_info

    # Lasso 회귀: 각 점수에 대해
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    score_models: dict[str, Any] = {}
    strong_signals: list[tuple[float, str]] = []  # (|coef|, 설명)

    for k, label in zip(SCORE_KEYS, SCORE_LABELS):
        y_raw = Y_dict[k]
        # NaN 제거
        valid_mask = [not np.isnan(v) for v in y_raw]
        if sum(valid_mask) < MIN_DATA_POINTS:
            continue
        y = np.array([v for v, m in zip(y_raw, valid_mask) if m])
        X_valid = X_scaled[[i for i, m in enumerate(valid_mask) if m]]

        alpha = 0.1 if len(y) < 10 else 0.05
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_valid, y)

        coefs = {
            f"PC{i+1}": round(float(c), 3)
            for i, c in enumerate(lasso.coef_)
            if abs(c) > 0.01
        }

        ss_res = np.sum((y - lasso.predict(X_valid)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = round(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 3)

        score_models[k] = {"label": label, "coefficients": coefs, "r2": r2}

        # 강한 신호 수집
        for pc_label, coef in coefs.items():
            if abs(coef) >= 0.2:
                pc_idx = int(pc_label[2:]) - 1
                tokens = (
                    components_info[pc_idx]["top_tokens"]
                    if pc_idx < len(components_info) else []
                )
                direction = "올림 ↑" if coef > 0 else "내림 ↓"
                strong_signals.append((
                    abs(coef),
                    f"{pc_label} ({', '.join(tokens[:3])}) → {label} {direction} (계수 {coef:+.2f})"
                ))

    result["score_models"] = score_models

    # 권고사항 생성 (강도 순 정렬)
    strong_signals.sort(key=lambda x: -x[0])
    recommendations = [sig for _, sig in strong_signals[:8]]
    result["recommendations"] = recommendations

    return result


# ── 4. 보고서 문자열 생성 ─────────────────────────────────────────────────────

def format_factor_report(model_result: dict, chapter_excerpt: str = "") -> str:
    """
    Manager 프롬프트에 삽입할 강한 평가 보고서 문자열 생성.
    chapter_excerpt: 최근 챕터 앞 500자 (소설 내용 참고용).
    """
    n = model_result["n_points"]
    warning = model_result.get("warning", "")
    components = model_result.get("components", [])
    score_models = model_result.get("score_models", {})
    recommendations = model_result.get("recommendations", [])

    lines: list[str] = []
    lines.append(f"## ⚡ Factor Analysis 보고서 (누적 데이터: {n}개 Fixer 사이클)")
    if warning:
        lines.append(f"⚠️ {warning}")
    lines.append("")

    if components:
        lines.append("### PCA 주성분 — 코드 변경의 주요 축")
        for c in components[:5]:
            tokens_str = ", ".join(c["top_tokens"]) if c["top_tokens"] else "(식별자 부족)"
            lines.append(
                f"  {c['label']} (분산 {c['variance_pct']}%): {tokens_str}"
            )
        lines.append("")

    if score_models:
        lines.append("### 점수별 Lasso 회귀 계수")
        lines.append("양수 = 해당 축 방향 변경 시 점수 상승 / 음수 = 하락")
        for k, m in score_models.items():
            label = m["label"]
            coefs = m["coefficients"]
            r2 = m["r2"]
            if not coefs:
                continue
            coef_str = "  ".join(f"{pc}: {v:+.2f}" for pc, v in coefs.items())
            lines.append(f"  [{label}] R²={r2}  |  {coef_str}")
        lines.append("")

    if recommendations:
        lines.append("### 🔴 강한 신호 — Codex 수정 우선순위")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    if chapter_excerpt:
        lines.append("### 최근 챕터 전문 (소설 내용 — Codex가 수정 전 반드시 전체를 읽을 것)")
        lines.append("```")
        lines.append(chapter_excerpt.strip())
        lines.append("```")
        lines.append("")

    lines.append(
        "위 분석을 기존 점수 이력과 함께 종합하여, "
        "어떤 코드 변경이 어떤 점수를 올리는지 근거 있게 판단하고 "
        "Codex 수정 지시사항을 작성하라."
    )
    return "\n".join(lines)


# ── 5. 최근 챕터 추출 ─────────────────────────────────────────────────────────

def get_latest_chapter_excerpt(run_dir: Path) -> str:
    """run_dir에서 가장 최근 챕터 텍스트 전체 반환."""
    chapter_files = sorted(
        list(run_dir.glob("*_chapter*.txt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if chapter_files:
        try:
            return chapter_files[0].read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


# ── 6. 메인 진입점 ────────────────────────────────────────────────────────────

def run_full_analysis(run_dir: Path, openai_client: Any) -> str:
    """
    전체 파이프라인 실행:
    1. output/daily/ 전체 스캔 → 데이터셋 수집
    2. diff 임베딩 (캐시 활용)
    3. PCA + Lasso 분석
    4. 보고서 문자열 반환
    """
    records = collect_dataset()
    embeddings = embed_diffs(records, openai_client)
    model_result = run_factor_model(records, embeddings)
    chapter_excerpt = get_latest_chapter_excerpt(run_dir)
    report = format_factor_report(model_result, chapter_excerpt)
    return report


if __name__ == "__main__":
    # CLI 테스트: python3 tools/analysis/factor_analysis.py
    import os
    sys.path.insert(0, str(REPO_ROOT))
    from src.novel_writer.env_loader import load_project_env
    load_project_env()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # 가장 최근 run_dir 탐색 (output/daily/DATE_KEY/HHMMSS/)
    _daily = OUTPUT_DIR / "daily"
    _run_dirs = sorted(_daily.glob("*/*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    _run_dir = _run_dirs[0] if _run_dirs else _daily
    report = run_full_analysis(_run_dir, client)
    print(report)
