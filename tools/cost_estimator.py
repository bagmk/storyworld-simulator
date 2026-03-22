"""tools/cost_estimator.py — 과거 실행 기록 기반 사이클당 비용 추정.

cycle_score_log.jsonl에 기록된 실제 비용(cost_usd)을 읽어서
다음 플랜의 예상 비용을 계산한다.

코드가 바뀌어도 다음 실행 후 자동으로 새 비용이 반영된다.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
CYCLE_SCORE_LOG = _REPO_ROOT / "data" / "cycle_score_log.jsonl"


class CostEstimate(NamedTuple):
    avg: float            # 평균 사이클 비용 (USD)
    median: float         # 중앙값
    p90: float            # 90th percentile (worst-case 근사)
    n_samples: int        # 샘플 수
    breakdown_avg: dict   # {step: avg_usd}
    note: str             # 추정 근거 설명


_FALLBACK_BREAKDOWN = {
    # 실측 데이터 없을 때 사용하는 보수적 추정
    # 2026-03-21 ep01_conference_shadow 1회 실행 실측값 기준
    "guardian":   0.003,
    "simulation": 0.013,
    "chapter":    0.000,
    "auto_chapter": 0.000,
    "manager":    0.002,
    "auto_review": 0.024,
    "code_review": 0.000,
    "regen_check": 0.000,
    "final_review": 0.000,
    "feedback_parse": 0.000,
}
_FALLBACK_TOTAL = sum(_FALLBACK_BREAKDOWN.values())  # ~$0.042


def estimate_cost_per_cycle(
    episode_id: str | None = None,
    log_path: Path | None = None,
    min_samples: int = 1,
) -> CostEstimate:
    """cycle_score_log.jsonl에서 실제 비용을 읽어 추정값 반환.

    episode_id가 지정되면 해당 에피소드 데이터만 사용,
    샘플이 min_samples 미만이면 전체 데이터로 확장.
    데이터가 전혀 없으면 하드코딩된 fallback 반환.
    """
    log_path = log_path or CYCLE_SCORE_LOG
    rows = _load_rows(log_path)

    # 비용 필드가 있는 행만 필터
    costed_rows = [r for r in rows if r.get("cost_usd", 0) > 0]

    # episode 필터
    ep_rows = [r for r in costed_rows if r.get("episode_id") == episode_id] if episode_id else []
    if len(ep_rows) < min_samples:
        ep_rows = costed_rows  # 전체로 확장

    if not ep_rows:
        return _fallback_estimate()

    costs = [r["cost_usd"] for r in ep_rows]
    avg = statistics.mean(costs)
    median = statistics.median(costs)
    p90 = sorted(costs)[int(len(costs) * 0.9)] if len(costs) >= 3 else max(costs)

    # 단계별 평균
    breakdown_avg: dict[str, float] = {}
    all_keys = set()
    for r in ep_rows:
        all_keys.update(r.get("cost_breakdown", {}).keys())
    for k in all_keys:
        vals = [r.get("cost_breakdown", {}).get(k, 0.0) for r in ep_rows]
        m = statistics.mean(vals)
        if m > 0.00001:
            breakdown_avg[k] = round(m, 5)

    src = f"ep={episode_id}" if episode_id and len(ep_rows) < len(costed_rows) else "전체"
    note = f"실측 {len(ep_rows)}회 기반 ({src})"

    return CostEstimate(
        avg=round(avg, 5),
        median=round(median, 5),
        p90=round(p90, 5),
        n_samples=len(ep_rows),
        breakdown_avg=breakdown_avg,
        note=note,
    )


def format_cost_estimate_for_plan(
    episode_id: str | None,
    outer_cycles: int,
    log_path: Path | None = None,
) -> str:
    """플랜 미리보기용 비용 예측 텍스트 반환."""
    est = estimate_cost_per_cycle(episode_id=episode_id, log_path=log_path)

    total_low  = est.avg    * outer_cycles
    total_high = est.p90    * outer_cycles

    if est.n_samples == 0:
        return (
            f"$~{est.avg:.3f}/cycle (fallback 추정, 실측 데이터 없음)\n"
            f"  {outer_cycles}회 → 약 **${total_low:.2f}** ~ **${total_high:.2f}**"
        )

    lines = [
        f"$~{est.avg:.3f}/cycle ({est.note})",
        f"  {outer_cycles}회 → 약 **${total_low:.2f}** ~ **${total_high:.2f}** "
        f"(avg ~ p90)",
    ]
    if est.breakdown_avg:
        top = sorted(est.breakdown_avg.items(), key=lambda x: -x[1])[:4]
        top_str = " | ".join(f"{k} ${v:.4f}" for k, v in top)
        lines.append(f"  주요 단계: {top_str}")
    return "\n".join(lines)


def _load_rows(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _fallback_estimate() -> CostEstimate:
    breakdown_avg = {k: v for k, v in _FALLBACK_BREAKDOWN.items() if v > 0}
    return CostEstimate(
        avg=_FALLBACK_TOTAL,
        median=_FALLBACK_TOTAL,
        p90=_FALLBACK_TOTAL * 1.5,
        n_samples=0,
        breakdown_avg=breakdown_avg,
        note="실측 데이터 없음 — 2026-03-21 1회 측정 기준 fallback",
    )


if __name__ == "__main__":
    import sys
    ep = sys.argv[1] if len(sys.argv) > 1 else None
    n  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    est = estimate_cost_per_cycle(episode_id=ep)
    print(f"샘플: {est.n_samples}회 | avg ${est.avg:.4f} | median ${est.median:.4f} | p90 ${est.p90:.4f}")
    print(f"→ {n}회 예상: ${est.avg*n:.2f} ~ ${est.p90*n:.2f}")
    print(f"단계별: {est.breakdown_avg}")
    print(f"근거: {est.note}")
