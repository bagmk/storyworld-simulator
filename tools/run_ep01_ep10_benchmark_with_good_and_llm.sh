#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PY="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON_BIN_FALLBACK:-python3}"
fi

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
EP_START=1
EP_END=10

REPORT_DIR="reports"
OUT_DIR="output"
DB_PATH="data/simulation.db"

REPORT_COH_GEN="$REPORT_DIR/character_coherence_ep01_ep10_generated_${STAMP}.json"
REPORT_QUAL_GEN="$REPORT_DIR/quality_ep01_ep10_generated_${STAMP}.json"
REPORT_COH_GOOD="$REPORT_DIR/character_coherence_ep01_ep10_good_example_${STAMP}.json"
REPORT_QUAL_GOOD="$REPORT_DIR/quality_ep01_ep10_good_example_${STAMP}.json"
REPORT_LLM="$REPORT_DIR/llm_chapter_review_ep01_ep10_${STAMP}.json"
SUMMARY_CSV="$REPORT_DIR/ep01_ep10_benchmark_summary_${STAMP}.csv"
SUMMARY_MD="$REPORT_DIR/ep01_ep10_benchmark_summary_${STAMP}.md"
NOTEBOOK_OUT="notebooks/ep01_ep10_benchmark_comparison_${STAMP}.ipynb"

mkdir -p "$REPORT_DIR" "$OUT_DIR" notebooks data

echo "============================================================"
echo "[run] ep01-ep10 backup -> regenerate -> benchmark -> notebook"
echo "[run] stamp=$STAMP"
echo "============================================================"

echo "[1/6] Backup current generated artifacts and DB"
bash tools/archive_workspace_state.sh "$STAMP"

echo "[2/6] Regenerate episodes ep01..ep10 and run generated benchmarks"
EP_START="$EP_START" \
EP_END="$EP_END" \
REPORT_COH="$REPORT_COH_GEN" \
REPORT_QUAL="$REPORT_QUAL_GEN" \
DB_PATH="$DB_PATH" \
OUT_DIR="$OUT_DIR" \
RESET_DB_ON_CYCLE="${RESET_DB_ON_CYCLE:-1}" \
RESET_OUTPUT_ON_CYCLE="${RESET_OUTPUT_ON_CYCLE:-1}" \
bash tools/eval_ep01_ep05_cycle.sh

echo "[3/6] Benchmark Good_example quality (ep1..ep10)"
good_files=()
for n in $(seq 1 10); do
  good_files+=("examples/Good_example/ep${n}.md")
done
"$PY" tools/quality_analyzer.py "${good_files[@]}" --json --output "$REPORT_QUAL_GOOD"

echo "[4/6] Benchmark Good_example coherence (ep01..ep10)"
ep_cfgs=()
for n in $(seq -f "%02g" 1 10); do
  match=(config/episodes/ep${n}_*.yaml)
  if [[ ${#match[@]} -eq 0 || ! -e "${match[0]}" ]]; then
    echo "Missing episode config for ep${n}" >&2
    exit 1
  fi
  ep_cfgs+=("${match[0]}")
done

"$PY" tools/character_coherence_benchmark.py \
  --chapters-dir examples/Good_example \
  --episodes "${ep_cfgs[@]}" \
  --db "$DB_PATH" \
  --report-out "$REPORT_COH_GOOD"

echo "[5/6] LLM review benchmark (generated + Good_example)"
"$PY" tools/llm_chapter_review_benchmark.py \
  --episodes "${ep_cfgs[@]}" \
  --mode both \
  --output-dir "$OUT_DIR" \
  --good-examples-dir examples/Good_example \
  --report-out "$REPORT_LLM"

echo "[6/6] Build benchmark summary table + notebook"
"$PY" tools/build_ep_benchmark_notebook.py \
  --quality-generated "$REPORT_QUAL_GEN" \
  --quality-good "$REPORT_QUAL_GOOD" \
  --coherence-generated "$REPORT_COH_GEN" \
  --coherence-good "$REPORT_COH_GOOD" \
  --llm-review "$REPORT_LLM" \
  --summary-csv-out "$SUMMARY_CSV" \
  --summary-md-out "$SUMMARY_MD" \
  --notebook-out "$NOTEBOOK_OUT"

echo "============================================================"
echo "[done]"
echo "  generated quality : $REPORT_QUAL_GEN"
echo "  generated coherence: $REPORT_COH_GEN"
echo "  good quality      : $REPORT_QUAL_GOOD"
echo "  good coherence    : $REPORT_COH_GOOD"
echo "  llm review        : $REPORT_LLM"
echo "  summary csv       : $SUMMARY_CSV"
echo "  summary md        : $SUMMARY_MD"
echo "  notebook          : $NOTEBOOK_OUT"
echo "============================================================"
