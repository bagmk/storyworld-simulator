#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Auto-load project .env for OPENAI_API_KEY if present
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

PY="${PYTHON_BIN:-./.venv/bin/python}"
OUT_DIR="${OUT_DIR:-output}"
DB_PATH="${DB_PATH:-data/simulation.db}"
EP_START="${EP_START:-1}"
EP_END="${EP_END:-5}"
REPORT_COH="${REPORT_COH:-reports/character_coherence_ep01_ep$(printf '%02d' "$EP_END").json}"
REPORT_QUAL="${REPORT_QUAL:-reports/quality_ep01_ep$(printf '%02d' "$EP_END").json}"
RESET_DB_ON_CYCLE="${RESET_DB_ON_CYCLE:-1}"
RESET_OUTPUT_ON_CYCLE="${RESET_OUTPUT_ON_CYCLE:-0}"
TRACK_RUN_ID="${NOVEL_RUN_ID:-${TRACK_RUN_ID:-}}"
TRACK_ITERATION="${NOVEL_ITERATION:-${TRACK_ITERATION:-}}"
TRACK_PHASE="${NOVEL_PHASE:-${TRACK_PHASE:-rl_eval}}"

mkdir -p "$OUT_DIR" reports

cycle_started_epoch="$(date +%s)"
echo "================================================================================"
echo "[cycle] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[cycle] episodes: ep$(printf '%02d' "$EP_START")..ep$(printf '%02d' "$EP_END")"
echo "[cycle] db=$DB_PATH out=$OUT_DIR"
echo "[cycle] report_coh=$REPORT_COH"
echo "[cycle] report_qual=$REPORT_QUAL"
echo "[cycle] tracking run_id=${TRACK_RUN_ID:-none} iter=${TRACK_ITERATION:-none} phase=${TRACK_PHASE:-none}"
echo "[cycle] reset_db=$RESET_DB_ON_CYCLE reset_output=$RESET_OUTPUT_ON_CYCLE"
echo "================================================================================"

if [[ "$RESET_DB_ON_CYCLE" == "1" ]]; then
  rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
  echo "[cycle] reset db: $DB_PATH"
fi
if [[ "$RESET_OUTPUT_ON_CYCLE" == "1" ]]; then
  rm -f "$OUT_DIR"/ep*_simulation.json "$OUT_DIR"/ep*_debug.log "$OUT_DIR"/ep*_scenes.json "$OUT_DIR"/ep*_chapter.md 2>/dev/null || true
  echo "[cycle] reset generated outputs in $OUT_DIR"
fi

cfg_files=()
for n in $(seq -f "%02g" "$EP_START" "$EP_END"); do
  for cfg in config/episodes/ep${n}_*.yaml; do
    [[ -e "$cfg" ]] || continue
    cfg_files+=("$cfg")
  done
done

if [[ ${#cfg_files[@]} -eq 0 ]]; then
  echo "No episode config files found for range $EP_START..$EP_END" >&2
  exit 1
fi

IFS=$'\n' cfg_files=($(printf "%s\n" "${cfg_files[@]}" | sort))
unset IFS

chapter_files=()
total_cfgs=${#cfg_files[@]}
idx=0

for cfg in "${cfg_files[@]}"; do
  idx=$((idx + 1))
  ep="$(basename "$cfg" .yaml)"
  eid="$($PY -c "import yaml; d=yaml.safe_load(open('$cfg', encoding='utf-8')); ep=d.get('episode', d) or {}; print(ep.get('id',''))")"
  if [[ -z "$eid" ]]; then
    echo "Failed to resolve episode id for $cfg" >&2
    exit 1
  fi

  echo "[cycle][$idx/$total_cfgs] $eid | run_id=${TRACK_RUN_ID:-none} iter=${TRACK_ITERATION:-none} phase=${TRACK_PHASE}"
  ep_started_epoch="$(date +%s)"
  export NOVEL_RUN_ID="$TRACK_RUN_ID"
  export NOVEL_ITERATION="$TRACK_ITERATION"
  export NOVEL_PHASE="$TRACK_PHASE"
  export NOVEL_EPISODE_INDEX="$idx"
  export NOVEL_EPISODE_ID="$eid"

  echo "[cycle][$idx/$total_cfgs] simulate:start $eid @ $(date '+%H:%M:%S')"
  $PY simulate.py \
    --episode "$cfg" \
    --characters config/characters.yaml \
    --world config/world_facts.yaml \
    --storyline config/storyline.yaml \
    --budget 2.0 \
    --db "$DB_PATH" \
    --output "$OUT_DIR"
  echo "[cycle][$idx/$total_cfgs] simulate:done  $eid elapsed=$(( $(date +%s) - ep_started_epoch ))s"

  ch_started_epoch="$(date +%s)"
  echo "[cycle][$idx/$total_cfgs] chapter:start  $eid @ $(date '+%H:%M:%S')"
  $PY generate_chapter.py \
    --episode "$eid" \
    --episode-config "$cfg" \
    --protagonist kim_sumin \
    --protagonist-name "Kim Sumin" \
    --style third_person_close \
    --words 1800 \
    --budget 3.0 \
    --db "$DB_PATH" \
    --output "$OUT_DIR"
  echo "[cycle][$idx/$total_cfgs] chapter:done   $eid elapsed=$(( $(date +%s) - ch_started_epoch ))s total_ep=$(( $(date +%s) - ep_started_epoch ))s"

  chapter_files+=("$OUT_DIR/${eid}_chapter.md")
done

echo "[cycle] benchmark:start coherence @ $(date '+%H:%M:%S')"
$PY tools/character_coherence_benchmark.py \
  --chapters-dir "$OUT_DIR" \
  --episodes "${cfg_files[@]}" \
  --report-out "$REPORT_COH"
echo "[cycle] benchmark:done coherence"

echo "[cycle] benchmark:start quality @ $(date '+%H:%M:%S')"
$PY tools/quality_analyzer.py "${chapter_files[@]}" --json --output "$REPORT_QUAL"
echo "[cycle] benchmark:done quality"

echo "Cycle complete"
echo "  coherence: $REPORT_COH"
echo "  quality:   $REPORT_QUAL"
echo "  elapsed:   $(( $(date +%s) - cycle_started_epoch ))s"
