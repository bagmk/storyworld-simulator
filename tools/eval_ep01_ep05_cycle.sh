#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

mkdir -p "$OUT_DIR" reports

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

for cfg in "${cfg_files[@]}"; do
  ep="$(basename "$cfg" .yaml)"
  eid="$($PY -c "import yaml; d=yaml.safe_load(open('$cfg', encoding='utf-8')); ep=d.get('episode', d) or {}; print(ep.get('id',''))")"
  if [[ -z "$eid" ]]; then
    echo "Failed to resolve episode id for $cfg" >&2
    exit 1
  fi

  $PY simulate.py \
    --episode "$cfg" \
    --characters config/characters.yaml \
    --world config/world_facts.yaml \
    --storyline config/storyline.yaml \
    --budget 2.0 \
    --db "$DB_PATH" \
    --output "$OUT_DIR"

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

  chapter_files+=("$OUT_DIR/${eid}_chapter.md")
done

$PY tools/character_coherence_benchmark.py \
  --chapters-dir "$OUT_DIR" \
  --episodes "${cfg_files[@]}" \
  --report-out "$REPORT_COH"

$PY tools/quality_analyzer.py "${chapter_files[@]}" --json --output "$REPORT_QUAL"

echo "Cycle complete"
echo "  coherence: $REPORT_COH"
echo "  quality:   $REPORT_QUAL"
