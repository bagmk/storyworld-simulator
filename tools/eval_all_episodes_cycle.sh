#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

max_ep="$(
  ls config/episodes/ep*.yaml \
    | sed -E 's#.*/ep([0-9]+)_.*#\1#' \
    | sort -n \
    | tail -1
)"

if [[ -z "$max_ep" ]]; then
  echo "No episode YAML files found under config/episodes" >&2
  exit 1
fi

EP_START="${EP_START:-1}"
EP_END="${EP_END:-$max_ep}"
export EP_START EP_END

exec bash tools/eval_ep01_ep05_cycle.sh
