#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
DEST_ROOT="backup/run_reset_${STAMP}"

mkdir -p "$DEST_ROOT"

move_if_exists() {
  local src="$1"
  local dest_dir="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$dest_dir"
    mv "$src" "$dest_dir/"
    echo "[backup] moved $src -> $dest_dir/"
  fi
}

# Main generated artifacts
move_if_exists "output" "$DEST_ROOT"
move_if_exists "reports" "$DEST_ROOT"

# Databases and DB sidecars
mkdir -p "$DEST_ROOT/data"
shopt -s nullglob
for f in data/*.db data/*.db-wal data/*.db-shm; do
  mv "$f" "$DEST_ROOT/data/"
  echo "[backup] moved $f -> $DEST_ROOT/data/"
done
shopt -u nullglob

# Recreate working dirs expected by scripts
mkdir -p output reports data

echo "[backup] done: $DEST_ROOT"
