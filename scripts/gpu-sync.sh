#!/usr/bin/env bash
# Sync training code in, and GPU-produced datasets/artifacts out.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
remote="${MW_GPU_HOST:-hmrdkn@workstation.tail4b94a8.ts.net}"
remote_root="${MW_GPU_ROOT:-~/mini-world}"
dry=()
if [[ "${2:-}" == "--dry-run" ]]; then
  dry=(--dry-run)
elif [[ "${2:-}" != "" ]]; then
  echo "Usage: $0 push|pull [--dry-run]" >&2
  exit 2
fi

case "${1:-}" in
  push)
    rsync -azv "${dry[@]}" --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '*.pyc' "$root/training/" "$remote:$remote_root/training/"
    ;;
  pull)
    rsync -azv "${dry[@]}" "$remote:$remote_root/training/artifacts/" "$root/training/artifacts/"
    ;;
  *)
    echo "Usage: $0 push|pull [--dry-run]" >&2
    exit 2
    ;;
esac
