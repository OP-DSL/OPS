#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Repository root: $REPO_ROOT"

# Find Makefiles under top-level apps/ directory
mapfile -t makefiles < <(find "$REPO_ROOT/apps" -maxdepth 6 -type f -name Makefile 2>/dev/null || true)

if [ ${#makefiles[@]} -eq 0 ]; then
  echo "No Makefiles found under $REPO_ROOT/apps"
  exit 0
fi

for mk in "${makefiles[@]}"; do
  dir=$(dirname "$mk")
  echo "---- Checking: $dir"
  # Try a dry-run to see if 'cleanall' is a valid target in this directory
  if (cd "$dir" && make -n cleanall >/dev/null 2>&1); then
    echo "---- Running 'make cleanall' in: $dir"
    (cd "$dir" && make cleanall)
  else
    echo "---- Skipping (no cleanall target or not applicable): $dir"
  fi
done

echo "Done."
