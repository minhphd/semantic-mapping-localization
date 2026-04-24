#!/usr/bin/env bash
# Serve the MkDocs documentation site locally.
#
# Usage:
#   ./serve_docs.sh            # default: http://localhost:8000
#   ./serve_docs.sh --port 9000
#   ./serve_docs.sh --build    # build static site to ./site/ instead of serving

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000
BUILD_ONLY=false

# ── Parse arguments ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)   PORT="$2"; shift 2 ;;
    --build)  BUILD_ONLY=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--port PORT] [--build]"
      echo "  --port PORT   Port to serve on (default: 8000)"
      echo "  --build       Build static site to ./site/ and exit"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Check for mkdocs ───────────────────────────────────────────
if ! command -v mkdocs &>/dev/null; then
  echo "mkdocs not found. Installing docs dependencies..."
  pip install -q -r "$ROOT/docs-requirements.txt"
fi

cd "$ROOT"

if $BUILD_ONLY; then
  echo "Building static site → $ROOT/site/"
  mkdocs build --clean
  echo "Done. Open $ROOT/site/index.html to preview offline."
else
  echo "Starting docs server at http://localhost:$PORT"
  echo "Live reload active — browser refreshes automatically on save."
  echo "Press Ctrl+C to stop."
  mkdocs serve \
    --dev-addr "localhost:$PORT" \
    --livereload \
    --watch "$ROOT/spot_semantic_mapping" \
    --watch "$ROOT/mkdocs.yml"
fi
