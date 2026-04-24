#!/usr/bin/env bash
# ── spot_semantic_mapping — build & run the Docker container ──────────────────
#
# Usage:
#   ./activate_docker.sh          # build (if needed) then run
#   ./activate_docker.sh --build  # force rebuild even if image exists
#   ./activate_docker.sh --run    # skip build, just run
#
# Before running, make sure you have:
#   1. spot_semantic_mapping/configs/api_keys.ini   (Groq / OpenAI keys)
#   2. spot_semantic_mapping/configs/spot_credentials.json  (Spot user/pass)
#   3. Model weights placed wherever your main_config.yml points to
#   4. nvidia-container-toolkit installed (for --gpus all)
#
# The repo is bind-mounted to /app inside the container, so code edits on the
# host are reflected immediately without rebuilding the image.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE_NAME="ros_semantic_mapping:foxglove"
FORCE_BUILD=false
SKIP_BUILD=false

for arg in "$@"; do
  case $arg in
    --build) FORCE_BUILD=true ;;
    --run)   SKIP_BUILD=true  ;;
  esac
done

# ── Build ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = false ]; then
  if [ "$FORCE_BUILD" = true ] || ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "[build] Building $IMAGE_NAME ..."
    docker build -t "$IMAGE_NAME" .
    echo "[build] Done."
  else
    echo "[build] Image $IMAGE_NAME already exists. Pass --build to rebuild."
  fi
fi

# ── Allow X11 forwarding (GUI tools / Open3D visualizer) ─────────────────────
xhost +local: 2>/dev/null || true

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[run] Starting container..."
docker run --rm -it \
  --gpus all \
  --net=host \
  -e DISPLAY="${DISPLAY:-}" \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/app \
  "$IMAGE_NAME" \
  bash
