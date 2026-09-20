#!/usr/bin/env bash
# builds and flashes the firmware on desktop-nixos (docker + esp-idf image),
# with the esp32 on /dev/ttyUSB0. usage from the repo root:
#   tools/flash.sh            # rsync + build + flash
#   tools/flash.sh build      # build only
#   tools/flash.sh monitor    # serial monitor only
set -euo pipefail
HOST=${GLASSJAW_HOST:-desktop-nixos}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:-all}

if [ "${1:-}" = "clean" ]; then
  echo ">> wiping remote build dir for a clean rebuild"
  ssh -o BatchMode=yes "$HOST" "bash -c 'rm -rf ~/glassjaw'"
  shift
fi

echo ">> syncing repo to $HOST:~/glassjaw"
rsync -a --delete \
  --exclude .git --exclude build --exclude .venv --exclude data \
  --exclude third_party/FreeRTOS-Kernel/examples \
  "$REPO/" "$HOST:glassjaw/"

ssh -o BatchMode=yes "$HOST" bash -s "$MODE" <<'REMOTE'
set -euo pipefail
MODE=$1
cd ~/glassjaw
[ -e /dev/ttyUSB0 ] || { echo "no /dev/ttyUSB0 on $(hostname)"; exit 1; }
docker image inspect espressif/idf:v5.5 >/dev/null 2>&1 || docker pull espressif/idf:v5.5
docker run --rm --device /dev/ttyUSB0 --group-add "$(stat -c %g /dev/ttyUSB0)" \
  -v "$PWD":/work -w /work/firmware \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e IDF_CCACHE_ENABLE=0 \
  espressif/idf:v5.5 \
  bash -c "case $1 in
    build)   idf.py build ;;
    monitor) timeout 60 idf.py monitor || true ;;
    *)       idf.py build flash ;;
  esac" -- "$MODE"
REMOTE
