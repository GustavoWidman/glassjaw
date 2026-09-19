#!/usr/bin/env bash
# dev wrapper: nixos hosts need store libs on LD_LIBRARY_PATH for the venv's
# manylinux wheels (libstdc++, libz). harmless elsewhere.
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:}/nix/store/6vzcxjxa2wlh3p9f5nhbk62bl3q313ri-gcc-14.3.0-lib/lib:/nix/store/lf793zr9yfa0dpph8jlxbbdnvnahvq8b-zlib-1.3.2/lib"
exec .venv/bin/python "$@"
