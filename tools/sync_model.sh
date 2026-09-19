#!/usr/bin/env bash
# copies the trained artifacts into the firmware tree and refreshes the
# decision threshold. run after tools/train.py + tools/export_model.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p firmware/main/model
cp models/detector_int8.onnx firmware/main/model/detector_int8.onnx
python3 - <<'EOF'
import json, pathlib
h = json.load(open("models/history.json"))
cfg = pathlib.Path("firmware/main/config.hpp")
cfg.write_text(
    "// generated from models/history.json by tools/sync_model.sh - do not edit by hand.\n"
    "#pragma once\n"
    f"#define GLASSJAW_THRESHOLD {h['threshold']:.4f}f\n"
)
print("threshold:", h["threshold"])
EOF
echo "firmware model synced."
