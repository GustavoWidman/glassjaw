#!/usr/bin/env bash
# live serial console for the esp32 on desktop-nixos (ctrl-c to exit).
# uses the idf docker image for pyserial, with dtr/rts held off so opening
# the port does not reset the board. usage: tools/serial.sh [seconds]
set -euo pipefail
HOST=${GLASSJAW_HOST:-desktop-nixos}
SECS=${1:-0}

cat > /tmp/gj_serial.py <<'EOF'
import serial, sys, time
ser = serial.Serial(None, 115200, timeout=1)
ser.port = "/dev/ttyUSB0"
ser.dtr = False
ser.rts = False
ser.open()
end = time.time() + int(sys.argv[1]) if sys.argv[1] != "0" else float("inf")
try:
    while time.time() < end:
        data = ser.read(4096)
        if data:
            sys.stdout.write(data.decode(errors="replace"))
            sys.stdout.flush()
except KeyboardInterrupt:
    pass
EOF

scp -q /tmp/gj_serial.py "$HOST":/tmp/
ssh -t -o BatchMode=yes "$HOST" "docker run --rm -i --entrypoint /opt/esp/python_env/idf5.5_py3.12_env/bin/python --device /dev/ttyUSB0 --group-add \$(stat -c %g /dev/ttyUSB0) -v /tmp/gj_serial.py:/s.py espressif/idf:v5.5 /s.py $SECS"
