#!/usr/bin/env python3
"""generates docs/diagrams/*.svg from a tiny layout helper so text always
fits its box and arrows land on real edges. run: tools/run.sh tools/diagrams.py"""
from pathlib import Path

MONO = 6.4  # px per char at 11px monospace (conservative)

class Box:
    def __init__(self, x, y, w, h, fill, title, lines=(), title_fill="#fff", sub_fill="#9fc6da", r=8):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.fill, self.title, self.lines = fill, title, list(lines)
        self.title_fill, self.sub_fill, self.r = title_fill, sub_fill, r

    def svg(self):
        out = [f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" rx="{self.r}" fill="{self.fill}"/>']
        cx = self.x + self.w / 2
        y = self.y + 22
        out.append(f'<text x="{cx:.0f}" y="{y:.0f}" text-anchor="middle" fill="{self.title_fill}" font-size="13" font-weight="bold">{self.title}</text>')
        for ln in self.lines:
            y += 17
            out.append(f'<text x="{cx:.0f}" y="{y:.0f}" text-anchor="middle" fill="{self.sub_fill}" font-size="11">{ln}</text>')
        return "\n".join(out)

    def check(self):
        longest = max([len(self.title) * 7.6] + [len(l) * MONO for l in self.lines])
        assert longest <= self.w - 12, f"overflow in {self.title!r}: {longest:.0f} > {self.w-12}"

    def right(self): return (self.x + self.w, self.y + self.h / 2)
    def left(self): return (self.x, self.y + self.h / 2)
    def top(self): return (self.x + self.w / 2, self.y)
    def bottom(self): return (self.x + self.w / 2, self.y + self.h)


def arrow(a, b, color="#4a7fa5", dash=False, label=None, lpos=0.5, dy=-6):
    d = "stroke-dasharray='5 4' " if dash else ""
    out = [f'<path d="M {a[0]:.0f} {a[1]:.0f} L {b[0]:.0f} {b[1]:.0f}" stroke="{color}" stroke-width="2" fill="none" {d}marker-end="url(#{ "g" if dash else "a"})"/>']
    if label:
        mx, my = a[0] + (b[0] - a[0]) * lpos, a[1] + (b[1] - a[1]) * lpos
        out.append(f'<text x="{mx:.0f}" y="{my + dy:.0f}" text-anchor="middle" fill="#555" font-size="10">{label}</text>')
    return "\n".join(out)


def header(title, sub, w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="ui-monospace,Menlo,monospace">\n'
            '<defs>'
            '<marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#4a7fa5"/></marker>'
            '<marker id="g" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#c9a227"/></marker>'
            '</defs>\n'
            f'<text x="24" y="32" font-size="19" fill="#111" font-weight="bold">{title}</text>\n'
            f'<text x="24" y="50" font-size="12" fill="#555">{sub}</text>\n')


# ---------------- rtos architecture ----------------
def rtos_svg():
    b = {}
    b["mic"] = Box(30, 84, 190, 62, "#3a2a1b", "INMP441", ["i2s 16 kHz mono", "dma 8 kb blocks"], "#f0d9b5", "#cbb394")
    b["capture"] = Box(300, 76, 260, 96, "#1b3a4b", "capture  (prio 5)",
                       ["i2s_channel_read, blocking", "writes spsc ring 32 k", "gives window sem / 8 k smp"])
    b["ring"] = Box(300, 216, 260, 62, "#204030", "spsc ring 32 k", ["lock-free, drop-oldest"], "#d7efe0", "#a9d3bb")
    b["sem"] = Box(30, 216, 190, 62, "#204030", "window semaphore", ["binary, 1 give / hop"], "#d7efe0", "#a9d3bb")
    b["features"] = Box(300, 326, 260, 96, "#1b3a4b", "features  (prio 4)",
                        ["peek newest 1 s window", "hamming, fft, mel, log10", "26 x 61 spectrogram"])
    b["queue"] = Box(650, 336, 220, 76, "#204030", "spec queue (3)", ["spectrogram + peak + t0", "full -> drop oldest"], "#d7efe0", "#a9d3bb")
    b["detect"] = Box(650, 466, 260, 150, "#1b3a4b", "detect  (prio 2)",
                      ["energy gate -40 dbfs", "onnx engine (int8 cnn)", "debounce 2/3", "led gpio25, buzzer opt.", "e2e latency sample"])
    b["monitor"] = Box(300, 470, 260, 80, "#3d2a4a", "monitor  (prio 1)", ["p50/p99 every 5 s"], "#e6d5f0", "#c5a9d6")
    b["stats"] = Box(30, 326, 190, 62, "#204030", "stats mutex", ["latency stats x4"], "#d7efe0", "#a9d3bb")

    parts = [header("glassjaw - freertos task graph", "same graph runs on the esp32 firmware and the host simulator (freertos posix port)", 960, 660)]
    for x in b.values():
        x.check()
        parts.append(x.svg())
    parts.append(arrow(b["mic"].right(), b["capture"].left(), label="i2s dma"))
    parts.append(arrow(b["capture"].bottom(), b["ring"].top()))
    parts.append(arrow(b["ring"].left(), b["sem"].right(), color="#c9a227", dash=True, label="give / 8k"))
    parts.append(arrow(b["sem"].bottom(), b["features"].left(), color="#c9a227", dash=True, label="take"))
    parts.append(arrow(b["features"].right(), b["queue"].left(), label="send"))
    parts.append(arrow(b["queue"].bottom(), b["detect"].top(), label="recv"))
    parts.append(arrow(b["features"].bottom(), b["monitor"].top(), color="#c9a227", dash=True, label="record"))
    parts.append(arrow(b["detect"].left(), b["monitor"].right(), color="#c9a227", dash=True))
    parts.append(arrow(b["monitor"].left(), b["stats"].bottom(), color="#c9a227", dash=True))
    parts.append('<text x="24" y="648" font-size="11" fill="#666">hop = 500 ms of audio. one window = 1 s, 50% overlap. e2e budget per hop; monitor reports per-stage p50/p99.</text>')
    return "\n".join(parts) + "\n</svg>\n"


# ---------------- wiring ----------------
def wiring_svg():
    W = 980
    parts = [header("glassjaw - kit wiring", "esp32 devkit v1 + inmp441 + led. buzzer optional (kconfig). follow pin names, not positions.", W, 620)]

    # esp32 board
    bx, by, bw, bh = 400, 80, 180, 460
    parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="10" fill="#1b3a4b"/>')
    parts.append(f'<rect x="{bx+48}" y="{by+12}" width="84" height="64" rx="6" fill="#0b1c26"/>')
    parts.append(f'<text x="{bx+90}" y="{by+48}" text-anchor="middle" fill="#8fd3f4" font-size="11">ESP-WROOM</text>')
    parts.append(f'<rect x="{bx+56}" y="{by+bh-26}" width="68" height="16" rx="3" fill="#0b1c26"/>')
    parts.append(f'<text x="{bx+90}" y="{by+bh-15}" text-anchor="middle" fill="#888" font-size="9">USB</text>')

    L = ["EN","VP","VN","D34","D35","D32","D33","D25","D26","D27","D14","D12","D13","GND","VIN"]
    R = ["3V3","GND","D15","D2","D4","RX2","TX2","D5","D18","D19","D21","RX0","TX0","D22","D23"]
    ys = [by + 24 + i * 28 for i in range(15)]
    for label, y in zip(L, ys):
        parts.append(f'<circle cx="{bx+8}" cy="{y}" r="5" fill="#d9b44a"/>')
        parts.append(f'<text x="{bx-4}" y="{y+4}" text-anchor="end" fill="#cde3ee" font-size="11">{label}</text>')
    for label, y in zip(R, ys):
        parts.append(f'<circle cx="{bx+bw-8}" cy="{y}" r="5" fill="#d9b44a"/>')
        parts.append(f'<text x="{bx+bw+4}" y="{y+4}" text-anchor="start" fill="#cde3ee" font-size="11">{label}</text>')

    # inmp441 module
    parts.append(f'<rect x="60" y="84" width="170" height="190" rx="10" fill="#204030"/>')
    parts.append('<circle cx="145" cy="128" r="24" fill="#101f16" stroke="#2e5842"/>')
    parts.append('<text x="145" y="132" text-anchor="middle" fill="#9fd3b4" font-size="10">MIC</text>')
    parts.append('<text x="145" y="172" text-anchor="middle" fill="#d7efe0" font-size="13" font-weight="bold">INMP441</text>')
    pins = [("VDD", 196), ("GND", 216), ("SD", 236), ("SCK", 256), ("WS", 276)]
    for name, y in pins:
        parts.append(f'<circle cx="224" cy="{y}" r="5" fill="#d9b44a"/>')
        parts.append(f'<text x="218" y="{y+4}" text-anchor="end" fill="#cde3ee" font-size="11">{name}</text>')
    parts.append('<circle cx="66" cy="256" r="5" fill="#d9b44a"/>')
    parts.append('<text x="72" y="260" text-anchor="start" fill="#cde3ee" font-size="11">L/R</text>')

    # led module
    parts.append('<rect x="740" y="230" width="150" height="86" rx="10" fill="#204030"/>')
    parts.append('<circle cx="815" cy="264" r="13" fill="#e05d5d" stroke="#7c2f2f"/>')
    parts.append('<text x="815" y="300" text-anchor="middle" fill="#d7efe0" font-size="13" font-weight="bold">ALARM LED</text>')
    parts.append('<circle cx="748" cy="240" r="5" fill="#d9b44a"/>')
    parts.append('<text x="740" y="236" text-anchor="start" fill="#cde3ee" font-size="11">A</text>')
    parts.append('<circle cx="882" cy="240" r="5" fill="#d9b44a"/>')
    parts.append('<text x="890" y="236" text-anchor="start" fill="#cde3ee" font-size="11">K</text>')

    def wire(d, color):
        parts.append(f'<path d="{d}" stroke="{color}" stroke-width="3.5" fill="none" stroke-linecap="round"/>')

    # VDD -> 3V3: out right, up over the board, down into 3V3 (top-right pin)
    wire("M 229 196 H 300 V 64 H 700 V 104 H 584", "#d33")
    # GND -> GND right pin
    wire("M 229 216 H 315 V 78 H 686 V 132 H 584", "#333")
    # L/R -> GND left rail
    wire("M 61 256 H 30 V 476 H 400", "#333")
    # SD -> D32
    wire("M 229 236 H 290 V 236 H 400", "#2b7fd9")
    # SCK -> D14
    wire("M 229 256 H 272 V 368 H 400", "#e8952b")
    # WS -> D15: down and around the bottom
    wire("M 229 276 H 254 V 580 H 690 V 160 H 584", "#c9c93a")
    # LED A -> D25 (left header, y=300), resistor drawn on the segment
    wire("M 743 240 H 700 V 300 H 392", "#e05d5d")
    parts.append('<rect x="620" y="294" width="36" height="11" fill="#c8b78e" stroke="#8f8060"/>')
    parts.append('<text x="612" y="288" fill="#a33" font-size="11">330R</text>')
    # LED K -> GND left rail (bottom)
    wire("M 887 240 H 930 V 566 H 46 V 476 H 400", "#333")

    legend = [("red", "#d33", "3v3"), ("blk", "#333", "gnd"), ("blu", "#2b7fd9", "sd"),
              ("org", "#e8952b", "sck"), ("yel", "#c9c93a", "ws"), ("red", "#e05d5d", "led")]
    x = 24
    parts.append('<text x="24" y="604" font-size="12" fill="#333">wires:</text>')
    x = 74
    for name, col, lbl in legend:
        parts.append(f'<text x="{x}" y="604" font-size="12" fill="{col}">{lbl}</text>')
        x += len(lbl) * 8 + 24
    parts.append('<text x="24" y="620" font-size="11" fill="#666">onboard led gpio2 blinks 1 hz (5 hz on alarm). buzzer: optional, gpio26, enable in menuconfig.</text>')
    return "\n".join(parts) + "\n</svg>\n"


out = Path(__file__).parent.parent / "docs" / "diagrams"
out.mkdir(parents=True, exist_ok=True)
(out / "rtos-architecture.svg").write_text(rtos_svg())
(out / "wiring.svg").write_text(wiring_svg())
print("diagrams written to", out)
