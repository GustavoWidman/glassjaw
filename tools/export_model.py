"""export the trained cnn to onnx (float + static int8) with calibration.

graph (input 1x1x26x61 log-mel spectrogram):
  x0   = Clip((spec - mu) / sd, -5, 5)
  h1   = Relu(Conv(x0, c1))            16ch, 3x3, s1  -> 24x59
  p1   = MaxPool(h1, 2x2)                             -> 12x29
  h2   = Relu(Conv(p1, c2))            32ch            -> 10x27
  p2   = MaxPool(h2, 2x2)                             -> 5x13
  h3   = Relu(Conv(p2, c3))            32ch            -> 3x11
  gap  = GlobalAveragePool(h3)                         -> 32
  logit= MatMul(gap, fc) + fcb                         -> 1
  out  = Sigmoid(logit)

weights come from train.py's models/weights.npz. the float model is the
reference; the int8 model (static, per-channel conv weights) is what runs on
the esp32. both are validated against onnxruntime before leaving this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

sys.path.insert(0, str(Path(__file__).parent))
from features import FRAME_LEN, N_MELS  # noqa: E402


def build_graph(w: dict) -> onnx.ModelProto:
    def arr(name, a):
        a = np.ascontiguousarray(a)
        if a.dtype not in (np.float32, np.int64):
            a = a.astype(np.float32)
        return numpy_helper.from_array(a, name)

    def conv_w(name, flat, cout, cin, k):
        # numpy side stores (k*k*cin, cout) with im2col patch ordering
        # (i, j, c); onnx wants [cout, cin, kH, kW]. transpose to
        # (cout, patch), reshape to (cout, kH, kW, cin), then reorder axes.
        w_ = np.ascontiguousarray(flat.T.reshape(cout, k, k, cin).transpose(0, 3, 1, 2))
        return arr(name, w_)

    inits = [
        arr("mu", w["mu"].reshape(1, 1, 26, 1)),
        arr("sd", w["sd"].reshape(1, 1, 26, 1)),
        arr("clip_lo", np.float32(-5.0)), arr("clip_hi", np.float32(5.0)),
        arr("gap_shape", np.array([-1, 32], dtype=np.int64)),
        conv_w("c1w", w["c1w"], 16, 1, 3), arr("c1b", w["c1b"]),
        conv_w("c2w", w["c2w"], 32, 16, 3), arr("c2b", w["c2b"]),
        conv_w("c3w", w["c3w"], 32, 32, 3), arr("c3b", w["c3b"]),
        arr("fcw", w["fcw"]), arr("fcb", w["fcb"]),
    ]
    nodes = [
        helper.make_node("Sub", ["spec", "mu"], ["centered"]),
        helper.make_node("Div", ["centered", "sd"], ["normed"]),
        helper.make_node("Max", ["normed", "clip_lo"], ["cl_lo"]),
        helper.make_node("Min", ["cl_lo", "clip_hi"], ["x0"]),
        helper.make_node("Conv", ["x0", "c1w", "c1b"], ["z1"], kernel_shape=[3, 3], strides=[1, 1]),
        helper.make_node("Relu", ["z1"], ["h1"]),
        helper.make_node("MaxPool", ["h1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p1", "c2w", "c2b"], ["z2"], kernel_shape=[3, 3], strides=[1, 1]),
        helper.make_node("Relu", ["z2"], ["h2"]),
        helper.make_node("MaxPool", ["h2"], ["p2"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p2", "c3w", "c3b"], ["z3"], kernel_shape=[3, 3], strides=[1, 1]),
        helper.make_node("Relu", ["z3"], ["h3"]),
        helper.make_node("GlobalAveragePool", ["h3"], ["gap4d"]),
        helper.make_node("Reshape", ["gap4d", "gap_shape"], ["gap"]),
        helper.make_node("MatMul", ["gap", "fcw"], ["mm"]),
        helper.make_node("Add", ["mm", "fcb"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["score"]),
    ]
    graph = helper.make_graph(
        nodes, "glassjaw-cnn",
        [helper.make_tensor_value_info("spec", TensorProto.FLOAT, [1, 1, N_MELS, FRAME_LEN])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])],
        inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def run(path, x):
    # optimizations OFF: the default optimizer mis-fuses q/dq pairs on this
    # graph (verified: opt=all turns a 0.87 into 0.12). the literal graph is
    # the reference semantics.
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return np.stack([sess.run(["score"], {"spec": s.reshape(1, 1, N_MELS, FRAME_LEN).astype(np.float32)})[0][0, 0] for s in x])


class Reader(CalibrationDataReader):
    def __init__(self, specs):
        self.specs = specs
        self.i = 0

    def get_next(self):
        if self.i >= len(self.specs):
            return None
        s = {"spec": self.specs[self.i].reshape(1, 1, N_MELS, FRAME_LEN).astype(np.float32)}
        self.i += 1
        return s


def fuse_qdq_to_qlinear(src: str, dst: str) -> int:
    """qdq -> qlinear rewrite so the esp32 executes raw int8 (the qdq form
    dequantizes every weight/activation to float: ~150 kb vs ~14 kb).

    patterns fused:
      DQ(a) DQ(w) DQ(b) Conv Q(out)      -> QLinearConv(a, w, b', out)
      DQ(a) Relu Q(out)                  -> Relu(a_q -> out)          (int8)
      DQ(a) MaxPool Q(out)               -> MaxPool(a_q -> out)       (int8)
    bias b' = b_q * b_scale / (x_scale * w_scale[m]) as int32.
    """
    import copy
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.load(src)
    g = m.graph
    nodes = list(g.node)

    producers = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producers[o] = i

    def node_of(tensor):
        i = producers.get(tensor)
        return nodes[i] if i is not None else None

    inits = {i.name: i for i in g.initializer}
    new_inits = {}

    def arr_of(name):
        t = inits.get(name) or new_inits.get(name)
        return numpy_helper.to_array(t) if t is not None else None

    def scalar_of(name):
        a = arr_of(name)
        return float(a.reshape(-1)[0]) if a is not None else 1.0

    removed = [False] * len(nodes)
    replaced = {}
    fused = 0

    for i, n in enumerate(nodes):
        if n.op_type not in ("Conv", "MatMul"):
            continue
        ins = list(n.input)
        if len(ins) < 2:
            continue
        dqs = [node_of(t) for t in ins]
        if any(p is None or p.op_type != "DequantizeLinear" for p in dqs):
            continue
        consumers = [x for x in nodes if n.output[0] in x.input]
        if len(consumers) != 1 or consumers[0].op_type != "QuantizeLinear":
            continue
        qn = consumers[0]
        x_scale = scalar_of(dqs[0].input[1])
        w_scale = numpy_helper.to_array(inits[dqs[1].input[1]])  # scalar or [M]

        bias_name = None
        inputs = []
        for k, p in enumerate(dqs):
            src_q = p.input[0]
            if k == 2:  # bias: rescale into qlinear int32 semantics
                b_arr = numpy_helper.to_array(inits[src_q]).astype(np.float64)
                b_scale = scalar_of(p.input[1])
                ws = np.asarray(w_scale, dtype=np.float64).reshape(-1)
                denom = x_scale * (ws[:b_arr.size] if b_arr.size > 1 else ws[0])
                b_q = np.round(b_arr * b_scale / denom).astype(np.int32)
                bias_name = f"bias_{qn.output[0]}"
                new_inits[bias_name] = numpy_helper.from_array(b_q, bias_name)
            else:
                inputs.extend([src_q, p.input[1], p.input[2] if len(p.input) > 2 else ""])
        # qlinearconv/matmul input order: x, xs, xzp, w, ws, wzp, y_scale, y_zero_point, B
        inputs.extend([qn.input[1], qn.input[2] if len(qn.input) > 2 else ""])
        if bias_name:
            inputs.append(bias_name)

        fused_node = helper.make_node(
            "QLinearConv" if n.op_type == "Conv" else "QLinearMatMul",
            inputs, [qn.output[0]], name=n.name)
        fused_node.attribute.extend(copy.deepcopy(n.attribute))
        replaced[i] = fused_node
        removed[i] = True
        for p in dqs:
            removed[producers[p.output[0]]] = True
        for j, x in enumerate(nodes):
            if x is qn:
                removed[j] = True
        fused += 1

    # relu / maxpool on dequantized tensors -> int8 passthrough
    for i, n in enumerate(nodes):
        if removed[i] or n.op_type not in ("Relu", "MaxPool"):
            continue
        p = node_of(n.input[0]) if n.input else None
        if p is None or removed[producers.get(n.input[0], -1)] if False else False:
            continue
        pi = producers.get(n.input[0])
        p = nodes[pi] if pi is not None else None
        if p is None or p.op_type != "DequantizeLinear":
            continue
        consumers = [x for x in nodes if n.output[0] in x.input]
        if len(consumers) != 1 or consumers[0].op_type != "QuantizeLinear":
            continue
        qn = consumers[0]
        out_node = helper.make_node(n.op_type, [p.input[0]], [qn.output[0]], name=n.name)
        out_node.attribute.extend(copy.deepcopy(n.attribute))
        replaced[i] = out_node
        removed[i] = True
        removed[pi] = True
        for j, x in enumerate(nodes):
            if x is qn:
                removed[j] = True
        fused += 1

    if not fused:
        return 0
    out = []
    for i, n in enumerate(nodes):
        if removed[i]:
            if i in replaced:
                out.append(replaced[i])
            continue
        out.append(n)
    del g.node[:]
    g.node.extend(out)
    g.initializer.extend(new_inits.values())
    onnx.save(m, dst)
    return fused


def main():
    w = dict(np.load("models/weights.npz"))
    onnx.save(build_graph(w), "models/detector.onnx")

    # calibration must span both classes: esc-50 is ordered alphabetically,
    # so a head slice is all one class and the quantized model saturates on
    # the other. seeded stratified sample instead.
    rng = np.random.default_rng(11)
    mall = np.load("data/cache/mels_train.npy").transpose(0, 2, 1)
    yall = np.load("data/train.npz")["y"]
    pos = np.where(yall == 1)[0]
    neg = np.where(yall == 0)[0]
    idx = np.concatenate([
        rng.choice(pos, min(256, len(pos)), replace=False),
        rng.choice(neg, 512, replace=False),
    ])
    M = mall[idx]
    quantize_static(
        "models/detector.onnx", "models/detector_int8.onnx", Reader(M),
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
        per_channel=True, extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    n_fused = fuse_qdq_to_qlinear("models/detector_int8.onnx", "models/detector_int8.onnx")
    print(f"fused {n_fused} qdq patterns into qlinear ops")

    rng = np.random.default_rng(3)
    X = rng.standard_normal((32, N_MELS, FRAME_LEN)).astype(np.float32) * 2
    pf, pq = run("models/detector.onnx", X), run("models/detector_int8.onnx", X)
    agree = np.mean((pf > 0.5) == (pq > 0.5))
    print(f"export ok. float-vs-int8 agreement {agree:.3f}, max|dp| {np.abs(pf-pq).max():.4f}")

    # the exported graph must reproduce the numpy trainer on real data: a
    # transpose bug in the conv weights passes the random-input check above
    # but inverts real scores.
    from train_core import Net, sig
    net = Net(np.random.default_rng(0))
    net.c1.w, net.c1.b = w["c1w"], w["c1b"]
    net.c2.w, net.c2.b = w["c2w"], w["c2b"]
    net.c3.w, net.c3.b = w["c3w"], w["c3b"]
    net.fc, net.fcb = w["fcw"], w["fcb"]
    Mt = np.load("data/cache/mels_test.npy")[:64].transpose(0, 2, 1)[:, :, :, None]
    Mtn = np.clip((Mt - w["mu"].reshape(1, 26, 1, 1)) / w["sd"].reshape(1, 26, 1, 1), -5, 5)
    ref = sig(net.forward(Mtn))[:, 0]
    got = run("models/detector.onnx", Mt[:, :, :, 0])
    err = float(np.abs(ref - got).max())
    print(f"numpy-vs-onnx max err {err:.4f}")
    assert err < 0.02, "exported graph disagrees with the trained weights"

    # golden fixtures use REAL spectrograms (+ small noise): gaussian noise
    # saturates every quantizer and turns rounding-mode deltas into sign
    # flips at the final dot product. in-distribution inputs pin the math
    # that actually matters.
    Mt = np.load("data/cache/mels_test.npy")[:8].transpose(0, 2, 1)
    golden_X = (Mt + rng.standard_normal(Mt.shape) * 0.05).astype(np.float32)
    Path("tests/golden").mkdir(exist_ok=True)
    out = ["// generated by tools/export_model.py - do not edit.", "#pragma once", ""]

    def lit(v):
        s = f"{float(v):.6g}"
        if "e" not in s and "." not in s:
            s += ".0"
        return s + "f"

    for i in range(8):
        out.append(f"inline constexpr float kIn{i}[] = {{ {', '.join(lit(v) for v in golden_X[i].ravel())} }};")
        out.append(f"inline constexpr float kOutF{i} = {lit(run('models/detector.onnx', golden_X[i:i+1])[0])};")
        out.append(f"inline constexpr float kOutQ{i} = {lit(run('models/detector_int8.onnx', golden_X[i:i+1])[0])};")
        out.append("")
    out.append("inline constexpr int kGoldenCount = 8;")
    out.append("inline constexpr int kSpecBands = 26;")
    out.append("inline constexpr int kSpecFrames = 61;")
    (Path("tests/golden") / "model_fixtures.hpp").write_text("\n".join(out) + "\n")
    print("wrote tests/golden/model_fixtures.hpp")


if __name__ == "__main__":
    main()
