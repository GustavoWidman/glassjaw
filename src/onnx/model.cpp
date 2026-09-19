#include "model.hpp"

#include <cstring>

#include "pb.hpp"

namespace gj::onnx {

namespace {

float le_f32(const std::byte* p) {
  uint32_t v = static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
               static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}

int32_t le_i32(const std::byte* p) {
  return static_cast<int32_t>(static_cast<uint32_t>(p[0]) |
                              static_cast<uint32_t>(p[1]) << 8 |
                              static_cast<uint32_t>(p[2]) << 16 |
                              static_cast<uint32_t>(p[3]) << 24);
}

// ---- tensor proto -----------------------------------------------------
// TensorProto fields: dims=1(r int64), data_type=2, float_data=4(r float),
// int32_data=5, int64_data=7, name=8, raw_data=9, double_data=10,
// uint64_data=11, data_location=14.
Tensor parse_tensor(pb::Reader r) {
  Tensor t;
  bool have_raw = false;
  std::span<const std::byte> raw{};
  std::vector<float> float_data;
  std::vector<int64_t> int64_data;
  std::vector<int32_t> int32_data;
  int64_t data_type = 1;

  while (!r.eof()) {
    uint32_t tag = r.tag();
    uint32_t field = tag >> 3, wire = tag & 7;
    switch (field) {
      case 1: t.dims.push_back(static_cast<int64_t>(r.varint())); break;
      case 2: data_type = static_cast<int64_t>(r.varint()); break;
      case 4: {  // packed floats
        auto b = r.bytes();
        for (std::size_t i = 0; i + 4 <= b.size(); i += 4) float_data.push_back(le_f32(b.data() + i));
        break;
      }
      case 5: int32_data.push_back(static_cast<int32_t>(r.varint())); break;
      case 7: int64_data.push_back(static_cast<int64_t>(r.varint())); break;
      case 8: r.skip(2); break;  // tensor name: unused (initializers keyed by graph)
      case 9: raw = r.bytes(); have_raw = true; break;
      default: r.skip(wire); break;
    }
  }

  switch (data_type) {
    case 1:  // FLOAT
      t.kind = Tensor::Kind::F32;
      if (have_raw) {
        t.f32.resize(raw.size() / 4);
        for (std::size_t i = 0; i < t.f32.size(); ++i) t.f32[i] = le_f32(raw.data() + i * 4);
      } else {
        t.f32 = std::move(float_data);
      }
      break;
    case 2:  // UINT8 (stored as int8 bits; Tensor::u8 marks it)
      t.kind = Tensor::Kind::INT8;
      t.u8 = true;
      if (have_raw) {
        t.i8.resize(raw.size());
        for (std::size_t i = 0; i < t.i8.size(); ++i)
          t.i8[i] = static_cast<int8_t>(raw[i]);
      }
      break;
    case 3:  // INT8
      t.kind = Tensor::Kind::INT8;
      if (have_raw) {
        t.i8.resize(raw.size());
        for (std::size_t i = 0; i < t.i8.size(); ++i)
          t.i8[i] = static_cast<int8_t>(raw[i]);
      }
      break;
    case 6:  // INT32
      t.kind = Tensor::Kind::INT32;
      if (have_raw) {
        t.i32.resize(raw.size() / 4);
        for (std::size_t i = 0; i < t.i32.size(); ++i) t.i32[i] = le_i32(raw.data() + i * 4);
      } else {
        t.i32.resize(int32_data.size());
        for (std::size_t i = 0; i < int32_data.size(); ++i) t.i32[i] = int32_data[i];
      }
      break;
    case 7:  // INT64
      t.kind = Tensor::Kind::INT64;
      if (have_raw) {
        t.i64.resize(raw.size() / 8);
        for (std::size_t i = 0; i < t.i64.size(); ++i) {
          uint64_t v = 0;
          auto* p = raw.data() + i * 8;
          for (int b = 0; b < 8; ++b)
            v |= static_cast<uint64_t>(p[b]) << (8 * b);
          t.i64[i] = static_cast<int64_t>(v);
        }
      } else {
        t.i64 = std::move(int64_data);
      }
      break;
    default: break;  // unsupported dtype: numel() will disagree; loader rejects
  }
  return t;
}

// ---- attribute proto ----------------------------------------------------
// AttributeProto: name=1, f=2, i=3, s=4, floats=7, ints=8, type=20.
Attribute parse_attribute(pb::Reader r) {
  Attribute a;
  a.kind = Attribute::Kind::INT;
  while (!r.eof()) {
    uint32_t tag = r.tag();
    uint32_t field = tag >> 3, wire = tag & 7;
    switch (field) {
      case 1: r.skip(wire); break;  // name parsed separately
      case 2: {  // f: fixed32
        uint32_t bits = r.fixed32();
        std::memcpy(&a.f, &bits, 4);
        a.kind = Attribute::Kind::FLOAT;
        break;
      }
      case 3: a.kind = Attribute::Kind::INT; a.i = static_cast<int64_t>(r.varint()); break;
      case 7:  // floats: packed (wire 2) or single (wire 5)
        a.kind = Attribute::Kind::FLOATS;
        if (wire == 2) {
          auto b = r.bytes();
          for (std::size_t i = 0; i + 4 <= b.size(); i += 4) a.floats.push_back(le_f32(b.data() + i));
        } else if (wire == 5) {
          uint32_t bits = r.fixed32();
          float fv;
          std::memcpy(&fv, &bits, 4);
          a.floats.push_back(fv);
        } else {
          r.skip(wire);
        }
        break;
      case 8:  // ints: packed (wire 2) or single varint (wire 0)
        a.kind = Attribute::Kind::INTS;
        if (wire == 2) {
          auto b = r.bytes();
          for (std::size_t i = 0; i < b.size();) {
            uint64_t v = 0; int shift = 0;
            while (i < b.size()) {
              std::byte bb = b[i++];
              v |= static_cast<uint64_t>(bb & std::byte{0x7f}) << shift;
              if ((bb & std::byte{0x80}) == std::byte{0}) break;
              shift += 7;
            }
            a.ints.push_back(static_cast<int64_t>(v));
          }
        } else if (wire == 0) {
          a.ints.push_back(static_cast<int64_t>(r.varint()));
        } else {
          r.skip(wire);
        }
        break;
      default: r.skip(wire); break;
    }
  }
  return a;
}

// attribute name needs the raw field-1 string; parse separately.
std::string parse_attribute_name(pb::Reader r) {
  while (!r.eof()) {
    uint32_t tag = r.tag();
    uint32_t field = tag >> 3, wire = tag & 7;
    if (field == 1) return std::string(pb::as_str(r.bytes()));
    r.skip(wire);
  }
  return {};
}

// ---- value info -----------------------------------------------------------
ValueInfo parse_value_info(pb::Reader r) {
  ValueInfo vi;
  while (!r.eof()) {
    uint32_t tag = r.tag();
    uint32_t field = tag >> 3, wire = tag & 7;
    if (field == 1) {  // name
      vi.name = pb::as_str(r.bytes());
    } else if (field == 2) {  // TypeProto -> tensor_type=1
      auto tb = r.bytes();
      pb::Reader tr(tb);
      while (!tr.eof()) {
        uint32_t t2 = tr.tag();
        uint32_t f2 = t2 >> 3, w2 = t2 & 7;
        if (f2 == 1) {  // tensor_type
          auto ttb = tr.bytes();
          pb::Reader ttr(ttb);
          while (!ttr.eof()) {
            uint32_t t3 = ttr.tag();
            uint32_t f3 = t3 >> 3, w3 = t3 & 7;
            if (f3 == 1) {  // elem_type
              vi.elem_type = static_cast<int64_t>(ttr.varint());
            } else if (f3 == 2) {  // shape
              auto sb = ttr.bytes();
              pb::Reader sr(sb);
              while (!sr.eof()) {
                uint32_t t4 = sr.tag();
                uint32_t f4 = t4 >> 3, w4 = t4 & 7;
                if (f4 == 1) {  // dim
                  auto db = sr.bytes();
                  pb::Reader dr(db);
                  while (!dr.eof()) {
                    uint32_t t5 = dr.tag();
                    uint32_t f5 = t5 >> 3, w5 = t5 & 7;
                    if (f5 == 1) vi.dims.push_back(static_cast<int64_t>(dr.varint()));
                    else dr.skip(w5);
                  }
                } else {
                  sr.skip(w4);
                }
              }
            } else {
              ttr.skip(w3);
            }
          }
        } else {
          tr.skip(w2);
        }
      }
    } else {
      r.skip(wire);
    }
  }
  return vi;
}

}  // namespace

bool is_supported_op(const std::string& op) {
  static const char* kOps[] = {
      "Sub", "Div", "Clip", "Conv", "Relu", "MaxPool", "GlobalAveragePool",
      "MatMul", "Add", "Min", "Max", "Sigmoid", "QuantizeLinear",
      "DequantizeLinear", "QLinearConv", "QLinearMatMul", "Reshape", "Identity",
  };
  for (const char* k : kOps) {
    if (op == k) return true;
  }
  return false;
}

bool parse_model(std::span<const std::byte> bytes, Model& out, std::string& error) {
  pb::Reader r(bytes);
  std::span<const std::byte> graph_bytes{};
  bool have_graph = false;

  while (!r.eof()) {
    uint32_t tag = r.tag();
    uint32_t field = tag >> 3, wire = tag & 7;
    switch (field) {
      case 1: out.ir_version = static_cast<int64_t>(r.varint()); break;
      case 7: graph_bytes = r.bytes(); have_graph = true; break;
      case 8: {  // opset_import
        auto b = r.bytes();
        pb::Reader or_(b);
        while (!or_.eof()) {
          uint32_t t = or_.tag();
          uint32_t f = t >> 3, w = t & 7;
          if (f == 2) out.opset_version = static_cast<int64_t>(or_.varint());
          else or_.skip(w);
        }
        break;
      }
      default: r.skip(wire); break;
    }
  }
  if (!have_graph) {
    error = "no graph in model";
    return false;
  }

  pb::Reader g(graph_bytes);
  while (!g.eof()) {
    uint32_t tag = g.tag();
    uint32_t field = tag >> 3, wire = tag & 7;
    switch (field) {
      case 1: {  // NodeProto
        auto nb = g.bytes();
        pb::Reader nr(nb);
        Node node;
        while (!nr.eof()) {
          uint32_t t = nr.tag();
          uint32_t f = t >> 3, w = t & 7;
          switch (f) {
            case 1: node.inputs.emplace_back(pb::as_str(nr.bytes())); break;
            case 2: node.outputs.emplace_back(pb::as_str(nr.bytes())); break;
            case 4: node.op = pb::as_str(nr.bytes()); break;
            case 5: {  // attribute
              auto ab = nr.bytes();
              node.attrs[parse_attribute_name(pb::Reader(ab))] =
                  parse_attribute(pb::Reader(ab));
              break;
            }
            default: nr.skip(w); break;
          }
        }
        if (!is_supported_op(node.op)) {
          error = "unsupported op in graph: " + node.op;
          return false;
        }
        out.graph.nodes.push_back(std::move(node));
        break;
      }
      case 2: out.graph.name = pb::as_str(g.bytes()); break;
      case 5: {  // initializer
        auto tb = g.bytes();
        Tensor t = parse_tensor(pb::Reader(tb));
        // initializer name lives inside the tensor proto (field 8); reparse
        pb::Reader tnr(tb);
        std::string name;
        while (!tnr.eof()) {
          uint32_t t = tnr.tag();
          uint32_t f = t >> 3, w = t & 7;
          if (f == 8) name = pb::as_str(tnr.bytes());
          else tnr.skip(w);
        }
        out.graph.initializers.emplace(std::move(name), std::move(t));
        break;
      }
      case 11: out.graph.inputs.push_back(parse_value_info(pb::Reader(g.bytes()))); break;
      case 12: out.graph.outputs.push_back(parse_value_info(pb::Reader(g.bytes()))); break;
      default: g.skip(wire); break;
    }
  }

  // graph inputs that are not initializers are runtime inputs
  for (auto& vi : out.graph.inputs) {
    if (out.graph.initializers.count(vi.name) == 0) {
      out.graph.inputs_runtime.push_back(vi);
    }
  }
  return true;
}

}  // namespace gj::onnx
