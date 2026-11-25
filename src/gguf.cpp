#include "kata/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace kata {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;   // "GGUF" little endian

struct Reader {
    const uint8_t * p;
    const uint8_t * end;

    void need(size_t n) const {
        if ((size_t)(end - p) < n) throw std::runtime_error("gguf: truncated file");
    }
    template <class T> T le() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;   // x86/arm are little endian; a big-endian port needs a swap here
    }
    std::string str() {
        const uint64_t n = le<uint64_t>();
        need(n);
        std::string s((const char *)p, n);
        p += n;
        return s;
    }
    void skip(size_t n) {
        need(n);
        p += n;
    }
};

enum class MetaType : uint32_t {
    UINT8 = 0, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
    STRING, ARRAY, UINT64, INT64, FLOAT64,
};

size_t meta_scalar_size(MetaType t) {
    switch (t) {
        case MetaType::UINT8:
        case MetaType::INT8:
        case MetaType::BOOL:    return 1;
        case MetaType::UINT16:
        case MetaType::INT16:   return 2;
        case MetaType::UINT32:
        case MetaType::INT32:
        case MetaType::FLOAT32: return 4;
        case MetaType::UINT64:
        case MetaType::INT64:
        case MetaType::FLOAT64: return 8;
        default:                return 0;
    }
}

struct MetaValue {
    enum class Kind { None, Str, UInt, Int, Float, Bool, Array } kind = Kind::None;
    std::string s;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0.0;
    bool b = false;
};

MetaValue read_meta(Reader & r, MetaType t) {
    MetaValue v;
    switch (t) {
        case MetaType::STRING: v.kind = MetaValue::Kind::Str; v.s = r.str(); break;
        case MetaType::BOOL:   v.kind = MetaValue::Kind::Bool; v.b = r.le<uint8_t>() != 0; break;
        case MetaType::UINT8:  v.kind = MetaValue::Kind::UInt; v.u = r.le<uint8_t>(); break;
        case MetaType::UINT16: v.kind = MetaValue::Kind::UInt; v.u = r.le<uint16_t>(); break;
        case MetaType::UINT32: v.kind = MetaValue::Kind::UInt; v.u = r.le<uint32_t>(); break;
        case MetaType::UINT64: v.kind = MetaValue::Kind::UInt; v.u = r.le<uint64_t>(); break;
        case MetaType::INT8:   v.kind = MetaValue::Kind::Int; v.i = r.le<int8_t>(); break;
        case MetaType::INT16:  v.kind = MetaValue::Kind::Int; v.i = r.le<int16_t>(); break;
        case MetaType::INT32:  v.kind = MetaValue::Kind::Int; v.i = r.le<int32_t>(); break;
        case MetaType::INT64:  v.kind = MetaValue::Kind::Int; v.i = r.le<int64_t>(); break;
        case MetaType::FLOAT32: { v.kind = MetaValue::Kind::Float;
                                  const uint32_t bits = r.le<uint32_t>();
                                  float f; std::memcpy(&f, &bits, 4); v.f = f; break; }
        case MetaType::FLOAT64: { const uint64_t bits = r.le<uint64_t>();
                                  double d; std::memcpy(&d, &bits, 8); v.f = d; break; }
        case MetaType::ARRAY: {
            v.kind = MetaValue::Kind::Array;
            const MetaType et = (MetaType)r.le<uint32_t>();
            const uint64_t n = r.le<uint64_t>();
            if (const size_t ss = meta_scalar_size(et)) {
                r.skip((size_t)n * ss);       // skip the payload, we do not need it
            } else if (et == MetaType::STRING) {
                for (uint64_t k = 0; k < n; ++k) (void)r.str();
            } else {
                for (uint64_t k = 0; k < n; ++k) (void)read_meta(r, et);
            }
            break;
        }
        default: throw std::runtime_error("gguf: bad metadata type");
    }
    return v;
}

} // namespace

size_t ggml_type_block_size(uint32_t type) {
    switch (type) {
        case 0: case 1: case 24: case 25: case 26: case 27: case 28: case 30: return 1;
        case 2: case 3: case 6: case 7: case 8: case 9: case 20: return 32;
        default: return 256;   // all K-quants and IQ types
    }
}

size_t ggml_type_size(uint32_t type) {
    switch (type) {
        case 0: return 4;    // F32
        case 1: return 2;    // F16
        case 2: return 18;   // Q4_0
        case 3: return 20;   // Q4_1
        case 6: return 22;   // Q5_0
        case 7: return 24;   // Q5_1
        case 8: return 34;   // Q8_0
        case 9: return 36;   // Q8_1
        case 10: return 84;  // Q2_K
        case 11: return 110; // Q3_K
        case 12: return 144; // Q4_K
        case 13: return 176; // Q5_K
        case 14: return 210; // Q6_K
        case 15: return 292; // Q8_K
        case 20: return 18;  // IQ4_NL
        case 23: return 36;  // IQ4_XS
        case 16: return 66;  // IQ2_XXS: 2 + 32*2
        case 17: return 74;  // IQ2_XS: 2 + 32*2 + 8
        case 18: return 98;  // IQ3_XXS: 2 + 3*32
        case 19: return 50;  // IQ1_S: 2 + 32 + 16
        case 21: return 110; // IQ3_S: 2 + 13*8 + 4
        case 22: return 82;  // IQ2_S: 2 + 64 + 16
        case 29: return 56;  // IQ1_M: 32 + 16 + 8
        case 24: return 1;   // I8
        case 25: return 2;   // I16
        case 26: return 4;   // I32
        case 27: return 8;   // I64
        case 28: return 8;   // F64
        case 30: return 2;   // BF16
        default: return 0;
    }
}

const char * ggml_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        default: return "UNKNOWN";
    }
}

GgufFile::~GgufFile() {
    if (map_) munmap(map_, size_);
}

void GgufFile::open(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("gguf: cannot open " + path);

    struct stat st {};
    if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("gguf: fstat failed"); }
    size_ = (size_t)st.st_size;
    map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("gguf: mmap failed"); }

    Reader r{(const uint8_t *)map_, (const uint8_t *)map_ + size_};

    if (r.le<uint32_t>() != GGUF_MAGIC) throw std::runtime_error("gguf: bad magic");
    const uint32_t version = r.le<uint32_t>();
    if (version != 2 && version != 3) throw std::runtime_error("gguf: unsupported version " + std::to_string(version));

    const uint64_t n_tensors = r.le<uint64_t>();
    const uint64_t n_kv = r.le<uint64_t>();

    for (uint64_t k = 0; k < n_kv; ++k) {
        const std::string key = r.str();
        const MetaType type = (MetaType)r.le<uint32_t>();
        const MetaValue v = read_meta(r, type);

        if (key == "general.architecture" && v.kind == MetaValue::Kind::Str) architecture_ = v.s;
        else if (key == "general.alignment" && v.kind == MetaValue::Kind::UInt) alignment_ = (uint32_t)v.u;
        else if (v.kind == MetaValue::Kind::UInt) int_meta_.emplace_back(key, (uint32_t)v.u);
        else if (v.kind == MetaValue::Kind::Int) int_meta_.emplace_back(key, v.i < 0 ? 0u : (uint32_t)v.i);
        else if (v.kind == MetaValue::Kind::Float) float_meta_.emplace_back(key, v.f);
    }

    if (alignment_ == 0 || (alignment_ & (alignment_ - 1))) alignment_ = 32;

    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = r.str();
        const uint32_t n_dims = r.le<uint32_t>();
        if (n_dims == 0 || n_dims > 8) throw std::runtime_error("gguf: bad tensor rank");
        t.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) t.dims[d] = r.le<uint64_t>();
        t.type = r.le<uint32_t>();
        t.offset = r.le<uint64_t>();

        size_t n = 1;
        for (auto d : t.dims) n *= (size_t)d;
        t.n_elements = n;
        const size_t bs = ggml_type_block_size(t.type);
        const size_t ts = ggml_type_size(t.type);
        if (bs == 0 || ts == 0) {
            throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(t.type));
        }
        if (n % bs) throw std::runtime_error("gguf: tensor " + t.name + " size not a multiple of block size");
        t.n_bytes = n / bs * ts;
        tensors_.push_back(std::move(t));
    }

    // Data section starts after the tensor table, aligned up.
    const size_t pos = (size_t)(r.p - (const uint8_t *)map_);
    const size_t pad = (alignment_ - (pos % alignment_)) % alignment_;
    data_offset_ = pos + pad;
    if (data_offset_ > size_) throw std::runtime_error("gguf: data offset past end of file");
}

const GgufTensor * GgufFile::find(const std::string & name) const {
    for (const auto & t : tensors_) if (t.name == name) return &t;
    return nullptr;
}

const void * GgufFile::data(const GgufTensor & t) const {
    if (data_offset_ + t.offset + t.n_bytes > size_) throw std::runtime_error("gguf: tensor out of range");
    return (const uint8_t *)map_ + data_offset_ + t.offset;
}

bool GgufFile::has_key(const std::string & key) const {
    for (const auto & kv : int_meta_) if (kv.first == key) return true;
    for (const auto & kv : float_meta_) if (kv.first == key) return true;
    return false;
}

float GgufFile::f32(const std::string & key) const {
    for (const auto & kv : float_meta_) if (kv.first == key) return kv.second;
    throw std::runtime_error("gguf: missing key " + key);
}

uint32_t GgufFile::u32(const std::string & key) const {
    for (const auto & kv : int_meta_) if (kv.first == key) return kv.second;
    throw std::runtime_error("gguf: missing key " + key);
}

} // namespace kata
