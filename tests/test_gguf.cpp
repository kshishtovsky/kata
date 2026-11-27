// GGUF parser tests against a synthetic file.
//
// src/gguf.cpp was at 1.24% coverage: every one of its error paths (bad magic,
// truncation, bad rank, unsupported type, tensor out of range) could regress
// silently, and a wrong parse is not a crash -- it is a model that loads and
// produces garbage. Writing the bytes here keeps the tests independent of any
// multi-gigabyte file on disk, which is the only reason this was never done.
#include "kata/gguf.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <functional>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t kQ4_K = 12;
constexpr uint32_t kF32 = 0;

struct Writer {
    std::vector<uint8_t> b;

    template <class T> void le(T v) {
        uint8_t buf[sizeof(T)];
        std::memcpy(buf, &v, sizeof(T));
        b.insert(b.end(), buf, buf + sizeof(T));
    }
    void str(const std::string & s) {
        le<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void pad_to(size_t align) {
        while (b.size() % align) b.push_back(0);
    }
};

// One tensor with the given element count; q4_K packs 256 values per block.
struct TensorSpec {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> dims;
};

class GgufBuilder {
public:
    GgufBuilder & kv_u32(const std::string & k, uint32_t v) {
        kv_.push_back([k, v](Writer & w) {
            w.str(k); w.le<uint32_t>(4); w.le<uint32_t>(v);   // type UINT32
        });
        return *this;
    }
    GgufBuilder & kv_f32(const std::string & k, float v) {
        kv_.push_back([k, v](Writer & w) {
            w.str(k); w.le<uint32_t>(6); w.le<float>(v);      // type FLOAT32
        });
        return *this;
    }
    GgufBuilder & kv_str(const std::string & k, const std::string & v) {
        kv_.push_back([k, v](Writer & w) {
            w.str(k); w.le<uint32_t>(8); w.str(v);            // type STRING
        });
        return *this;
    }
    GgufBuilder & tensor(const TensorSpec & t) {
        tensors_.push_back(t);
        return *this;
    }
    GgufBuilder & version(uint32_t v) { version_ = v; return *this; }
    GgufBuilder & magic(uint32_t m) { magic_ = m; return *this; }

    std::vector<uint8_t> build() {
        Writer w;
        w.le<uint32_t>(magic_);
        w.le<uint32_t>(version_);
        w.le<uint64_t>(tensors_.size());
        w.le<uint64_t>(kv_.size());
        for (auto & f : kv_) f(w);

        // Offsets are computed as the file is emitted, exactly as a real writer
        // must: the table comes before the data, so the size of the table is
        // not known until every name and dim has been written.
        std::vector<uint64_t> offsets;
        uint64_t cursor = 0;
        for (const auto & t : tensors_) {
            size_t n = 1;
            for (auto d : t.dims) n *= (size_t)d;
            const size_t bs = kata::ggml_type_block_size(t.type);
            const size_t ts = kata::ggml_type_size(t.type);
            offsets.push_back(cursor);
            cursor += bs ? (n / bs) * ts : 0;
        }
        for (size_t i = 0; i < tensors_.size(); ++i) {
            const auto & t = tensors_[i];
            w.str(t.name);
            w.le<uint32_t>((uint32_t)t.dims.size());
            for (auto d : t.dims) w.le<uint64_t>(d);
            w.le<uint32_t>(t.type);
            w.le<uint64_t>(offsets[i]);
        }
        w.pad_to(32);
        // Data section: zeroes of the right total length.
        w.b.resize(w.b.size() + cursor, 0);
        return w.b;
    }

    // Writes to a temp file and hands the path back; the caller owns removal.
    std::string write_temp(const char * tag) {
        const std::vector<uint8_t> bytes = build();
        const std::string path = std::string("/tmp/kata_gguf_test_") + tag + ".gguf";
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) return {};
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
        return path;
    }

private:
    uint32_t magic_ = GGUF_MAGIC;
    uint32_t version_ = 3;
    std::vector<std::function<void(Writer &)>> kv_;
    std::vector<TensorSpec> tensors_;
};

class TempFile {
public:
    explicit TempFile(const char * tag) : path_(GgufBuilder().write_temp(tag)) {}
    ~TempFile() { if (!path_.empty()) std::remove(path_.c_str()); }
    const std::string & path() const { return path_; }
private:
    std::string path_;
};

} // namespace

TEST(Gguf, ReadsMetadataAndFindsTensors) {
    GgufBuilder b;
    b.kv_str("general.architecture", "lfm2moe")
     .kv_u32("lfm2moe.block_count", 40)
     .kv_u32("lfm2moe.embedding_length", 2048)
     .kv_f32("lfm2moe.attention.layer_norm_rms_epsilon", 1e-5f)
     .tensor({"a.weight", kQ4_K, {2048, 4}})
     .tensor({"b.weight", kQ4_K, {512, 2}});

    const std::string path = b.write_temp("meta");
    ASSERT_FALSE(path.empty());
    kata::GgufFile g;
    g.open(path);

    EXPECT_EQ(g.architecture(), "lfm2moe");
    EXPECT_EQ(g.u32("lfm2moe.block_count"), 40u);
    EXPECT_EQ(g.u32("lfm2moe.embedding_length"), 2048u);
    EXPECT_FLOAT_EQ(g.f32("lfm2moe.attention.layer_norm_rms_epsilon"), 1e-5f);
    EXPECT_TRUE(g.has_key("lfm2moe.block_count"));
    EXPECT_FALSE(g.has_key("nope"));
    EXPECT_EQ(g.tensors().size(), 2u);
    ASSERT_NE(g.find("b.weight"), nullptr);
    EXPECT_EQ(g.find("b.weight")->dims[0], 512u);
    EXPECT_EQ(g.find("b.weight")->dims[1], 2u);
    EXPECT_EQ(g.find("b.weight")->type, kQ4_K);
    std::remove(path.c_str());
}

TEST(Gguf, ComputesByteSizeFromTypeAndShape) {
    GgufBuilder b;
    // q4_K is 144 bytes per 256 values, so 2048x4 values is 32 blocks = 4608 B.
    b.tensor({"w", kQ4_K, {2048, 4}});
    const std::string path = b.write_temp("bytes");
    kata::GgufFile g;
    g.open(path);
    const kata::GgufTensor * t = g.find("w");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->n_elements, 8192u);
    EXPECT_EQ(t->n_bytes, 32u * 144u);
    std::remove(path.c_str());
}

TEST(Gguf, RejectsBadMagic) {
    GgufBuilder b;
    b.magic(0xdeadbeef).tensor({"w", kQ4_K, {256, 1}});
    const std::string path = b.write_temp("magic");
    kata::GgufFile g;
    EXPECT_THROW(g.open(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Gguf, RejectsUnsupportedVersion) {
    GgufBuilder b;
    b.version(7).tensor({"w", kQ4_K, {256, 1}});
    const std::string path = b.write_temp("version");
    kata::GgufFile g;
    EXPECT_THROW(g.open(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Gguf, RejectsMissingFile) {
    kata::GgufFile g;
    EXPECT_THROW(g.open("/tmp/kata_does_not_exist_xyz.gguf"), std::runtime_error);
}

TEST(Gguf, RejectsMissingKeyRatherThanReturningZero) {
    GgufBuilder b;
    b.kv_u32("present", 1).tensor({"w", kQ4_K, {256, 1}});
    const std::string path = b.write_temp("missing");
    kata::GgufFile g;
    g.open(path);
    // A silent 0 here would become a zero-length dimension in the model loader,
    // which is how an architecture mismatch turns into a division by zero much
    // later and much further from the cause.
    EXPECT_THROW(g.u32("absent"), std::runtime_error);
    EXPECT_THROW(g.f32("absent"), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Gguf, RejectsTruncatedFile) {
    GgufBuilder b;
    b.tensor({"w", kQ4_K, {2048, 4}});
    std::vector<uint8_t> bytes = b.build();
    bytes.resize(bytes.size() / 2);   // cut the data section in half
    const std::string path = "/tmp/kata_gguf_test_trunc.gguf";
    FILE * f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);

    kata::GgufFile g;
    // Either the open or the data() must reject it; a short mmap that silently
    // reads past the end is worse than an exception.
    bool threw = false;
    try {
        g.open(path);
        const kata::GgufTensor * t = g.find("w");
        if (t) (void)g.data(*t);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    EXPECT_TRUE(threw);
    std::remove(path.c_str());
}

TEST(Gguf, TensorDataPointerIsInRangeAndDistinctPerTensor) {
    GgufBuilder b;
    b.tensor({"a", kQ4_K, {256, 2}}).tensor({"b", kQ4_K, {256, 3}});
    const std::string path = b.write_temp("data");
    kata::GgufFile g;
    g.open(path);
    const auto * a = g.find("a");
    const auto * bb = g.find("b");
    ASSERT_NE(a, nullptr);
    ASSERT_NE(bb, nullptr);
    const uint8_t * pa = static_cast<const uint8_t *>(g.data(*a));
    const uint8_t * pb = static_cast<const uint8_t *>(g.data(*bb));
    EXPECT_NE(pa, pb);
    EXPECT_EQ(pb - pa, (ptrdiff_t)(2u * 144u));   // b follows a's 2 blocks
    std::remove(path.c_str());
}

TEST(Gguf, AlignmentFallsBackTo32WhenAbsent) {
    GgufBuilder b;
    b.tensor({"w", kQ4_K, {256, 1}});
    const std::string path = b.write_temp("align");
    kata::GgufFile g;
    g.open(path);
    EXPECT_EQ(g.alignment(), 32u);
    std::remove(path.c_str());
}