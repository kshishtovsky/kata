#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kata {

// Minimal GGUF v3 reader: header, metadata, tensor table and the data section
// (memory mapped, never copied). Unknown metadata is skipped, not rejected.
struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;   // ggml order: dims[0] varies fastest
    uint32_t type = 0;
    uint64_t offset = 0;          // bytes from the start of the data section
    size_t n_elements = 0;
    size_t n_bytes = 0;
};

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    // Throws std::runtime_error on a malformed or unsupported file.
    void open(const std::string & path);

    const std::vector<GgufTensor> & tensors() const { return tensors_; }
    const GgufTensor * find(const std::string & name) const;
    uint32_t alignment() const { return alignment_; }
    const std::string & architecture() const { return architecture_; }

    // Pointer into the mapping for a tensor's first byte.
    const void * data(const GgufTensor & t) const;

    uint32_t u32(const std::string & key) const;
    float f32(const std::string & key) const;
    bool has_key(const std::string & key) const;

private:
    void * map_ = nullptr;
    size_t size_ = 0;
    size_t data_offset_ = 0;
    uint32_t alignment_ = 32;
    std::string architecture_;
    std::vector<GgufTensor> tensors_;
    std::vector<std::pair<std::string, uint32_t>> int_meta_;
    std::vector<std::pair<std::string, float>> float_meta_;
};

// ggml type table, limited to the types kata can reason about.
size_t ggml_type_block_size(uint32_t type);
size_t ggml_type_size(uint32_t type);
const char * ggml_type_name(uint32_t type);

} // namespace kata
