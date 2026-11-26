// Perplexity stand for kata (EXP-0051 prerequisite): teacher-forced NLL over a
// token file, so every lossy change gets a number instead of an assumption.
#include "kata/model.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <model.gguf> <tokens.bin> [n_ctx] [max_tokens]\n", argv[0]); return 2; }
    const size_t n_ctx = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 512;
    std::ifstream f(argv[2], std::ios::binary);
    if (!f) { std::fprintf(stderr, "kata: cannot open %s\n", argv[2]); return 1; }
    std::vector<uint32_t> tok;
    uint32_t t;
    while (f.read(reinterpret_cast<char *>(&t), sizeof t)) tok.push_back(t);
    const size_t max_tok = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 2048;
    if (tok.size() > max_tok) tok.resize(max_tok);
    if (tok.size() < 3) { std::fprintf(stderr, "kata: need at least 3 tokens\n"); return 1; }
    kata::Model m;
    m.load(argv[1], n_ctx + 1);
    size_t counted = 0;
    const double pp = m.perplexity(tok.data(), tok.size(), n_ctx, &counted);
    std::printf("PPL %.4f over %zu tokens (n_ctx %zu, %zu of %zu used)\n",
                pp, counted, n_ctx, tok.size(), tok.size());
    return 0;
}
