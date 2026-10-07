#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include <algorithm>
#include <stdexcept>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
struct Batch {
    cpu::NativeFmt fmt;
    std::vector<uint8_t> blob, act;
    std::vector<float> out, expected;
    std::vector<cpu::ExpertJobMulti> jobs;
    Batch(int type, int count, int seed) {
        std::string err;
        if (!cpu::native_fmt(type, GGML_TYPE_Q8_0, cpu::H, cpu::FF, fmt, err))
            throw std::runtime_error(err);
        blob.resize(fmt.bytes);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.025f, 0.025f);
        auto quant = [&](int t, int rows, int cols, uint8_t* dst) {
            std::vector<float> w((size_t) rows * cols);
            for (float& v : w) v = dist(rng);
            std::vector<float> importance((size_t) cols, 1.f);
            ggml_quantize_chunk((ggml_type) t, w.data(), dst, 0, rows, cols,
                ggml_quantize_requires_imatrix((ggml_type) t) ? importance.data() : nullptr);
        };
        quant(type, cpu::FF, cpu::H, blob.data());
        quant(type, cpu::FF, cpu::H, blob.data() + fmt.up_off);
        quant(GGML_TYPE_Q8_0, cpu::H, cpu::FF, blob.data() + fmt.down_off);
        act.resize(2 * cpu::kNativeActBytes);
        std::vector<float> x(cpu::H);
        for (int t = 0; t < 2; ++t) {
            for (float& v : x) v = dist(rng) * 20.f;
            cpu::native_quant_act(fmt, x.data(), act.data() + t * cpu::kNativeActBytes);
        }
        out.resize((size_t) count * 2 * cpu::H);
        jobs.resize(count);
        for (int e = 0; e < count; ++e) {
            jobs[e].blob = blob.data();
            jobs[e].nt = 1 + e % 2;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = act.data() + t * cpu::kNativeActBytes;
                jobs[e].out[t] = out.data() + ((size_t) e * 2 + t) * cpu::H;
            }
        }
    }
    bool matches() const {
        for (size_t i = 0; i < out.size(); ++i)
            if (!std::isfinite(out[i]) || out[i] != expected[i]) return false;
        return true;
    }
};

int main() {
    // Different formats, token counts, and a batch that crosses the pool's scratch capacity.
    Batch a(GGML_TYPE_IQ3_S, 3, 17), b(GGML_TYPE_IQ2_S, cpu::ExpertPool::kMaxSplitMulti + 1, 29);
    cpu::ExpertPool pool(2, false, true);
    auto run = [&](Batch& batch) { pool.run_split_multi_native(batch.fmt, batch.jobs.data(), (int) batch.jobs.size()); };
    run(a); a.expected = a.out;
    run(b); b.expected = b.out;
    for (int round = 0; round < 6; ++round) {
        std::fill(a.out.begin(), a.out.end(), 0.f);
        std::fill(b.out.begin(), b.out.end(), 0.f);
        std::barrier start(3);
        std::thread ta([&] { start.arrive_and_wait(); run(a); });
        std::thread tb([&] { start.arrive_and_wait(); run(b); });
        start.arrive_and_wait(); ta.join(); tb.join();
        if (!a.matches() || !b.matches()) {
            std::fprintf(stderr, "concurrent native batches differ from serial outputs, round %d\n", round);
            return 1;
        }
    }
    pool.run_split_multi_native(a.fmt, nullptr, 0);
    run(a);
    if (!a.matches()) return 1;
    std::puts("concurrent native batches match serial outputs; pool remains reusable");
}
