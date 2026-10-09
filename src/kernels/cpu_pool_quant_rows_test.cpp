// src/kernels/cpu_pool_quant_rows_test.cpp - ExpertPool::quant_act_rows, the native prefill stage's row batch.
//
// A native CPU prefill stage quantizes each activation row through the same kernel the submitting thread's serial
// loop would call - `act_quant_any` for a Q2_0 gate/up layer Strata runs itself (`q2_native_kernels`), ggml-cpu's
// `native_quant_act` otherwise - and hands the rows to `quant_act_rows` so the pool spreads them over its
// threads.  The claim is small and exact, so it is checked byte for byte rather than through a timing number:
//
//   1. THE SAME BYTES AS SERIAL.  Every needed row must quantize to exactly what the serial per-row kernel
//      produces, for both activation paths (IQ3_S's native ggml quantization and Q2_0's own kernels), over
//      257 / 17 / 1 / 0 rows (shrinking, empty last), one and four workers, with and without host participation.
//   2. SKIPPED ROWS, PADDING AND EVERYTHING PAST THE BATCH ARE UNTOUCHED.  A row whose `need` flag is clear
//      must not be written, the bytes past a row's activation inside its `kNativeActBytes` slot must not be
//      either, and neither must the guard row after the last one.  Every buffer starts at a sentinel and the
//      whole allocation is compared, so a skipped row, a padding byte or a write past `n_rows` shows up as a
//      difference rather than as a plausible value.
//   3. THE MASK SHAPES THE ENGINE ACTUALLY SEES: dense, empty, the last row only, and every 31st row.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int kN = 2560;       // n_embd of the probed layer
constexpr int kMaxRows = 257;  // the largest row count checked
constexpr uint8_t kSentinel = 0xa5;

// One allocation set reused by every case: a row's inputs depend only on (row, column), so the first `rows` rows
// of the max-size buffers are already the right activations for a smaller case.
struct Fixture {
    std::vector<float> x;                   // kMaxRows * kN
    std::vector<char> need;                 // kMaxRows
    std::vector<cpu::ActQ> actq, ref_actq;  // kMaxRows + 1 (one guard row)
    std::vector<uint8_t> nact, ref_nact;    // (kMaxRows + 1) * kNativeActBytes (one guard row)

    Fixture()
        : x((size_t) kMaxRows * kN), need(kMaxRows), actq(kMaxRows + 1), ref_actq(kMaxRows + 1),
          nact((size_t) (kMaxRows + 1) * cpu::kNativeActBytes),
          ref_nact((size_t) (kMaxRows + 1) * cpu::kNativeActBytes) {
        for (int t = 0; t < kMaxRows; ++t)
            for (int j = 0; j < kN; ++j) {
                const float v = (float) ((t * 19 + j * 31) % 257 - 128) * 0.015625f;
                // Row 0 is all zero; row 2 is subnormal (a value in [-2, 2] x 1e-39) so the scale search sees
                // the bottom of the float range as well.
                x[(size_t) t * kN + j] = t == 0 ? 0.0f : t == 2 ? v * 1.0e-39f : v;
            }
    }
};

size_t first_diff(const uint8_t* got, const uint8_t* want, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (got[i] != want[i]) return i;
    return n;
}

}  // namespace

int main() {
    Fixture fx;
    int cases = 0, failures = 0;

    for (int host = 0; host < 2; ++host)
        for (int workers : {1, 4}) {
            cpu::ExpertPool pool(workers, true, host != 0);
            for (int gu : {21, 42}) {  // GGML_TYPE_IQ3_S (ggml-cpu activation) and GGML_TYPE_Q2_0 (Strata's kernels)
                cpu::NativeFmt f;
                std::string err;
                if (!cpu::native_fmt(gu, 20 /*GGML_TYPE_IQ4_NL*/, kN, 640, f, err)) {
                    std::fprintf(stderr, "cpu_pool_quant_rows_test: native_fmt(gu=%d): %s\n", gu, err.c_str());
                    return 2;
                }
                const bool q2 = cpu::q2_native_kernels(f.gu_type);
                // Shrinking row counts, empty last: at 1 row "last only" and "every 31st" are the dense mask
                // again, and at 0 rows every mask is the empty one, so those duplicates are not repeated.
                for (int rows : {257, 17, 1, 0}) {
                    const int patterns = rows == 0 ? 1 : rows == 1 ? 2 : 4;
                    for (int pattern = 0; pattern < patterns; ++pattern) {
                        for (int t = 0; t < rows; ++t)
                            fx.need[(size_t) t] = (char) (pattern == 0 || (pattern == 2 && t == rows - 1) ||
                                                          (pattern == 3 && t % 31 == 0));
                        // The whole allocation, guard row included, starts at the sentinel every case.
                        std::memset(fx.actq.data(), kSentinel, fx.actq.size() * sizeof(cpu::ActQ));
                        std::memset(fx.ref_actq.data(), kSentinel, fx.ref_actq.size() * sizeof(cpu::ActQ));
                        std::memset(fx.nact.data(), kSentinel, fx.nact.size());
                        std::memset(fx.ref_nact.data(), kSentinel, fx.ref_nact.size());

                        // The serial reference: the same per-row kernel the submitting thread would call.
                        for (int t = 0; t < rows; ++t) {
                            if (fx.need[(size_t) t] == 0) continue;
                            const float* xr = fx.x.data() + (size_t) t * kN;
                            if (q2)
                                cpu::act_quant_any(xr, kN, fx.ref_actq[(size_t) t]);
                            else
                                cpu::native_quant_act(f, xr, fx.ref_nact.data() + (size_t) t * cpu::kNativeActBytes);
                        }

                        const cpu::ExpertPool::ActRows batch{fx.x.data(), fx.need.data(), rows, kN,
                                                             q2 ? fx.actq.data() : nullptr,
                                                             q2 ? nullptr : fx.nact.data()};
                        pool.quant_act_rows(f, batch);

                        const uint8_t* got = q2 ? (const uint8_t*) fx.actq.data() : fx.nact.data();
                        const uint8_t* want = q2 ? (const uint8_t*) fx.ref_actq.data() : fx.ref_nact.data();
                        const size_t bytes = q2 ? fx.actq.size() * sizeof(cpu::ActQ) : fx.nact.size();
                        const size_t at = first_diff(got, want, bytes);
                        if (at != bytes) {
                            ++failures;
                            const size_t row_bytes = q2 ? sizeof(cpu::ActQ) : cpu::kNativeActBytes;
                            const size_t row = at / row_bytes;
                            const bool in_batch = row < (size_t) rows;
                            std::fprintf(stderr,
                                         "quant_act_rows mismatch: host=%d workers=%d gu=%d rows=%d pattern=%d "
                                         "(%s path) row %zu (%s) byte %zu: got 0x%02x want 0x%02x\n",
                                         host, workers, gu, rows, pattern, q2 ? "Q2_0" : "native", row,
                                         in_batch ? (fx.need[row] ? "need" : "skip") : "past batch",
                                         at % row_bytes, got[at], want[at]);
                        }
                        ++cases;
                    }
                }
            }
        }

    if (failures) {
        std::printf("cpu_pool_quant_rows_test: %d of %d cases differ from the serial reference\n", failures, cases);
        return 1;
    }
    std::printf("cpu_pool_quant_rows_test: %d cases byte-exact (native/Q2_0, 257/17/1/0 rows, dense/empty/last/"
                "sparse masks, host on/off, 1/4 workers); skipped rows, native padding and the guard row "
                "untouched\n",
                cases);
    return 0;
}
