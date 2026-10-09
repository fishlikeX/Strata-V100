// src/kernels/pf_wmma_parity.cpp - gemm_iq_f16_grouped (the Volta prompt experts, STRATA_PF_WMMA) against a double
// reference over the same FP16 weights (iq_dequant_f16, the FP16 path's dequantization) and FP16 activations.
//
//     build/pf_wmma_parity            (needs a CUDA GPU with FP16 tensor cores: sm_70+; synthetic blobs, no model)
//
// Every gate/up format the fork's native packs hand it (IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, and IQ2_XXS / IQ1_M through
// dq_dispatch: 1280 rows of 2560) and the down product (IQ4_NL and Q2_0: 2560 rows of 640).  A group of experts with
// 0, 1, 63, 64, 65, 200 and 300 rows (one past the 256-row chunk), the rows placed past a nonzero first bound,
// input and output row strides wider than the matrices, and - for the second run of each type - a sparse `ids`
// gather and an indirect weight table in reverse expert order. Every output row outside the bounds and every
// column past n_out must remain at its sentinel. Results must be finite. The FP32 tensor-core sums use another
// order than the double reference: max |got - ref| <= 2e-5 * (max |ref| of the matrix) + 1e-6.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 || t == 7 || t == 8 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of its four scale words (bytes 48..55), the sign and the
            // exponent's top bits in the last: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {   // the other formats start their block with the fp16 scale: 2^-13 .. 2^-6
            const int e = ex(rng);
            const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
            std::memcpy(&w[o], &h, 2);
        }
    }
    return w;
}

int run(int ty, int n_out, int K, std::mt19937& rng, bool use_ids) {
    const std::vector<int> rows = {0, 1, 63, 64, 65, 200, 300};
    const int E = (int) rows.size(), base = 5;
    std::vector<int32_t> bounds(E + 1);
    bounds[0] = base;
    for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + rows[e];
    const int total = bounds[E];
    const int ldx = K + 8, ldy = n_out + 8;      // row strides wider than the matrices (multiples of 8 / 4)
    const int xrows = total + 37;                // the ids gather may reach past the group's rows
    const int yrows = total + 16;
    const float SENT = -12345.5f;
    const size_t eb = k::iq_row_bytes(ty, K) * (size_t) n_out;
    std::vector<uint8_t> w;
    for (int e = 0; e < E; ++e) { auto r = random_rows(ty, n_out, K, rng); w.insert(w.end(), r.begin(), r.end()); }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> x((size_t) xrows * ldx);
    for (auto& v : x) v = __float2half(nd(rng));
    std::uniform_int_distribution<int> pick(0, xrows - 1);
    std::vector<int32_t> ids((size_t) total);
    for (int r = 0; r < total; ++r) ids[(size_t) r] = use_ids ? pick(rng) : r;
    std::vector<float> yh((size_t) yrows * ldy, SENT);
    uint8_t* dw; __half* dx; int32_t* db; int32_t* dids; float* dy; uint16_t* dq;
    ck(cudaMalloc(&dw, w.size() + 4096), "w");
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 2), "x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&db, bounds.size() * 4), "b");
    ck(cudaMemcpy(db, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "b");
    ck(cudaMalloc(&dids, ids.size() * 4), "ids");
    ck(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMalloc(&dy, yh.size() * 4), "y");
    ck(cudaMemcpy(dy, yh.data(), yh.size() * 4, cudaMemcpyHostToDevice), "y");
    ck(cudaMalloc(&dq, (size_t) E * n_out * K * 2), "dq");
    const int maxr = *std::max_element(rows.begin(), rows.end());
    const void** dptr = nullptr;
    std::vector<const void*> ptrs(E);
    for (int e = 0; e < E; ++e) ptrs[e] = dw + (E - 1 - e) * eb;
    ck(cudaMalloc(&dptr, ptrs.size() * sizeof(void*)), "weight pointers");
    ck(cudaMemcpy(dptr, ptrs.data(), ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice), "weight pointers");
    if (!k::gemm_iq_f16_grouped(ty, use_ids ? static_cast<const void*>(dptr) : dw, eb, n_out, K, dx, ldx,
                                db, E, maxr, dy, ldy, nullptr, use_ids ? dids : nullptr, use_ids)) {
        std::printf("%-8s %5d x %4d: not launched\n", name_of(ty), n_out, K);
        return 1;
    }
    for (int e = 0; e < E; ++e) k::iq_dequant_f16(ty, dw + e * eb, (int64_t) n_out * K, dq + (size_t) e * n_out * K, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> y(yh.size());
    std::vector<uint16_t> q((size_t) E * n_out * K);
    ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "y");
    ck(cudaMemcpy(q.data(), dq, q.size() * 2, cudaMemcpyDeviceToHost), "dq");
    int bad = 0;   // rows outside the bounds and columns past n_out must be untouched
    for (int r = 0; r < yrows; ++r)
        for (int n = 0; n < ldy; ++n)
            if ((r < base || r >= total || n >= n_out) && y[(size_t) r * ldy + n] != SENT) ++bad;
    double worst = 0, scale = 0;
    for (int e = 0; e < E; ++e)
        for (int r = bounds[e]; r < bounds[e + 1]; ++r) {
            const __half* xr = &x[(size_t) (use_ids ? ids[(size_t) r] : r) * ldx];
            for (int n = 0; n < n_out; ++n) {
                double s = 0;
                const int we = use_ids ? E - 1 - e : e;
                const uint16_t* wr = &q[((size_t) we * n_out + n) * K];
                for (int kk = 0; kk < K; ++kk) {
                    const __half h = __half_raw{wr[kk]};
                    s += (double) __half2float(h) * (double) __half2float(xr[kk]);
                }
                scale = std::max(scale, std::fabs(s));
                if (!std::isfinite(y[(size_t) r * ldy + n]) || !std::isfinite(s)) ++bad;
                worst = std::max(worst, std::fabs(s - (double) y[(size_t) r * ldy + n]));
            }
        }
    const bool ok = worst <= 2e-5 * scale + 1e-6 && bad == 0;
    std::printf("%-8s %5d x %4d, %d rows in %d experts%s: max |err| %.3g of max |ref| %.3g, %d stray %s\n", name_of(ty),
                n_out, K, total - base, E, use_ids ? " (ids)" : "", worst, scale, bad, ok ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dx); cudaFree(db); cudaFree(dids); cudaFree(dptr); cudaFree(dy); cudaFree(dq);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 7) { std::printf("pf_wmma_parity: needs sm_70+, skipped\n"); return 77; }
    std::mt19937 rng(70);
    int fails = 0;
    for (int ty : {18, 21, 22, 23, 16, 29}) {
        fails += run(ty, 1280, 2560, rng, false);
        fails += run(ty, 1280, 2560, rng, true);
    }
    for (int ty : {20, 42}) {
        fails += run(ty, 2560, 640, rng, false);
        fails += run(ty, 2560, 640, rng, true);
    }
    std::printf("pf_wmma_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
