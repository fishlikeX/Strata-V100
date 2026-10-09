// The device-local VMM contract (vmm.hpp, the split-stage --kv-grow): a range reserved on one device allocates its
// physical chunks, maps, sets access, unmaps and frees on THAT device, even when the ambient CUDA device has since
// changed.  The once-captured-global bug this guards against placed a second stage's chunks on the device of the
// process's first VMM call (GPU 0), so the physical device of every chunk is read back from the driver
// (cuMemGetAllocationPropertiesFromHandle) instead of being inferred from a copy that would succeed either way.
// Needs two VMM-capable devices; exit 77 (a CTest skip) otherwise.
#include "strata/core/vmm.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

using strata::core::VmmChunk;
using strata::core::VmmRange;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

namespace {

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || p == nullptr)
        return false;
    f = (F) p;
    return true;
}

struct Drv {
    decltype(&cuMemGetAllocationPropertiesFromHandle) props = nullptr;
    decltype(&cuMemRetainAllocationHandle) retain = nullptr;
    decltype(&cuMemRelease) release = nullptr;
};

// The device a handle's physical allocation lives on (the driver's own account, no inference).
int handle_device(const Drv& d, CUmemGenericAllocationHandle h, bool* ok) {
    *ok = false;
    CUmemAllocationProp p{};
    if (d.props(&p, h) != CUDA_SUCCESS) return -1;
    if (p.location.type != CU_MEM_LOCATION_TYPE_DEVICE) return -1;
    *ok = true;
    return (int) p.location.id;
}

// The same for a mapped address: retain its allocation (an extra reference, dropped again), then query it.
int mapped_device(const Drv& d, const void* addr, bool* ok) {
    CUmemGenericAllocationHandle h = 0;
    if (d.retain(&h, (void*) (uintptr_t) addr) != CUDA_SUCCESS) {
        *ok = false;
        return -1;
    }
    const int dev = handle_device(d, h, ok);
    d.release(h);
    return dev;
}

// The VMM-capable devices, in order.
std::vector<int> vmm_devices() {
    std::vector<int> out;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return out;
    decltype(&cuDeviceGetAttribute) attr = nullptr;
    if (!resolve("cuDeviceGetAttribute", attr)) return out;
    for (int d = 0; d < n; ++d) {
        int vmm = 0;
        if (attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) d) == CUDA_SUCCESS && vmm)
            out.push_back(d);
    }
    return out;
}

void check_range_device(const Drv& d, const VmmRange& r, int dev) {
    for (int64_t i = 0; i < r.chunks(); ++i) {
        bool ok = false;
        const int pd = mapped_device(d, r.base() + (uint64_t) i * r.granularity(), &ok);
        CHECK(ok && pd == dev);
    }
}

}  // namespace

int main() {
    const std::vector<int> devs = vmm_devices();
    if (devs.size() < 2) {
        std::printf("vmm_device_local_test: fewer than two VMM-capable CUDA devices - skipped\n");
        return 77;
    }
    const int d0 = devs[0], d1 = devs[1];

    Drv drv;
    if (!resolve("cuMemGetAllocationPropertiesFromHandle", drv.props) || !resolve("cuMemRetainAllocationHandle", drv.retain) ||
        !resolve("cuMemRelease", drv.release)) {
        std::printf("vmm_device_local_test: allocation-properties entry points unavailable - skipped\n");
        return 77;
    }

    // Each device's primary context, bound so driver calls act on it (and the first vmm call is on d0, which the
    // once-captured-global bug would then use for every later device).
    for (const int d : {d0, d1}) {
        CHECK(cudaSetDevice(d) == cudaSuccess);
        CHECK(cudaFree(nullptr) == cudaSuccess);
    }

    // ---- a range reserved on d0 maps chunks physically on d0 ----
    CHECK(cudaSetDevice(d0) == cudaSuccess);
    CHECK(strata::core::vmm_available());
    const uint64_t G0 = strata::core::vmm_granularity();
    CHECK(G0 >= 4096);
    VmmRange a;
    CHECK(a.reserve(4 * G0 + 1));   // rounds up to 5 chunks
    CHECK(a.device() == d0);
    CHECK(a.granularity() == G0);
    CHECK(a.chunks() == 5);
    CHECK(a.map_range(0, a.chunks(), [] { return (VmmChunk) 0; }));
    check_range_device(drv, a, d0);

    // ---- a range reserved on d1 maps chunks physically on d1 (the bug put these on d0) ----
    CHECK(cudaSetDevice(d1) == cudaSuccess);
    CHECK(strata::core::vmm_available());
    const uint64_t G1 = strata::core::vmm_granularity();
    CHECK(G1 >= 4096);
    VmmRange b;
    CHECK(b.reserve(3 * G1));
    CHECK(b.device() == d1);
    CHECK(b.granularity() == G1);
    CHECK(b.chunks() == 3);
    CHECK(b.map_range(0, b.chunks(), [] { return (VmmChunk) 0; }));
    check_range_device(drv, b, d1);

    // ---- a new chunk for an explicit device targets it, not the ambient one ----
    CHECK(cudaSetDevice(d1) == cudaSuccess);
    const VmmChunk x0 = strata::core::vmm_chunk_new(d0);
    CHECK(x0 != 0);
    { bool ok = false; CHECK(handle_device(drv, (CUmemGenericAllocationHandle) x0, &ok) == d0 && ok); }
    strata::core::vmm_chunk_free(x0);
    CHECK(cudaSetDevice(d0) == cudaSuccess);
    const VmmChunk x1 = strata::core::vmm_chunk_new(d1);
    CHECK(x1 != 0);
    { bool ok = false; CHECK(handle_device(drv, (CUmemGenericAllocationHandle) x1, &ok) == d1 && ok); }
    strata::core::vmm_chunk_free(x1);

    // ---- unmap/remap and data preservation on d0's range while the ambient device is d1 ----
    std::vector<uint8_t> h(2 * G0), back(2 * G0);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t) (i * 2654435761u >> 11);
    CHECK(cudaSetDevice(d0) == cudaSuccess);
    CHECK(cudaMemcpy(a.base(), h.data(), h.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    // an in-flight pageable copy must be drained before a chunk is unmapped under it (as generate.cpp does)
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaSetDevice(d1) == cudaSuccess);   // the range's device is NOT the ambient one now
    const VmmChunk keep = a.unmap(1);
    CHECK(keep != 0 && !a.mapped(1));
    { bool ok = false; CHECK(handle_device(drv, (CUmemGenericAllocationHandle) keep, &ok) == d0 && ok); }
    CHECK(a.map_range(1, 2, [&] { return keep; }));   // remapped into d0's range, on the same physical chunk
    CHECK(a.mapped(1) && a.map_range(1, 2, [] { return (VmmChunk) 0; }));   // already mapped: a no-op
    CHECK(cudaSetDevice(d0) == cudaSuccess);
    CHECK(cudaMemcpy(back.data(), a.base() + G0, G0, cudaMemcpyDeviceToHost) == cudaSuccess);
    bool same = true;
    for (uint64_t i = 0; i < G0; ++i) same = same && back[i] == h[G0 + i];
    CHECK(same);

    // ---- release both ranges while the ambient device is d1: the captured device/granularity drive the frees ----
    CHECK(cudaSetDevice(d1) == cudaSuccess);
    a.release();
    b.release();
    CHECK(a.device() == -1 && a.granularity() == 0 && a.chunks() == 0);
    CHECK(b.device() == -1 && b.granularity() == 0 && b.chunks() == 0);
    CHECK(cudaGetLastError() == cudaSuccess);

    std::printf("vmm_device_local_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
