#include "strata/core/vmm.hpp"

#if !defined(STRATA_USE_HIP)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>
#include <map>

namespace strata::core {
namespace {

// The entry points are the same for every device and are resolved once; granularity and VMM support are properties
// of an allocation location, so they are queried once per device (a second prefill stage runs on another device).
struct Api {
    bool ok = false;
    decltype(&cuDeviceGetAttribute) attr = nullptr;
    decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || p == nullptr)
        return false;
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        a.ok = resolve("cuDeviceGetAttribute", a.attr) &&
               resolve("cuMemGetAllocationGranularity", a.granularity) &&
               resolve("cuMemAddressReserve", a.reserve) && resolve("cuMemAddressFree", a.free_va) &&
               resolve("cuMemCreate", a.create) && resolve("cuMemRelease", a.release) &&
               resolve("cuMemMap", a.map) && resolve("cuMemUnmap", a.unmap) &&
               resolve("cuMemSetAccess", a.access);
    });
    return a;
}

struct DeviceCap {
    bool ok = false;
    uint64_t gran = 0;
};

// The current device's VMM support and granularity.  Cached per device: on a multi-GPU host the second stage's
// device is a different one, with its own granularity, and capturing either once (from GPU 0) would be wrong.
DeviceCap device_cap(int dev) {
    static std::mutex mu;
    static std::map<int, DeviceCap> cache;
    std::lock_guard<std::mutex> lk(mu);
    const auto it = cache.find(dev);
    if (it != cache.end()) return it->second;
    DeviceCap c;
    const Api& a = api();
    if (a.ok && dev >= 0) {
        int vmm = 0;
        if (a.attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) dev) == CUDA_SUCCESS && vmm) {
            CUmemAllocationProp prop{};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = dev;
            size_t g = 0;
            if (a.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && g != 0) {
                c.gran = g;
                c.ok = true;
            }
        }
    }
    cache.emplace(dev, c);
    return c;
}

int current_device() {
    int d = -1;
    return cudaGetDevice(&d) == cudaSuccess ? d : -1;
}

}  // namespace

bool vmm_available() { return device_cap(current_device()).ok; }
uint64_t vmm_granularity() { return device_cap(current_device()).gran; }

VmmChunk vmm_chunk_new(int device) {
    const Api& a = api();
    const DeviceCap c = device_cap(device);
    if (!a.ok || !c.ok) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) c.gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

VmmChunk vmm_chunk_new() { return vmm_chunk_new(current_device()); }

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    const int dev = current_device();
    const DeviceCap c = device_cap(dev);
    if (!a.ok || !c.ok || bytes == 0) return false;
    const uint64_t n = (bytes + c.gran - 1) / c.gran;
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * c.gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    dev_ = dev;
    gran_ = c.gran;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * gran_));
    base_ = 0;
    dev_ = -1;
    gran_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (a.map((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS)
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = dev_;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * gran_), (size_t) ((uint64_t) (hi - lo) * gran_), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (a.unmap((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_) != CUDA_SUCCESS) return 0;
    h_[(size_t) i] = 0;
    return h;
}

}  // namespace strata::core

#else  // HIP: no virtual memory here; the elastic K/V stays off

namespace strata::core {
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
VmmChunk vmm_chunk_new(int) { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace strata::core

#endif
