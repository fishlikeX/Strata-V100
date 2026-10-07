// src/program/conv_cache_test.cpp - the conversation cache's retention policy: the shared root survives,
// the leaves rotate by least recent use.
//
// The scenarios, in the serve loop's terms (generate.cpp): checkpoints are a prefix chain - stamps[0] is
// the root, the deepest point every request so far has shared (the end of the system prompt, in practice).
// A stamp advances on creation, when a new checkpoint lands on an existing length, and when a request
// mounts through the checkpoint.
//
//   1. one conversation over its budget: the oldest leaf leaves - with fresh stamps in creation order this
//      is exactly the previous first-in-first-out behaviour;
//   2. the root survives arbitrarily many newer checkpoints however old its stamp is;
//   3. a mount advances a checkpoint's stamp and the rotation then takes a different leaf;
//   4. a one-slot budget has no room for root and leaf, so the pin is off and the oldest leaves (the
//      newest point stays, as before);
//   5. the policy is pure: the same stamps, the same victim.
//
// The switch tier order is a second pure rule: the longest compatible prefix wins, and an equal-length
// RAM image beats the disk record (disk_reuse_wins / ram_reuse_wins).
#include "strata/program/conv_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using strata::program::conv_cache::eviction_victim;
using strata::program::conv_cache::disk_reuse_wins;
using strata::program::conv_cache::ram_reuse_wins;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("conv_cache_test\n");
    {
        const std::vector<uint64_t> stamps = {1, 2, 3, 4, 5, 6, 7};   // chat A: root + six turns, one over
        check(eviction_victim(stamps.data(), stamps.size(), 6) == 1,
              "over budget with fresh stamps: the oldest leaf leaves (the FIFO it was)");
    }
    {
        // chat B mounts through the root long after it was created; its stamp stays ancient
        const std::vector<uint64_t> stamps = {1, 100, 101, 102, 103, 104, 105};
        const size_t v = eviction_victim(stamps.data(), stamps.size(), 6);
        check(v == 1, "the root's stale stamp never makes it the victim");
        check(v != 0, "the root - the shared prefix - is never the victim");
    }
    {
        // the same chain after mounting checkpoint 1: its stamp jumps, the rotation moves to the next leaf
        const std::vector<uint64_t> stamped = {1, 8, 3, 4, 5, 6, 7};
        check(eviction_victim(stamped.data(), stamped.size(), 6) == 2,
              "after a mount, the rotation takes the next least recently used leaf");
    }
    {
        const std::vector<uint64_t> stamps = {1, 90, 91, 92};
        check(eviction_victim(stamps.data(), stamps.size(), 2) == 1,
              "a two-slot budget keeps root + newest and drops the leaf in between");
        check(eviction_victim(stamps.data(), stamps.size(), 1) == 0,
              "a one-slot budget has no pin: the oldest leaves, the newest point stays");
    }
    {
        const std::vector<uint64_t> stamps = {7, 3, 3, 5};
        const size_t a = eviction_victim(stamps.data(), stamps.size(), 4);
        const size_t b = eviction_victim(stamps.data(), stamps.size(), 4);
        check(a == 1 && b == 1, "ties among equally stale leaves: the earlier index, deterministically");
    }
    {
        // The tier order on a session switch: the disk record must STRICTLY beat the parked RAM image,
        // the live prefix and a slot session; an equal-length RAM image wins the tie (host memory, no
        // file read). This is the 7455 regression: `>=` let the disk win an equal-length tie.
        check(!disk_reuse_wins(true, 9000, 9000, 0, 0, false),
              "disk equal to the RAM image: the RAM image wins the tie (no file read)");
        check(disk_reuse_wins(true, 9001, 9000, 0, 0, false),
              "disk strictly longer than the RAM image: the disk record wins");
        check(!disk_reuse_wins(true, 8000, 9000, 0, 0, false),
              "disk shorter than the RAM image: the disk record loses");
        check(disk_reuse_wins(true, 9000, 0, 0, 0, false),
              "disk longer than the live prefix with no RAM image: the disk record wins");
        check(!disk_reuse_wins(true, 9000, 0, 9000, 0, false),
              "disk equal to the live prefix: the resident state is kept");
        check(!disk_reuse_wins(false, 0, 0, 0, 0, false), "no disk hit: never the disk tier");
    }
    {
        // A slot session is state already on the device; the disk must beat it strictly too.
        check(!disk_reuse_wins(true, 9000, 5000, 0, 9000, true),
              "disk equal to the slot session: the slot state is kept");
        check(disk_reuse_wins(true, 9001, 5000, 0, 9000, true),
              "disk strictly longer than the slot session: the disk record wins");
    }
    {
        // Taking the RAM image: only when it strictly beats both the live prefix and the slot session.
        check(ram_reuse_wins(9000, 0, 0), "a longer parked RAM image is taken over the live prefix");
        check(!ram_reuse_wins(9000, 9000, 0), "a RAM image no longer than the live prefix is not taken");
        check(!ram_reuse_wins(9000, 0, 9000), "a RAM image equal to the slot session is not taken");
        check(ram_reuse_wins(9001, 0, 9000), "a RAM image longer than the slot session is taken");
        check(!ram_reuse_wins(0, 0, 0), "no RAM match: nothing is taken");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
