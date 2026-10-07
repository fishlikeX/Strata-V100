// include/strata/program/conv_cache.hpp - the conversation cache's retention policy: which checkpoint
// leaves when the cache is over its slot budget.
//
// The cache holds conversation checkpoints: snapshots of the running state (the GDN recurrence, the PLE
// history, the QSA indexer tails - `ConvCheckpoint` in generate.cpp) that a request resumes its prompt
// from instead of reading those tokens again.  The request path keeps only the checkpoints whose tokens
// are a prefix of the current prompt - any other checkpoint's positional cells have been overwritten,
// because the KV cache is one arena that holds one branch of history at a time - so at any moment the
// retained checkpoints are a CHAIN: sorted by length, each a prefix of the next.  That is a radix cache's
// tree collapsed onto the one branch of history the session can hold.
//
// The chain's root is therefore the deepest point every request so far has shared - in practice the end
// of the system prompt, which every new chat of the same client mounts through - and it is exactly the
// node a radix cache keeps alive while its leaves rotate.  First-in-first-out kept it only by accident of
// being first: after `prompt_cache` newer checkpoints it was gone, and the next chat read the whole
// prefix again at prefill speed.  So:
//
//   * the root is PINNED: dropping it frees one slot's ~118 MB and costs every future conversation that
//     shares the prefix a full re-read of it (a 30K system prompt is ~30 s at ~1,000 tok/s);
//   * every other slot rotates by least recent use.  A checkpoint's stamp advances when it is created,
//     when a new checkpoint lands on its length, and when a request mounts through it.
//
// With a budget below two slots there is no room to keep root and leaf apart, so the pin switches off and
// the oldest checkpoint leaves - the previous behaviour, one slot = the newest point only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::program::conv_cache {

/// The index in `stamps` of the chain item to drop once the chain holds more than `cap` items.  `stamps`
/// are the items' last-use stamps; the caller owns the chain and erases the returned index.  Pure and
/// deterministic so conv_cache_test.cpp can walk the scenarios by hand.
inline size_t eviction_victim(const uint64_t* stamps, size_t n, int64_t cap) {
    if (cap < 2 || n < 2) return 0;   // no room for root and leaf: the pin is off, the oldest leaves
    size_t v = 1;                     // the root (0) is pinned; least recent use among the rest
    for (size_t i = 2; i < n; ++i)
        if (stamps[i] < stamps[v]) v = i;
    return v;
}

/// Which parked tier a request resumes from, given the longest compatible prefix each tier holds.  A
/// longer prefix always wins; on an equal prefix the resident RAM image wins over the disk record: it
/// restores from host memory without a file read (~0.58 s against the disk's ~2.08 s), and the equal
/// disk record holds nothing the RAM image does not.  `disk_hit` is the store's own match flag and
/// `disk_tokens` its matched length; `slot_available` is a slot session that starts this prompt.
inline bool disk_reuse_wins(bool disk_hit, int64_t disk_tokens, int64_t ram_tokens, int64_t resume,
                            int64_t slot_tokens, bool slot_available) {
    return disk_hit && disk_tokens > resume && disk_tokens > ram_tokens &&
           (!slot_available || disk_tokens > slot_tokens);
}

/// Whether the parked RAM image is worth taking over the state already on the device: it must be
/// strictly longer than both the live prefix (`resume`) and a slot session that starts this prompt.
inline bool ram_reuse_wins(int64_t ram_tokens, int64_t resume, int64_t slot_tokens) {
    return ram_tokens > (resume > slot_tokens ? resume : slot_tokens);
}

}  // namespace strata::program::conv_cache
