// include/strata/core/conversation_disk.hpp - the L3 disk tier for parked conversations.
//
// The RAM cache (conversation_cache.hpp) holds whole conversations in host memory. It forgets
// them when the process exits. This store keeps the same images in files. A restarted server
// can then resume a conversation without a full prompt read.
//
// The store is CPU only. It makes no CUDA call. The caller captures and restores the images
// with conversation_snapshot.hpp.
//
// Each conversation is an append-only CHAIN: one file, a fixed master header followed by one
// PANEL per park.  A park writes only the delta since the chain's last panel - the ids/imgs it
// gained, new checkpoints, and per-layer K/V tails - so a growing conversation never rewrites
// its history.  A self-contained chain covers tokens [0 .. N); a chain may instead reference a
// shared system-prompt ('p') chain as its base, in which case its panels cover [root .. N) and
// the root's K/V is stored once in the 'p' file.  Reading merges the base and the panels back
// into full stage images, then hands them to the caller exactly like a one-record store.
//
// A torn append (crash mid-panel) is recovered on the next open: the panel checksum fails and
// the file is truncated to the last complete panel, so the conversation survives to its previous
// park - strictly better than a full-snapshot store, where a crash mid-write loses the record.
//
// Ownership and lifetime:
//  * The store owns its index and its files.
//  * seed() and append() consume the record. The store writes it and the caller must not reuse it.
//  * get() fills a record the caller owns. The record owns its buffers.
//  * A ConversationDiskMatch holds a record name. That name stays valid until the next mutation
//    or remove(). The caller must call get() before then.
//  * One thread uses one store. The store is synchronous and makes no lock.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace strata::core {

// The FNV-1a 64 checksum seed and mix. The store uses one checksum for the metadata block and
// one for the payload block.
inline constexpr uint64_t conversation_disk_checksum_seed = 14695981039346656037ull;

inline uint64_t conversation_disk_checksum(const uint8_t* data, size_t size,
                                           uint64_t seed = conversation_disk_checksum_seed) {
    uint64_t hash = seed;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

// A caller-supplied compatibility key. Two runs share a store only if the keys match.
// The key must cover the model weights and every runtime flag that changes the payload layout
// (quantization, rope scaling, layer split, cvec mode). The store compares the 32 bytes exactly.
struct ConversationDiskIdentity {
    static constexpr size_t size = 32;
    std::array<uint8_t, size> bytes{};
    bool operator==(const ConversationDiskIdentity&) const = default;
    // A stable label from a text key. This is a compatibility label, not a secure hash.
    static ConversationDiskIdentity from_string(std::string_view text);
};

// One conversation as a set of layer-split stage images.
struct ConversationDiskRecord {
    // One entry per stage, ordered by layer_lo. A single-GPU run uses one entry.
    std::vector<SavedConversation> stages;
    // The RAM size of the images. The caller uses it for admission.
    size_t bytes() const {
        size_t n = stages.capacity() * sizeof(SavedConversation);
        for (const auto& stage : stages) n += stage.bytes();
        return n;
    }
};

// A lookup result. name is the record to load. tokens is the matched prefix length.
struct ConversationDiskMatch {
    std::string name;
    int64_t tokens = 0;
};

enum class ConversationDiskStatus {
    ok,        // the call succeeded
    miss,      // no record has that name or prefix
    corrupt,   // the file failed validation and the store removed it
    invalid,   // the caller's record, name, or options are not usable
    disabled,  // the store has no byte budget
    failed,    // an I/O or allocation error; the caller may retry
};

struct ConversationDiskOptions {
    std::filesystem::path directory;   // the store root; open() creates it
    ConversationDiskIdentity identity; // the current run's compatibility key
    uint64_t budget_bytes = 0;         // the file budget; 0 disables the store
    size_t max_records = 0;            // the record cap; 0 means no cap
    uint64_t min_free_bytes = 0;       // the free-space floor the store keeps
};

// A safe record name from a token prefix. The store uses the same name in put().
std::string conversation_disk_name(const std::vector<int32_t>& ids,
                                   const std::vector<ConversationImageKey>& images, bool cvec);
// The name of a shared system-prompt prefix record: the same ids for every chat of a client, and the
// record class ('p') the LRU evicts only after every conversation record ('c').
std::string conversation_disk_prefix_name(const std::vector<int32_t>& ids,
                                          const std::vector<ConversationImageKey>& images, bool cvec);

class ConversationDiskStore {
public:
    static constexpr uint32_t format_version = 2;
    static constexpr const char* file_suffix = ".conversation";

    ConversationDiskStore() = default;
    ConversationDiskStore(const ConversationDiskStore&) = delete;
    ConversationDiskStore& operator=(const ConversationDiskStore&) = delete;
    ConversationDiskStore(ConversationDiskStore&&) = default;
    ConversationDiskStore& operator=(ConversationDiskStore&&) = default;

    // Opens the directory and rebuilds the index. Removes every file that fails validation,
    // including files from an older format and chains whose base record is gone.  A chain whose
    // last panel is torn is truncated to its last complete panel instead (the crash recovery).
    ConversationDiskStatus open(const ConversationDiskOptions& options, std::string& error);

    bool enabled() const { return options_.budget_bytes != 0; }

    // Finds the longest stored token prefix. It considers only chains with the same cvec flag.
    // A tie prefers the most recently used chain. This call does no I/O and no allocation.
    bool best(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
              bool cvec, ConversationDiskMatch& match) const;

    // Publishes a NEW chain: one self-contained panel covering [0 .. end) OR, when `base_name`
    // is non-empty, a panel covering [base_tokens .. end) whose earlier tokens the named 'p'
    // chain contributes (base_tokens == that chain's coverage; it is pinned while this chain
    // lives).  `name` must be the caller's stable chain key (the record name of the seed
    // content).  Replaces a chain with the same name.  Consumes the record in all cases.
    ConversationDiskStatus seed(const std::string& name, ConversationDiskRecord&& record,
                                std::string base_name, uint64_t base_tokens, std::string& error);

    // Appends one panel to an existing chain: the delta record covers [the chain's current
    // coverage .. its ids length later).  Returns miss when the chain is gone (the caller then
    // seeds it again).  Consumes the record in all cases except `invalid`.
    ConversationDiskStatus append(const std::string& name, ConversationDiskRecord&& record,
                                  std::string& error);

    // The token coverage of a chain (its latest panel's end), or false when it is not indexed.
    bool cover(const std::string& name, uint64_t& tokens) const;

    // Rewinds a chain to at most `at` tokens: drops every panel whose end exceeds `at`.
    // `new_cover` is the deepest kept panel's end (the caller then re-captures the delta from
    // there, or reseeds from the chain's base).  No-op when the chain already covers `at`.
    ConversationDiskStatus rewind_chain(const std::string& name, uint64_t at, uint64_t& new_cover,
                                        std::string& error);

    // Loads a chain into RAM: resolves the base, merges every panel, validates each panel's
    // checksums while it streams, and truncates a torn tail.  Removes the file on corruption.
    ConversationDiskStatus get(const std::string& name, ConversationDiskRecord& record, std::string& error);
    bool has(const std::string& name) const { return find(name) != nullptr; }

    ConversationDiskStatus remove(const std::string& name, std::string& error);

    size_t records() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    uint64_t evictions() const { return evictions_; }
    uint64_t corruptions() const { return corruptions_; }
    uint64_t recoveries() const { return recoveries_; }
    const std::filesystem::path& directory() const { return options_.directory; }

private:
    // The in-RAM index of one chain. The points hold ids and imgs only, never payload bytes.
    struct Entry {
        std::string name;
        uint64_t bytes = 0;
        uint64_t stamp = 0;
        uint64_t cover = 0;            ///< tokens the chain covers at its last write
        bool cvec = true;
        bool root = false;             ///< a system-prompt ('p') chain: evicted after conversations
        size_t refs = 0;               ///< chat chains whose base this root is (roots only)
        std::string base;              ///< the referenced 'p' chain's name, or empty
        std::vector<ConversationCheckpoint> points;
    };

    std::filesystem::path path_for(const std::string& name) const;
    static bool valid_name(const std::string& name);
    const Entry* find(const std::string& name) const;
    Entry* find(const std::string& name);
    size_t lru_victim(const std::string& skip) const;
    void evict(size_t index);
    ConversationDiskStatus make_room(uint64_t incoming, const std::string& skip, bool adds_record,
                                     std::string& error);
    ConversationDiskStatus index_file(const std::filesystem::path& path, const std::string& name,
                                      Entry& entry, std::string& error) const;
    // Truncates a torn last panel (a crash mid-append) back to the chain's previous coverage and
    // re-indexes the entry.  Returns false when the tail is intact or the file does not validate.
    bool truncate_torn_tail(const std::string& name);
    // The referenced 'p' chain's full ids/imgs from the index (its final match point).
    bool base_ids(const std::string& base_name, std::vector<int32_t>& ids,
                  std::vector<ConversationImageKey>& imgs, std::string& error) const;
    // Same, by reading the base chain's metadata directly (used at open(), when the index may not
    // be ordered yet; depth is limited to one root level).
    ConversationDiskStatus base_ids_from_file(const std::string& base_name, std::vector<int32_t>& ids,
                                              std::vector<ConversationImageKey>& imgs,
                                              std::string& error) const;
    // Loads one chain file into `out`.  A referenced 'p' base is resolved through the same path
    // (depth is limited to one root level).  Panels are merged in order; a torn tail is detected
    // by its checksum and dropped.
    ConversationDiskStatus read_chain(const std::filesystem::path& path, int depth,
                                      ConversationDiskRecord& out, std::string& error) const;
    static ConversationDiskStatus read_chain_impl(const std::filesystem::path& path,
                                                  const ConversationDiskStore& self, int depth,
                                                  ConversationDiskRecord& out, std::string& error);

    ConversationDiskOptions options_{};
    std::vector<Entry> entries_;
    uint64_t bytes_ = 0;
    uint64_t stamp_ = 0;
    uint64_t evictions_ = 0;
    uint64_t corruptions_ = 0;
    uint64_t recoveries_ = 0;
};

} // namespace strata::core
