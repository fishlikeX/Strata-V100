// src/core/conversation_disk_test.cpp - the L3 disk tier (append-only chains), CPU only. No CUDA.
//
// The test builds SavedConversation deltas by hand - the same shape the engine captures: sliced
// ids/imgs, the current running state, only the NEW checkpoints, and K/V TAILS from the first new
// page - writes them through the store, and checks the merged read-back, the prefix lookup, the
// base ('p' root) resolution, the torn-tail recovery, the rewind, the eviction ordering, the
// startup re-index, and every corruption path.
#include "strata/core/conversation_disk.hpp"
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

ConversationBuffer make_buffer(size_t bytes, uint8_t value) {
    ConversationBuffer buffer;
    buffer.resize(bytes);
    buffer.visit(0, bytes, [&](uint8_t* data, size_t chunk, size_t) {
        std::memset(data, value, chunk);
        return true;
    });
    return buffer;
}

int64_t aligned(int64_t cells, int64_t page) { return (cells + page - 1) / page * page; }

// A stage skeleton with one K/V layer; the caller sets the ids and the page-rounded cells.
SavedConversation make_stage(int64_t rate, uint8_t fill, bool cvec = true) {
    SavedConversation stage;
    stage.geometry[0] = 8;
    stage.geometry[1] = 1;
    stage.geometry[2] = 64;
    stage.layer_lo = 0;
    stage.layer_hi = 4;
    stage.cvec = cvec;
    stage.live.gdn.assign(8, fill);
    stage.live.ple.assign(5, static_cast<uint8_t>(fill + 1));
    stage.live.tails.assign(3, static_cast<uint8_t>(fill + 2));
    stage.live.dead.assign(2, static_cast<uint8_t>(fill + 3));
    stage.live.block_pos.assign(4, static_cast<uint8_t>(fill + 4));
    stage.kv.resize(1);
    ConversationKv& kv = stage.kv.front();
    kv.format = 0;
    kv.heads = 1;
    kv.head_dim = 64;
    kv.page_size = 8;
    kv.pooled_rows = 2;
    kv.idx_dim = 8;
    return stage;
}

// The full image of a conversation at `upto` tokens (the reference the merges must equal).

const std::vector<int32_t>& all_ids() {
    static const std::vector<int32_t> ids = [] {
        std::vector<int32_t> v;
        for (int32_t i = 11; v.size() < 40; i += 11) v.push_back(i);
        return v;
    }();
    return ids;
}
SavedConversation full_image(size_t upto, uint8_t fill = 77, int64_t rate = 4) {
    SavedConversation stage = make_stage(rate, fill);
    stage.live.ids.assign(all_ids().begin(), all_ids().begin() + (int64_t) upto);
    stage.kv.front().cells = aligned((int64_t) upto, 8);
    const int64_t cells = stage.kv.front().cells;
    stage.kv.front().k = make_buffer((size_t) (cells * rate), fill);
    stage.kv.front().v = make_buffer((size_t) (cells * rate), static_cast<uint8_t>(fill + 1));
    stage.kv.front().k_scale = make_buffer((size_t) (cells * 2), static_cast<uint8_t>(fill + 2));
    stage.kv.front().v_scale = make_buffer((size_t) (cells * 2), static_cast<uint8_t>(fill + 3));
    stage.kv.front().pooled = make_buffer((size_t) ((upto / 4 + 1) * 32), static_cast<uint8_t>(fill + 4));
    return stage;
}

// The engine-captured DELTA for the same stage: ids [first..end), the CURRENT run-state, only
// checkpoints newer than `first`, and K/V TAILS beginning at the page containing `first`.

// A byte-exact tail slice of a full buffer, starting at byte `offset`.
ConversationBuffer copy_slice(const ConversationBuffer& src, size_t offset, size_t count) {
    ConversationBuffer out;
    out.resize(count);
    if (count != 0) {
        std::vector<uint8_t> tmp(count);
        check(src.read(tmp.data(), offset, count), "a tail slice reads its source");
        size_t done = 0;
        out.visit(0, count, [&](uint8_t* p, size_t chunk, size_t) {
            std::memcpy(p, tmp.data() + done, chunk);
            done += chunk;
            return true;
        });
    }
    return out;
}
SavedConversation make_delta_stage(const SavedConversation& full, int64_t first, int64_t rate) {
    SavedConversation delta;
    delta.geometry = full.geometry;
    delta.layer_lo = full.layer_lo;
    delta.layer_hi = full.layer_hi;
    delta.cvec = full.cvec;
    delta.live.ids.assign(full.live.ids.begin() + first, full.live.ids.end());
    delta.live.gdn = full.live.gdn;
    delta.live.ple = full.live.ple;
    delta.live.tails = full.live.tails;
    delta.live.dead = full.live.dead;
    delta.live.block_pos = full.live.block_pos;
    delta.live.used = full.live.used;
    delta.kv.resize(1);
    const ConversationKv& kv = full.kv.front();
    const int64_t page_pad = (first / kv.page_size) * kv.page_size;
    ConversationKv& out = delta.kv.front();
    out.format = kv.format;
    out.cells = kv.cells;
    out.heads = kv.heads;
    out.head_dim = kv.head_dim;
    out.page_size = kv.page_size;
    out.pooled_rows = kv.pooled_rows;
    out.idx_dim = kv.idx_dim;
    out.k = copy_slice(kv.k, (size_t) page_pad * rate, kv.k.size() - (size_t) page_pad * rate);
    out.v = copy_slice(kv.v, (size_t) page_pad * rate, kv.v.size() - (size_t) page_pad * rate);
    out.k_scale = copy_slice(kv.k_scale, (size_t) page_pad * 2, kv.k_scale.size() - (size_t) page_pad * 2);
    out.v_scale = copy_slice(kv.v_scale, (size_t) page_pad * 2, kv.v_scale.size() - (size_t) page_pad * 2);
    out.pooled = copy_slice(kv.pooled, (size_t) ((first / 4) * 32), kv.pooled.size() - (size_t) ((first / 4) * 32));
    out.first_units = {page_pad, page_pad, page_pad, page_pad, first / 4};
    return delta;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails && a.dead == b.dead &&
           a.block_pos == b.block_pos && a.used == b.used && a.stage_parts.size() == b.stage_parts.size();
}

bool same_kv(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads && a.head_dim == b.head_dim &&
           a.page_size == b.page_size && a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim && a.k == b.k &&
           a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}

bool same_stage(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec)
        return false;
    if (!same_checkpoint(a.live, b.live) || !same_kv(a.kv.front(), b.kv.front())) return false;
    return true;
}

bool same_record(const ConversationDiskRecord& a, const ConversationDiskRecord& b) {
    if (a.stages.size() != b.stages.size()) return false;
    for (size_t i = 0; i < a.stages.size(); ++i)
        if (!same_stage(a.stages[i], b.stages[i])) return false;
    return true;
}

ConversationDiskOptions options_for(const std::filesystem::path& directory,
                                    const ConversationDiskIdentity& identity, uint64_t budget) {
    ConversationDiskOptions options;
    options.directory = directory;
    options.identity = identity;
    options.budget_bytes = budget;
    return options;
}

void patch_byte(const std::filesystem::path& path, uint64_t offset, uint8_t value) {
    std::FILE* file = std::fopen(path.string().c_str(), "r+b");
    check(file != nullptr, "a patched record opens");
    check(std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0, "a patched record seeks");
    check(std::fwrite(&value, 1, 1, file) == 1, "a patched record writes");
    std::fclose(file);
}

std::filesystem::path scratch_dir(const char* tag) {
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                ("strata-disk-" + std::string(tag) + "-" +
                                 std::to_string((unsigned long) getpid()));
    std::filesystem::remove_all(dir);
    return dir;
}

void test_basic_chain() {
    std::filesystem::path dir = scratch_dir("chain");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");

    // Seed [0..12), append [12..24), append [24..32).  The merged read-back must equal the full
    // image at 32 tokens.
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "a seed parks");
    uint64_t cover = 0;
    check(store.cover(name, cover) && cover == 12, "the seed covers 12 tokens");
    check(store.records() == 1, "one chain after the seed");

    ConversationDiskRecord merged;
    check(store.get(name, merged, error) == ConversationDiskStatus::ok, "the seed reads back");
    check(merged.stages[0].live.ids.size() == 12, "the seed merges to 12 ids");

    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "an append parks");
    ConversationDiskRecord d24;
    d24.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(name, std::move(d24), error) == ConversationDiskStatus::ok, "the second append parks");
    check(store.cover(name, cover) && cover == 32, "the chain covers 32 tokens");
    check(store.records() == 1, "still one chain after appends");

    check(store.get(name, merged, error) == ConversationDiskStatus::ok, "the chain reads back");
    check(merged.stages[0].live.ids ==
              std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32),
          "the merged ids are the full chain");
    check(same_record(merged, [&] {
        ConversationDiskRecord full;
        full.stages.push_back(full_image(32));
        return full;
    }()), "the merged image equals the full capture");

    // best() returns the longest stored prefix.
    ConversationDiskRecord short_seed;
    short_seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    const std::string short_name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 12), {}, true);
    check(store.seed(short_name, std::move(short_seed), "", 0, error) == ConversationDiskStatus::ok,
          "a second seed");
    ConversationDiskMatch m;
    check(store.best(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 20), {}, true, m), "best finds a prefix");
    check(m.tokens == 12 && (m.name == short_name || m.name == name), "best returns a 12-token chain");

    check(store.remove(name, error) == ConversationDiskStatus::ok, "the chain removes");
    check(store.remove(short_name, error) == ConversationDiskStatus::ok, "the second chain removes");
    check(store.records() == 0, "the store is empty after removal");
    std::filesystem::remove_all(dir);
}

void test_root_base() {
    std::filesystem::path dir = scratch_dir("root");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");

    // The shared system prompt: a self-contained 'p' chain at 8 tokens.
    const std::string pname = conversation_disk_prefix_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 8), {}, true);
    ConversationDiskRecord root_rec;
    root_rec.stages.push_back(make_delta_stage(full_image(8), 0, 4));
    check(store.seed(pname, std::move(root_rec), "", 0, error) == ConversationDiskStatus::ok, "the root parks");

    // Two chats sharing the root: each seeds from the root's coverage and appends its own tail.
    const std::string cname1 = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 16), {}, true);
    const std::string cname2 = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 12), {}, true);
    ConversationDiskRecord seed1;
    seed1.stages.push_back(make_delta_stage(full_image(16), 8, 4));
    ConversationDiskRecord seed2;
    seed2.stages.push_back(make_delta_stage(full_image(12), 8, 4));
    check(store.seed(cname1, std::move(seed1), pname, 8, error) == ConversationDiskStatus::ok,
          "a chat seeds on the root base");
    check(store.seed(cname2, std::move(seed2), pname, 8, error) == ConversationDiskStatus::ok,
          "the second chat seeds");

    // (At these small scales the per-chat delta and the root's snapshot are comparable in bytes -
    // the real dedup signal is the content below: a root-backed chat must MERGE to its full image
    // without ever holding the root's K/V, and the eviction test pins the referenced root.)

    ConversationDiskRecord back;
    check(store.get(cname1, back, error) == ConversationDiskStatus::ok, "a root-backed chat reads back");
    check(back.stages[0].live.ids.size() == 16, "the merged chat has the full ids");
    check(same_record(back, [&] {
        ConversationDiskRecord full;
        full.stages.push_back(full_image(16));
        return full;
    }()), "the merged chat equals its full capture");

    // A 9-token request matches the root at 8 (a record may never cover the whole prompt).
    ConversationDiskMatch m;
    check(store.best(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 9), {}, true, m),
          "a 9-token request matches a stored prefix");
    check(m.tokens == 8 && m.name == pname, "the root wins the equal-length tie");
    check(store.records() == 3, "root + two chats");
    std::filesystem::remove_all(dir);
}

void test_torn_tail() {
    std::filesystem::path dir = scratch_dir("torn");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "the seed parks");
    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "the first append");
    ConversationDiskRecord d24;
    d24.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(name, std::move(d24), error) == ConversationDiskStatus::ok, "the second append");

    // Tear the LAST panel's payload (one byte in its middle) and re-read: the store truncates
    // back to the last complete panel and serves the chain at 24 tokens.
    const std::filesystem::path file = dir / (name + ".conversation");
    std::FILE* f = std::fopen(file.string().c_str(), "rb");
    check(f != nullptr, "the chain file opens");
    uint64_t pos = 128;
    uint64_t payload_at = 0;
    uint64_t meta = 0, payload = 0;
    for (int k = 0; k < 3; ++k) {
        uint64_t magic, base, end, mck, pck;
        check(std::fseek(f, (long) pos, SEEK_SET) == 0, "the scan seeks to a panel");
        check(std::fread(&magic, 8, 1, f) == 1 && std::fread(&base, 8, 1, f) == 1 &&
                  std::fread(&end, 8, 1, f) == 1 && std::fread(&meta, 8, 1, f) == 1 &&
                  std::fread(&mck, 8, 1, f) == 1 && std::fread(&payload, 8, 1, f) == 1 &&
                  std::fread(&pck, 8, 1, f) == 1,
              "panel headers read");
        pos += 56 + meta + payload;
        if (k == 2) payload_at = pos - payload;   // the last panel's payload start
    }
    std::fclose(f);
    patch_byte(file, payload_at + 100, 0x5A);
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "a torn tail recovers on read");
    check(back.stages[0].live.ids ==
              std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 24),
          "the recovered ids are the pre-tear chain");
    uint64_t cover = 0;
    check(store.cover(name, cover) && cover == 24, "the coverage reverts to the last good panel");
    store.remove(name, error);
    std::filesystem::remove_all(dir);
}

void test_rewind() {
    std::filesystem::path dir = scratch_dir("rewind");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "the seed parks");
    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "the first append");
    ConversationDiskRecord d24;
    d24.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(name, std::move(d24), error) == ConversationDiskStatus::ok, "the second append");

    uint64_t cover = 0;
    check(store.rewind_chain(name, 15, cover, error) == ConversationDiskStatus::ok && cover == 12,
          "a rewind to 15 truncates at the last panel end <= 15 (12)");
    check(store.cover(name, cover) && cover == 12, "the chain covers 12 after the rewind");

    // The editor rewrote the tail: a fresh lineage from 12 onward (a different run-state).
    SavedConversation redo = full_image(20, 90);
    redo.live.used = 7;
    ConversationDiskRecord redo_rec;
    redo_rec.stages.push_back(make_delta_stage(redo, 12, 4));
    check(store.append(name, std::move(redo_rec), error) == ConversationDiskStatus::ok, "the rewrite appends");
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "the rewound chain reads back");
    check(back.stages[0].live.ids ==
              std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 20),
          "the rewound chain holds [0..20)");
    check(back.stages[0].live.used == 7, "the merged live state is the rewrite's");
    check(back.stages[0].kv.front().k.size() == (size_t) (aligned(20, 8) * 4),
          "the merged K/V is back to the full extent");
    store.remove(name, error);
    std::filesystem::remove_all(dir);
}

void test_eviction_order() {
    std::filesystem::path dir = scratch_dir("evict");
    std::string error;
    // Phase 1: root + two root-backed chats, in a generous store.
    uint64_t budget = 1ull << 30;
    ConversationDiskStore store;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), budget), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");
    const std::string pname = conversation_disk_prefix_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 8), {}, true);
    ConversationDiskRecord root_rec;
    root_rec.stages.push_back(make_delta_stage(full_image(8), 0, 4));
    check(store.seed(pname, std::move(root_rec), "", 0, error) == ConversationDiskStatus::ok, "the root parks");
    const uint64_t root_bytes = std::filesystem::file_size(dir / (pname + ".conversation"));

    auto chat_seed = [&](size_t upto, const char* suffix, std::string& cname, uint8_t fill) {
        SavedConversation full = full_image(upto, fill);
        ConversationDiskRecord rec;
        rec.stages.push_back(make_delta_stage(full, 8, 4));
        cname = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + (int64_t) upto),
                                       {}, true) + std::string(suffix);
        check(store.seed(cname, std::move(rec), pname, 8, error) == ConversationDiskStatus::ok,
              "a root-backed chat seeds");
    };
    std::string c1, c2;
    chat_seed(12, "-a", c1, 80);
    chat_seed(12, "-b", c2, 81);
    const uint64_t chat_bytes = std::filesystem::file_size(dir / (c1 + ".conversation"));
    check(store.records() == 3, "root + two chats fit");

    // Phase 2: reopen the same directory with a budget that fits the root + ONE chat + the
    // forcing self-contained chat.  Seeding the force then evicts the OLDEST unpinned chat;
    // the referenced root stays.
    ConversationDiskStore tight;
    const uint64_t tight_budget = root_bytes + chat_bytes * 2 + 1;
    check(tight.open(options_for(dir, ConversationDiskIdentity::from_string("test"), tight_budget), error) ==
              ConversationDiskStatus::ok,
          "a snug store reopens");
    check(tight.records() == 3, "the snug store re-indexed everything");
    SavedConversation other = full_image(6, 82);
    ConversationDiskRecord other_rec;
    other_rec.stages.push_back(make_delta_stage(other, 0, 4));
    std::string oname = "c-other-" + std::to_string((unsigned long) getpid());
    check(tight.seed(oname, std::move(other_rec), "", 0, error) == ConversationDiskStatus::ok,
          "the forcing chat seeds");
    check(tight.has(oname), "the forcing chat is in");
    check(!tight.has(c1), "the oldest unpinned chat was evicted to make room");
    check(tight.has(pname), "the referenced root is still pinned");
    std::filesystem::remove_all(dir);
}

void test_reopen_and_purge() {
    std::filesystem::path dir = scratch_dir("reopen");
    std::string error;
    {
        ConversationDiskStore store;
        check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
                  ConversationDiskStatus::ok,
              "a fresh store opens");
        const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
        ConversationDiskRecord seed;
        seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
        check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "the seed parks");
        ConversationDiskRecord d12;
        d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
        check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "an append parks");
    }
    // An invalid old-format file is purged at open.
    std::filesystem::path garbage = dir / ("oldbad.conversation");
    {
        std::FILE* f = std::fopen(garbage.string().c_str(), "wb");
        check(f != nullptr, "the garbage file opens");
        for (int i = 0; i < 64; ++i) std::fwrite("garbage!", 1, 8, f);
        std::fclose(f);
    }
    ConversationDiskStore store;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the store reopens the directory");
    check(store.records() == 1, "one valid chain after reopen");
    check(!std::filesystem::exists(garbage), "the garbage file was purged");
    for (auto& p : std::filesystem::directory_iterator(dir)) {
        std::string name = p.path().filename().string();
        name = name.substr(0, name.size() - std::strlen(ConversationDiskStore::file_suffix));
        ConversationDiskRecord back;
        check(store.get(name, back, error) == ConversationDiskStatus::ok, "the reopened chain reads back");
        check(back.stages[0].live.ids.size() == 24, "the reopened chain keeps both panels");
    }
    std::filesystem::remove_all(dir);
}

void test_two_stages() {
    std::filesystem::path dir = scratch_dir("stages");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");
    // A layer split: two stages with separate carves and run-states.
    auto make2 = [&](size_t upto, int64_t first, uint8_t fill) {
        ConversationDiskRecord rec;
        SavedConversation full0 = full_image(upto, fill);
        full0.layer_lo = 0;
        full0.layer_hi = 20;
        SavedConversation full1 = full_image(upto, static_cast<uint8_t>(fill + 10));
        full1.layer_lo = 20;
        full1.layer_hi = 40;
        rec.stages.push_back(make_delta_stage(full0, first, 4));
        rec.stages.push_back(make_delta_stage(full1, first, 4));
        return rec;
    };
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 24), {}, true);
    check(store.seed(name, make2(12, 0, 60), "", 0, error) == ConversationDiskStatus::ok,
          "a two-stage seed parks");
    check(store.append(name, make2(24, 12, 61), error) == ConversationDiskStatus::ok, "a two-stage append parks");
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "a two-stage chain reads back");
    check(back.stages.size() == 2, "both stages merge");
    check(back.stages[0].live.ids == back.stages[1].live.ids, "the stages share the token chain");
    check(back.stages[0].live.ids.size() == 24, "each stage has the full ids");
    store.remove(name, error);
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    test_basic_chain();
    test_root_base();
    test_torn_tail();
    test_rewind();
    test_eviction_order();
    test_reopen_and_purge();
    test_two_stages();
    std::fprintf(stderr, "conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
