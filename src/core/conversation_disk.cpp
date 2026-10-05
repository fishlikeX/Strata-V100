// src/core/conversation_disk.cpp - the L3 disk tier for parked conversations (append-only chains).
//
// One conversation = one file, a fixed 128-byte master header followed by one PANEL per park.
// A park writes only the delta since the chain's last panel - the token ids/imgs it gained, new
// checkpoints, the current running state, and per-layer K/V tails - so a growing conversation
// never rewrites its own history.  A chain may reference a shared system-prompt ('p') chain as
// its base, storing the root's K/V once instead of inside every chat.
//
// File layout, from offset zero:
//   chain header   a fixed 128-byte block: magic, version, coverage, panel count, identity, cvec,
//                  base reference (kind/tokens/name)
//   panel 1        header + metadata + payload
//   panel 2        header + metadata + payload
//   ...            (appended at EOF; the chain header is patched in place)
//
// Panel header   a fixed 56-byte block: magic, base/end tokens, metadata size + checksum,
//                payload size + checksum.
// Panel metadata the delta's shape: every scalar, every id and image key, and every section length.
// Panel payload  the delta's variable-length arrays back to back, in the metadata's section order.
//
// The seed (first panel) writes to a temporary file and renames it over the target, exactly like
// the old full-snapshot records.  Appends write in place at EOF: a crash mid-append leaves a torn
// tail whose payload checksum fails on get(), and the store truncates back to the last complete
// panel - the conversation survives to its previous park.
//
// The store is CPU only.  It makes no CUDA call; the caller captures and restores delta images
// with conversation_snapshot.hpp.  Reading merges the base and every panel back into full stage
// images whose K/V buffers are pre-sized and streamed into place, so the engine's restore path
// never changes (peak host RAM is one merged conversation plus the small base record).
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>
#include "strata/kernels/qsa.hpp"

namespace strata::core {
namespace {

// "STRCDISK", little-endian.
constexpr uint64_t kMagic = 0x4B53494443525453ull;
// "PANELKDS": the panel block magic.
constexpr uint64_t kPanelMagic = 0x53444B4C454E4150ull;
constexpr uint32_t kChainHeaderBytes = 128;
constexpr uint32_t kPanelHeaderBytes = 56;
constexpr uint64_t kMaxMetadataBytes = 64ull << 20;
constexpr uint32_t kMaxStages = 1024;
constexpr uint32_t kMaxCheckpoints = 1u << 20;
constexpr uint32_t kMaxParts = 64;
constexpr uint32_t kMaxPartDepth = 4;
constexpr uint32_t kMaxKv = 1u << 16;
constexpr uint64_t kMaxPanels = 1u << 24;

// ---- K/V layout units.  The final extent of each section is measured in units: page-capped
// cells for the four K/V buffers, pooled indexer ROWS (idx_block-aligned, with the moving spare
// row) for the pooled buffer.  Each panel's kv metadata carries its own tail's START unit per
// section (`ConversationKv::first_units`), so the merge places every tail at that offset - later
// tails overwrite the previous content's partial page/row.
int64_t kv_end_units(int64_t end, int64_t page, int e) {
    if (e == 4) return end / strata::kernels::qsa_real_shapes().idx_block + 1;
    return (end + page - 1) / page * page;
}

// The units a tail covers: from its stored start unit to the panel's end extent.
int64_t kv_units(int64_t first, int64_t end, int64_t page, int e) {
    return kv_end_units(end, page, e) - first;
}
constexpr size_t kMaxNameBytes = 96;

struct ChainHeader {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t reserved = 0;
    uint64_t total_tokens = 0;              // the chain's coverage after its last panel
    uint64_t panel_count = 0;
    uint8_t identity[ConversationDiskIdentity::size] = {};
    uint8_t cvec = 0;
    uint8_t base_kind = 0;                  // 0: none; 1: the named 'p' chain contributes [0..base_tokens)
    uint8_t padding0[6] = {};
    uint64_t base_tokens = 0;
    char base_name[40] = {};
    uint8_t padding1[8] = {};
};
static_assert(sizeof(ChainHeader) == kChainHeaderBytes, "the chain header is a fixed 128 bytes");

struct PanelHeader {
    uint64_t magic = 0;
    uint64_t base_tokens = 0;               // the chain's coverage BEFORE this panel
    uint64_t end_tokens = 0;                // the chain's coverage AFTER this panel
    uint64_t metadata_bytes = 0;
    uint64_t metadata_checksum = 0;
    uint64_t payload_bytes = 0;
    uint64_t payload_checksum = 0;
};
static_assert(sizeof(PanelHeader) == kPanelHeaderBytes, "the panel header is a fixed 56 bytes");

bool add_overflow(uint64_t& total, uint64_t extra) {
    if (extra > UINT64_MAX - total) return false;
    total += extra;
    return true;
}

bool read_exact(std::FILE* file, void* destination, uint64_t bytes) {
    uint8_t* out = static_cast<uint8_t*>(destination);
    while (bytes != 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, 1u << 20));
        if (std::fread(out, 1, chunk, file) != chunk) return false;
        out += chunk;
        bytes -= chunk;
    }
    return true;
}

// One variable-length payload section: either a plain byte vector or a segmented buffer.
struct Section {
    std::vector<uint8_t>* bytes = nullptr;
    ConversationBuffer* buffer = nullptr;
};

template<class Fn>
void visit_checkpoint(ConversationCheckpoint& checkpoint, Fn& fn) {
    fn(Section{&checkpoint.gdn, nullptr});
    fn(Section{&checkpoint.ple, nullptr});
    fn(Section{&checkpoint.tails, nullptr});
    fn(Section{&checkpoint.dead, nullptr});
    fn(Section{&checkpoint.block_pos, nullptr});
    for (auto& part : checkpoint.stage_parts) visit_checkpoint(part, fn);
}

// Visits every payload section in file order. The metadata writer and reader use the same order.
template<class Fn>
void visit_payload(ConversationDiskRecord& record, Fn& fn) {
    for (auto& stage : record.stages) {
        visit_checkpoint(stage.live, fn);
        for (auto& checkpoint : stage.checkpoints) visit_checkpoint(checkpoint, fn);
        for (auto& kv : stage.kv) {
            fn(Section{nullptr, &kv.k});
            fn(Section{nullptr, &kv.v});
            fn(Section{nullptr, &kv.k_scale});
            fn(Section{nullptr, &kv.v_scale});
            fn(Section{nullptr, &kv.pooled});
        }
    }
}

uint64_t count_sections(ConversationDiskRecord& record) {
    uint64_t count = 0;
    auto fn = [&](Section) { ++count; };
    visit_payload(record, fn);
    return count;
}

// Writes the record's shape and collects the section lengths in payload order.
struct MetaWriter {
    std::vector<uint8_t> data;
    std::vector<uint64_t> lengths;

    void raw(const void* source, size_t bytes) {
        const uint8_t* first = static_cast<const uint8_t*>(source);
        data.insert(data.end(), first, first + bytes);
    }
    void u8(uint8_t value) { data.push_back(value); }
    void u32(uint32_t value) { raw(&value, sizeof(value)); }
    void u64(uint64_t value) { raw(&value, sizeof(value)); }
    void i32(int32_t value) { raw(&value, sizeof(value)); }
    void i64(int64_t value) { raw(&value, sizeof(value)); }
    void section(uint64_t bytes) { u64(bytes); lengths.push_back(bytes); }

    void write_checkpoint(const ConversationCheckpoint& value) {
        u32(static_cast<uint32_t>(value.ids.size()));
        if (!value.ids.empty()) raw(value.ids.data(), value.ids.size() * sizeof(int32_t));
        u32(static_cast<uint32_t>(value.imgs.size()));
        for (const auto& key : value.imgs) {
            i64(key.start);
            u64(key.hash);
        }
        u64(value.used);
        section(value.gdn.size());
        section(value.ple.size());
        section(value.tails.size());
        section(value.dead.size());
        section(value.block_pos.size());
        u32(static_cast<uint32_t>(value.stage_parts.size()));
        for (const auto& part : value.stage_parts) write_checkpoint(part);
    }

    void write_kv(const ConversationKv& layer) {
        i32(layer.format);
        i64(layer.cells);
        i64(layer.heads);
        i64(layer.head_dim);
        i64(layer.page_size);
        i64(layer.pooled_rows);
        i64(layer.idx_dim);
        for (int64_t first : layer.first_units) i64(first);
        section(layer.k.size());
        section(layer.v.size());
        section(layer.k_scale.size());
        section(layer.v_scale.size());
        section(layer.pooled.size());
    }

    void write_stage(const SavedConversation& image) {
        i64(image.layer_lo);
        i64(image.layer_hi);
        for (int64_t value : image.geometry) i64(value);
        u8(image.cvec ? 1 : 0);
        write_checkpoint(image.live);
        u32(static_cast<uint32_t>(image.checkpoints.size()));
        for (const auto& checkpoint : image.checkpoints) write_checkpoint(checkpoint);
        u32(static_cast<uint32_t>(image.kv.size()));
        for (const auto& layer : image.kv) write_kv(layer);
    }
};

// Reads the record's shape and collects the section lengths in payload order. It allocates only
// ids and image keys, never a payload buffer.
struct MetaReader {
    const uint8_t* cursor = nullptr;
    const uint8_t* end = nullptr;
    bool ok = true;
    std::vector<uint64_t> lengths;

    size_t remaining() const { return ok ? static_cast<size_t>(end - cursor) : 0; }
    bool need(size_t bytes) {
        if (!ok || remaining() < bytes) {
            ok = false;
            return false;
        }
        return true;
    }
    uint8_t u8() { return need(1) ? *cursor++ : 0; }
    uint32_t u32() {
        uint32_t value = 0;
        if (need(4)) { std::memcpy(&value, cursor, 4); cursor += 4; }
        return value;
    }
    uint64_t u64() {
        uint64_t value = 0;
        if (need(8)) { std::memcpy(&value, cursor, 8); cursor += 8; }
        return value;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    int64_t i64() { return static_cast<int64_t>(u64()); }
    bool section(uint64_t& bytes) {
        bytes = u64();
        if (!ok) return false;
        lengths.push_back(bytes);
        return true;
    }

    bool read_checkpoint(ConversationCheckpoint& out, uint32_t depth) {
        if (depth > kMaxPartDepth) { ok = false; return false; }
        const uint32_t ids = u32();
        if (!ok || ids > remaining() / sizeof(int32_t)) { ok = false; return false; }
        out.ids.resize(ids);
        if (ids != 0) {
            std::memcpy(out.ids.data(), cursor, ids * sizeof(int32_t));
            cursor += ids * sizeof(int32_t);
        }
        const uint32_t imgs = u32();
        if (!ok || imgs > remaining() / 16) { ok = false; return false; }
        out.imgs.resize(imgs);
        for (uint32_t i = 0; i < imgs; ++i) {
            out.imgs[i].start = i64();
            out.imgs[i].hash = u64();
        }
        out.used = u64();
        uint64_t bytes = 0;
        for (int i = 0; i < 5; ++i)
            if (!section(bytes)) return false;
        const uint32_t parts = u32();
        if (!ok || parts > kMaxParts || parts > remaining() / 8) { ok = false; return false; }
        out.stage_parts.resize(parts);
        for (uint32_t i = 0; i < parts; ++i)
            if (!read_checkpoint(out.stage_parts[i], depth + 1)) return false;
        return ok;
    }

    bool read_kv(ConversationKv& out) {
        out.format = i32();
        out.cells = i64();
        out.heads = i64();
        out.head_dim = i64();
        out.page_size = i64();
        out.pooled_rows = i64();
        out.idx_dim = i64();
        for (int i = 0; i < 5; ++i) out.first_units[(size_t) i] = i64();
        uint64_t bytes = 0;
        for (int i = 0; i < 5; ++i)
            if (!section(bytes)) return false;
        return ok;
    }

    bool read_stage(SavedConversation& out) {
        out.layer_lo = i64();
        out.layer_hi = i64();
        for (int i = 0; i < 18; ++i) out.geometry[i] = i64();
        out.cvec = u8() != 0;
        if (!read_checkpoint(out.live, 0)) return false;
        const uint32_t checkpoints = u32();
        if (!ok || checkpoints > kMaxCheckpoints || checkpoints > remaining() / 24) { ok = false; return false; }
        out.checkpoints.resize(checkpoints);
        for (uint32_t i = 0; i < checkpoints; ++i)
            if (!read_checkpoint(out.checkpoints[i], 0)) return false;
        const uint32_t kv = u32();
        if (!ok || kv > kMaxKv || kv > remaining() / 40) { ok = false; return false; }
        out.kv.resize(kv);
        for (uint32_t i = 0; i < kv; ++i)
            if (!read_kv(out.kv[i])) return false;
        return ok;
    }

    bool parse(ConversationDiskRecord& out) {
        const uint32_t stages = u32();
        if (!ok || stages == 0 || stages > kMaxStages) { ok = false; return false; }
        out.stages.resize(stages);
        for (uint32_t i = 0; i < stages; ++i)
            if (!read_stage(out.stages[i])) return false;
        return ok;
    }
};

std::string hex64(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

// Validates the parsed metadata against the payload size. No allocation.
bool valid_lengths(ConversationDiskRecord& record, const std::vector<uint64_t>& lengths,
                   uint64_t payload_bytes) {
    uint64_t total = 0;
    for (const uint64_t bytes : lengths) {
        if (total > payload_bytes || bytes > payload_bytes - total) return false;
        total += bytes;
    }
    if (total != payload_bytes) return false;
    return count_sections(record) == lengths.size();
}

// One harness parsed from the metadata: shapes/ids/checkpoints/lengths, no payload bytes.
struct ParsedPanel {
    uint64_t base_tokens = 0;
    uint64_t end_tokens = 0;
    uint64_t payload_bytes = 0;
    uint64_t payload_checksum = 0;
    uint64_t payload_at = 0;                // file offset of the panel's payload
    ConversationDiskRecord delta;
    std::vector<uint64_t> lengths;
};

// The 5 running-state byte vectors of a checkpoint, in payload order.
std::array<std::vector<uint8_t>*, 5> live_arrays(ConversationCheckpoint& c) {
    return {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
}

// The K/V section lengths of a parsed panel, in payload order: [stage][layer] = the 5 buffer sizes.
// (The meta parse records the section lengths in `lengths` but never sizes the buffers themselves.)
std::vector<std::vector<std::array<size_t, 5>>> panel_kv_lens(const ConversationDiskRecord& delta,
                                                              const std::vector<uint64_t>& lengths) {
    std::vector<std::vector<std::array<size_t, 5>>> out(delta.stages.size());
    size_t i = 0;
    for (size_t si = 0; si < delta.stages.size(); ++si) {
        const SavedConversation& stage = delta.stages[si];
        i += 5;                                  // the live checkpoint's run-state
        i += 5 * stage.checkpoints.size();
        out[si].resize(stage.kv.size());
        for (size_t m = 0; m < stage.kv.size(); ++m)
            for (int e = 0; e < 5; ++e) out[si][m][(size_t) e] = (size_t) lengths[i++];
    }
    return out;
}

// The 5 K/V buffers of a layer, in payload order.
std::array<ConversationBuffer*, 5> kv_arrays(ConversationKv& kv) {
    return {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
}

// Reads and validates the chain header and every panel header + metadata (skipping payloads).
// Fills `panels` and leaves the file positioned at the end of the chain.
bool read_chain_meta(std::FILE* file, const ConversationDiskIdentity& identity,
                     uint64_t budget_bytes, ChainHeader& header, std::vector<ParsedPanel>& panels,
                     std::string& error) {
    auto bad = [&](const char* why) {
        error = std::string("corrupt conversation record: ") + why;
        return false;
    };
    if (!read_exact(file, &header, sizeof(header))) return bad("header");
    if (header.magic != kMagic || header.version != ConversationDiskStore::format_version ||
        std::memcmp(header.identity, identity.bytes.data(), ConversationDiskIdentity::size) != 0 ||
        header.base_kind > 1 || header.panel_count > kMaxPanels)
        return bad("header");
    uint64_t expected = header.base_kind != 0 ? header.base_tokens : 0;
    if (header.total_tokens < expected) return bad("lengths");
    panels.clear();
    panels.reserve(static_cast<size_t>(header.panel_count));
    uint64_t payload_at = kChainHeaderBytes;
    for (uint64_t k = 0; k < header.panel_count; ++k) {
        PanelHeader p{};
        if (!read_exact(file, &p, sizeof(p))) return bad("panel header");
        if (p.magic != kPanelMagic || p.base_tokens != expected || p.end_tokens < p.base_tokens)
            return bad("panel header");
        if (p.metadata_bytes > kMaxMetadataBytes || p.payload_bytes > budget_bytes)
            return bad("panel sizes");
        std::vector<uint8_t> meta(static_cast<size_t>(p.metadata_bytes));
        if (!read_exact(file, meta.data(), meta.size())) return bad("metadata read");
        if (conversation_disk_checksum(meta.data(), meta.size()) != p.metadata_checksum)
            return bad("metadata checksum");
        ParsedPanel panel;
        panel.base_tokens = p.base_tokens;
        panel.end_tokens = p.end_tokens;
        panel.payload_bytes = p.payload_bytes;
        panel.payload_checksum = p.payload_checksum;
        panel.payload_at = payload_at + sizeof(p) + p.metadata_bytes;
        MetaReader reader{meta.data(), meta.data() + meta.size(), true, {}};
        if (!reader.parse(panel.delta) || !reader.ok) return bad("metadata layout");
        if (panel.delta.stages.empty() || panel.delta.stages.front().cvec != (header.cvec != 0))
            return bad("cvec");
        if (!valid_lengths(panel.delta, reader.lengths, p.payload_bytes)) return bad("section sizes");
        // Every K/V layer's cells must be the panel's end rounded up to its page size.
        for (const auto& stage : panel.delta.stages)
            for (const auto& kv : stage.kv) {
                const int64_t page = std::max<int64_t>(1, kv.page_size);
                const int64_t aligned = ((int64_t) p.end_tokens + page - 1) / page * page;
                if (kv.cells != aligned) return bad("kv cells");
            }
        panel.lengths = std::move(reader.lengths);
        panels.push_back(std::move(panel));
        expected = p.end_tokens;
        payload_at += sizeof(p) + p.metadata_bytes + p.payload_bytes;
        if (k + 1 < header.panel_count)
            if (std::fseek(file, static_cast<long>(p.payload_bytes), SEEK_CUR) != 0) return bad("panel sizes");
    }
    if (panels.empty()) return bad("no panels");
    if (panels.back().end_tokens != header.total_tokens) return bad("lengths");
    return true;
}


// True when two deltas (or the base and a delta) describe the same stage layout.
bool stages_compatible(const ConversationDiskRecord& a, const ConversationDiskRecord& b,
                       std::string& error) {
    if (a.stages.size() != b.stages.size()) {
        error = "incompatible stage count";
        return false;
    }
    for (size_t i = 0; i < a.stages.size(); ++i) {
        if (a.stages[i].geometry != b.stages[i].geometry ||
            a.stages[i].layer_lo != b.stages[i].layer_lo ||
            a.stages[i].layer_hi != b.stages[i].layer_hi || a.stages[i].cvec != b.stages[i].cvec ||
            a.stages[i].kv.size() != b.stages[i].kv.size()) {
            error = "incompatible stage shape";
            return false;
        }
        for (size_t m = 0; m < a.stages[i].kv.size(); ++m) {
            const ConversationKv& x = a.stages[i].kv[m];
            const ConversationKv& y = b.stages[i].kv[m];
            // pooled_rows scales with the panel's extent (the pooled row count at its end), so it
            // is not part of the fixed geometry: only the newest panel's value is merged later.
            if (x.format != y.format || x.heads != y.heads || x.head_dim != y.head_dim ||
                x.page_size != y.page_size || x.idx_dim != y.idx_dim) {
                error = "incompatible K/V geometry across panels";
                return false;
            }
        }
    }
    return true;
}


} // namespace

bool ConversationDiskStore::base_ids(const std::string& base_name, std::vector<int32_t>& ids,
                                     std::vector<ConversationImageKey>& imgs, std::string& error) const {
    ids.clear();
    imgs.clear();
    const Entry* base = find(base_name);
    if (base == nullptr || base->points.empty() || base->points.front().ids.empty()) {
        error = "the base chain is not indexed";
        return false;
    }
    ids = base->points.front().ids;
    imgs = base->points.front().imgs;
    return true;
}

ConversationDiskStatus ConversationDiskStore::base_ids_from_file(const std::string& base_name,
                                                                 std::vector<int32_t>& ids,
                                                                 std::vector<ConversationImageKey>& imgs,
                                                                 std::string& error) const {
    ids.clear();
    imgs.clear();
    std::FILE* file = std::fopen(path_for(base_name).string().c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open the base chain " + base_name;
        return ConversationDiskStatus::miss;
    }
    ChainHeader header;
    std::vector<ParsedPanel> panels;
    if (!read_chain_meta(file, options_.identity, options_.budget_bytes, header, panels, error)) {
        std::fclose(file);
        return ConversationDiskStatus::corrupt;
    }
    std::fclose(file);
    if (header.base_kind != 0) {
        error = "a base chain cannot have its own base";
        return ConversationDiskStatus::corrupt;
    }
    ids.reserve((size_t) header.total_tokens);
    for (auto& panel : panels) {
        const SavedConversation& stage = panel.delta.stages.front();
        ids.insert(ids.end(), stage.live.ids.begin(), stage.live.ids.end());
        imgs.insert(imgs.end(), stage.live.imgs.begin(), stage.live.imgs.end());
    }
    if ((uint64_t) ids.size() != header.total_tokens) {
        error = "the base chain's ids do not match its coverage";
        return ConversationDiskStatus::corrupt;
    }
    return ConversationDiskStatus::ok;
}

ConversationDiskIdentity ConversationDiskIdentity::from_string(std::string_view text) {
    ConversationDiskIdentity identity;
    uint64_t lanes[4] = {};
    for (uint64_t lane = 0; lane < 4; ++lane) {
        const uint64_t seed = conversation_disk_checksum_seed + lane * 0x9e3779b97f4a7c15ull;
        lanes[lane] = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(text.data()), text.size(), seed);
    }
    std::memcpy(identity.bytes.data(), lanes, sizeof(lanes));
    return identity;
}

// One name class per purpose: 'c' for a conversation's own chain, 'p' for the shared system-prompt
// (chain root) prefix chain.  The rest of the name is the FNV-1a over the token prefix (with its
// image keys and steering mode), so the same prefix makes the same name in every conversation that
// shares it.  For a conversation chain the engine picks the name ONCE at seed time from the seed's
// content; appends keep it, so the file key is stable while the chain grows.
std::string disk_name(char cls, const std::vector<int32_t>& ids,
                      const std::vector<ConversationImageKey>& images, bool cvec) {
    uint64_t hash = conversation_disk_checksum_seed;
    const uint64_t count = ids.size();
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&count), sizeof(count), hash);
    if (!ids.empty())
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(ids.data()),
                                          ids.size() * sizeof(int32_t), hash);
    const uint64_t image_count = images.size();
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&image_count), sizeof(image_count), hash);
    for (const auto& key : images) {
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.start), sizeof(key.start), hash);
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.hash), sizeof(key.hash), hash);
    }
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&cvec), sizeof(cvec), hash);
    return std::string(1, cls) + hex64(hash) + "-" + hex64(count);
}

std::string conversation_disk_name(const std::vector<int32_t>& ids,
                                   const std::vector<ConversationImageKey>& images, bool cvec) {
    return disk_name('c', ids, images, cvec);
}

std::string conversation_disk_prefix_name(const std::vector<int32_t>& ids,
                                          const std::vector<ConversationImageKey>& images, bool cvec) {
    return disk_name('p', ids, images, cvec);
}

bool ConversationDiskStore::valid_name(const std::string& name) {
    if (name.empty() || name.size() > kMaxNameBytes || name == "." || name == "..") return false;
    for (const char c : name)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-')) return false;
    return true;
}

std::filesystem::path ConversationDiskStore::path_for(const std::string& name) const {
    return options_.directory / (name + file_suffix);
}

const ConversationDiskStore::Entry* ConversationDiskStore::find(const std::string& name) const {
    for (const auto& entry : entries_)
        if (entry.name == name) return &entry;
    return nullptr;
}

ConversationDiskStore::Entry* ConversationDiskStore::find(const std::string& name) {
    for (auto& entry : entries_)
        if (entry.name == name) return &entry;
    return nullptr;
}


size_t ConversationDiskStore::lru_victim(const std::string& skip) const {
    // tier 0: ordinary conversation chains.  tier 1: system-prompt chains nobody references.
    // tier 2: referenced system-prompt chains (only evicted when the budget must hold).
    for (int tier = 0; tier < 3; ++tier) {
        size_t victim = SIZE_MAX;
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (!skip.empty() && entries_[i].name == skip) continue;
            const bool root = entries_[i].root;
            const int tier_of = root ? (entries_[i].refs != 0 ? 2 : 1) : 0;
            if (tier_of != tier) continue;
            if (entries_[i].stamp < oldest) {
                oldest = entries_[i].stamp;
                victim = i;
            }
        }
        if (victim != SIZE_MAX) return victim;
    }
    return SIZE_MAX;
}

void ConversationDiskStore::evict(size_t index) {
    std::error_code error;
    std::filesystem::remove(path_for(entries_[index].name), error);
    bytes_ -= entries_[index].bytes;
    if (!entries_[index].base.empty())
        if (Entry* base = find(entries_[index].base)) --base->refs;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    ++evictions_;
}

ConversationDiskStatus ConversationDiskStore::make_room(uint64_t incoming, const std::string& skip,
                                                        bool adds_record, std::string& error) {
    auto over_budget = [&]() {
        return bytes_ + incoming > options_.budget_bytes ||
               (options_.max_records != 0 && adds_record && entries_.size() + 1 > options_.max_records);
    };
    while (over_budget()) {
        const size_t victim = lru_victim(skip);
        if (victim == SIZE_MAX) {
            error = "the disk cache cannot hold the record within its budget";
            return ConversationDiskStatus::invalid;
        }
        evict(victim);
    }
    std::error_code space_error;
    const std::filesystem::space_info space = std::filesystem::space(options_.directory, space_error);
    if (!space_error && space.available < incoming + options_.min_free_bytes) {
        uint64_t free_bytes = space.available;
        while (free_bytes < incoming + options_.min_free_bytes) {
            const size_t victim = lru_victim(skip);
            if (victim == SIZE_MAX) {
                error = "the disk cache has no free space for the record";
                return ConversationDiskStatus::failed;
            }
            free_bytes += entries_[victim].bytes;
            evict(victim);
        }
    }
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::open(const ConversationDiskOptions& options, std::string& error) {
    options_ = options;
    entries_.clear();
    bytes_ = 0;
    stamp_ = 0;
    recoveries_ = 0;
    if (options_.budget_bytes == 0) return ConversationDiskStatus::disabled;
    if (options_.directory.empty()) {
        error = "the disk cache has no directory";
        return ConversationDiskStatus::invalid;
    }
    std::error_code error_code;
    std::filesystem::create_directories(options_.directory, error_code);
    if (error_code) {
        error = "cannot create " + options_.directory.string();
        return ConversationDiskStatus::failed;
    }
    const size_t suffix = std::strlen(file_suffix);
    const std::string temporary_suffix = std::string(file_suffix) + ".tmp";
    std::vector<Entry> found;
    std::vector<std::filesystem::file_time_type> written;
    for (std::filesystem::directory_iterator it(options_.directory, error_code), end; !error_code && it != end;
         it.increment(error_code)) {
        std::error_code type_error;
        if (!it->is_regular_file(type_error)) continue;
        const std::string filename = it->path().filename().string();
        if (filename.size() > temporary_suffix.size() &&
            filename.compare(filename.size() - temporary_suffix.size(), temporary_suffix.size(),
                             temporary_suffix) == 0) {
            std::filesystem::remove(it->path(), type_error);
            continue;
        }
        if (filename.size() <= suffix || filename.compare(filename.size() - suffix, suffix, file_suffix) != 0)
            continue;
        const std::string name = filename.substr(0, filename.size() - suffix);
        if (!valid_name(name)) {
            std::filesystem::remove(it->path(), type_error);
            continue;
        }
        Entry entry;
        const ConversationDiskStatus status = index_file(it->path(), name, entry, error);
        if (status == ConversationDiskStatus::ok) {
            found.push_back(std::move(entry));
            written.push_back(it->last_write_time(type_error));
            ++recoveries_;
        } else if (status == ConversationDiskStatus::corrupt) {
            std::filesystem::remove(it->path(), type_error);
            ++corruptions_;
        }
        // A read failure keeps the file and skips it, so a transient error loses no data.
    }
    std::vector<size_t> order(found.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (written[a] != written[b]) return written[a] < written[b];
        return found[a].name < found[b].name;
    });
    entries_.reserve(found.size());
    for (const size_t i : order) {
        Entry entry = std::move(found[i]);
        entry.stamp = ++stamp_;
        bytes_ += entry.bytes;
        entries_.push_back(std::move(entry));
    }
    // Reference accounting: every chat chain pins its base root.  A chain whose base is gone
    // cannot load - drop it here (the engine will just re-read the prompt).
    for (size_t i = 0; i < entries_.size();) {
        Entry& entry = entries_[i];
        if (!entry.base.empty()) {
            Entry* base = find(entry.base);
            if (base == nullptr) {
                std::error_code remove_error;
                std::filesystem::remove(path_for(entry.name), remove_error);
                bytes_ -= entry.bytes;
                entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(i));
                ++corruptions_;
                continue;
            }
            ++base->refs;
        }
        ++i;
    }
    while (bytes_ > options_.budget_bytes ||
           (options_.max_records != 0 && entries_.size() > options_.max_records)) {
        const size_t victim = lru_victim(std::string());
        if (victim == SIZE_MAX) break;
        evict(victim);
    }
    error.clear();
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::index_file(const std::filesystem::path& path,
                                                         const std::string& name, Entry& entry,
                                                         std::string& error) const {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open " + path.string();
        return ConversationDiskStatus::failed;
    }
    auto bad = [&](const char* why) {
        std::fclose(file);
        error = std::string("invalid conversation record: ") + why;
        return ConversationDiskStatus::corrupt;
    };
    ChainHeader header;
    std::vector<ParsedPanel> panels;
    if (!read_chain_meta(file, options_.identity, options_.budget_bytes, header, panels, error))
        return bad(error.empty() ? "header" : error.c_str());
    std::fclose(file);
    // Build the merged ids/points from the metadata (no payload reads).
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> all_imgs;
    const uint64_t base_tokens = header.base_kind != 0 ? header.base_tokens : 0;
    if (header.base_kind != 0) {
        std::vector<int32_t> base_ids;
        std::vector<ConversationImageKey> base_imgs;
        const ConversationDiskStatus base_status =
            base_ids_from_file(header.base_name, base_ids, base_imgs, error);
        if (base_status != ConversationDiskStatus::ok) return bad("base chain");
        ids = std::move(base_ids);
        all_imgs = std::move(base_imgs);
        if ((uint64_t) ids.size() != base_tokens) return bad("base tokens");
    }
    ids.reserve((size_t) header.total_tokens);
    for (auto& panel : panels) {
        for (size_t si = 0; si < panel.delta.stages.size(); ++si) {
            const SavedConversation& stage = panel.delta.stages[si];
            if ((uint64_t) stage.live.ids.size() != panel.end_tokens - panel.base_tokens)
                return bad("token slice");
            if (si == 0) {
                ids.insert(ids.end(), stage.live.ids.begin(), stage.live.ids.end());
                all_imgs.insert(all_imgs.end(), stage.live.imgs.begin(), stage.live.imgs.end());
            }
        }
    }
    if ((uint64_t) ids.size() != header.total_tokens) return bad("lengths");
    entry.name = name;
    entry.cover = header.total_tokens;
    entry.cvec = header.cvec != 0;
    entry.root = !name.empty() && name[0] == 'p';
    entry.base = header.base_kind != 0 ? std::string(header.base_name) : std::string();
    entry.refs = 0;
    entry.bytes = std::filesystem::file_size(path);
    entry.points.clear();
    {
        ConversationCheckpoint final_point;
        final_point.ids = std::move(ids);
        final_point.imgs = std::move(all_imgs);
        entry.points.push_back(std::move(final_point));
    }
    for (auto& panel : panels)
        for (const auto& c : panel.delta.stages.front().checkpoints) {
            ConversationCheckpoint point;
            point.ids = c.ids;
            point.imgs = c.imgs;
            entry.points.push_back(std::move(point));
        }
    return ConversationDiskStatus::ok;
}

bool ConversationDiskStore::best(const std::vector<int32_t>& ids,
                                 const std::vector<ConversationImageKey>& images, bool cvec,
                                 ConversationDiskMatch& match) const {
    const Entry* chosen = nullptr;
    int64_t chosen_tokens = 0;
    for (const auto& entry : entries_) {
        if (entry.cvec != cvec) continue;
        for (const auto& point : entry.points) {
            const int64_t tokens = conversation_prefix(point, ids, images);
            const bool longer = tokens > chosen_tokens;
            // An equal-length tie prefers the SMALLER record: a 24k system-prompt record reads back
            // far less than the 80k conversation that contains the same prefix, for the same answer.
            const bool equal_smaller = tokens == chosen_tokens && tokens > 0 && chosen != nullptr &&
                                       (entry.bytes < chosen->bytes ||
                                        (entry.bytes == chosen->bytes && entry.stamp > chosen->stamp));
            if (longer || equal_smaller) {
                chosen_tokens = tokens;
                chosen = &entry;
            }
        }
    }
    if (chosen == nullptr || chosen_tokens <= 0) return false;
    match.name = chosen->name;
    match.tokens = chosen_tokens;
    return true;
}

// Validates that a seed/delta's checkpoints are new prefixes of the chain content.  Returns the
// point list (final point first, then the checkpoints, ids/imgs only).
std::vector<ConversationCheckpoint> chain_points(const std::vector<int32_t>& full_ids,
                                                 const std::vector<ConversationImageKey>& full_imgs,
                                                 const SavedConversation& stage, uint64_t base_tokens,
                                                 uint64_t end_tokens, std::string& error) {
    std::vector<ConversationCheckpoint> points;
    points.reserve(1 + stage.checkpoints.size());
    ConversationCheckpoint final_point;
    final_point.ids = full_ids;
    final_point.imgs = full_imgs;
    points.push_back(std::move(final_point));
    for (const auto& c : stage.checkpoints) {
        if (c.ids.empty() || (uint64_t) c.ids.size() <= base_tokens || (uint64_t) c.ids.size() > end_tokens ||
            !std::equal(c.ids.begin(), c.ids.end(), points.front().ids.begin())) {
            error = "an invalid checkpoint in the delta";
            return {};
        }
        ConversationCheckpoint point;
        point.ids = c.ids;
        point.imgs = c.imgs;
        points.push_back(std::move(point));
    }
    return points;
}

ConversationDiskStatus ConversationDiskStore::seed(const std::string& name, ConversationDiskRecord&& record,
                                                   std::string base_name, uint64_t base_tokens,
                                                   std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    if (record.stages.empty()) {
        error = "the record has no stage";
        return ConversationDiskStatus::invalid;
    }
    const bool cvec = record.stages.front().cvec;
    for (const auto& stage : record.stages)
        if (stage.cvec != cvec) {
            error = "the stages disagree on the cvec flag";
            return ConversationDiskStatus::invalid;
        }
    const uint64_t slice = (uint64_t) record.stages.front().live.ids.size();
    if (slice == 0) {
        error = "the record has no live token slice";
        return ConversationDiskStatus::invalid;
    }
    for (const auto& stage : record.stages)
        if ((uint64_t) stage.live.ids.size() != slice) {
            error = "the stages disagree on the token chain";
            return ConversationDiskStatus::invalid;
        }
    if (base_name.empty() && base_tokens != 0) {
        error = "base tokens without a base chain";
        return ConversationDiskStatus::invalid;
    }
    if (!base_name.empty() && (base_tokens == 0 || !valid_name(base_name))) {
        error = "an invalid base chain";
        return ConversationDiskStatus::invalid;
    }
    const uint64_t end = base_tokens + slice;
    if (end < base_tokens) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    // The full chain ids for checkpoint validation (and the index points).
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> all_imgs;
    if (!base_name.empty()) {
        if (!base_ids(base_name, ids, all_imgs, error))
            return ConversationDiskStatus::invalid;
        if ((uint64_t) ids.size() != base_tokens) {
            error = "the base chain does not cover base_tokens";
            return ConversationDiskStatus::invalid;
        }
    }
    ids.insert(ids.end(), record.stages.front().live.ids.begin(), record.stages.front().live.ids.end());
    all_imgs.insert(all_imgs.end(), record.stages.front().live.imgs.begin(),
                    record.stages.front().live.imgs.end());
    const std::vector<ConversationCheckpoint> points =
        chain_points(ids, all_imgs, record.stages.front(), base_tokens, end, error);
    if (points.empty()) return ConversationDiskStatus::invalid;
    MetaWriter metadata;
    metadata.u32(static_cast<uint32_t>(record.stages.size()));
    for (const auto& stage : record.stages) metadata.write_stage(stage);
    uint64_t payload_bytes = 0;
    for (const uint64_t bytes : metadata.lengths)
        if (!add_overflow(payload_bytes, bytes)) {
            error = "the record size overflows";
            return ConversationDiskStatus::invalid;
        }
    uint64_t record_bytes = kChainHeaderBytes + kPanelHeaderBytes;
    if (!add_overflow(record_bytes, metadata.data.size()) || !add_overflow(record_bytes, payload_bytes)) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    if (record_bytes > options_.budget_bytes) {
        error = "the record exceeds the byte budget";
        return ConversationDiskStatus::invalid;
    }
    // A re-seed replaces the chain with the same name; its root references transfer to the new chain.
    size_t old_refs = 0;
    if (Entry* old = find(name)) {
        old_refs = old->refs;
        if (!old->base.empty())
            if (Entry* b = find(old->base)) --b->refs;
        std::error_code ignored;
        std::filesystem::remove(path_for(name), ignored);
        bytes_ -= old->bytes;
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(old - entries_.data()));
    }
    const ConversationDiskStatus room = make_room(record_bytes, name, true, error);
    if (room != ConversationDiskStatus::ok) return room;
    // Publish atomically through a temporary file, like the old full-snapshot records.
    ChainHeader header;
    header.magic = kMagic;
    header.version = format_version;
    header.total_tokens = end;
    header.panel_count = 1;
    std::memcpy(header.identity, options_.identity.bytes.data(), ConversationDiskIdentity::size);
    header.cvec = cvec ? 1 : 0;
    header.base_kind = base_name.empty() ? 0 : 1;
    header.base_tokens = base_tokens;
    std::snprintf(header.base_name, sizeof(header.base_name), "%s", base_name.c_str());
    const std::filesystem::path target = path_for(name);
    const std::filesystem::path temporary = target.string() + ".tmp";
    std::error_code error_code;
    std::filesystem::remove(temporary, error_code);
    std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
    if (file == nullptr) {
        error = "cannot create " + temporary.string();
        return ConversationDiskStatus::failed;
    }
    bool ok = std::fwrite(&header, 1, sizeof(header), file) == sizeof(header);
    if (ok) {
        PanelHeader panel;
        panel.magic = kPanelMagic;
        panel.base_tokens = base_tokens;
        panel.end_tokens = end;
        panel.metadata_bytes = metadata.data.size();
        panel.metadata_checksum = conversation_disk_checksum(metadata.data.data(), metadata.data.size());
        panel.payload_bytes = payload_bytes;
        ok = std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
             (metadata.data.empty() ||
              std::fwrite(metadata.data.data(), 1, metadata.data.size(), file) == metadata.data.size());
        if (ok) {
            uint64_t checksum = conversation_disk_checksum_seed;
            auto write_section = [&](Section section) {
                if (!ok) return;
                if (section.bytes != nullptr) {
                    if (!section.bytes->empty() &&
                        std::fwrite(section.bytes->data(), 1, section.bytes->size(), file) != section.bytes->size())
                        ok = false;
                    checksum = conversation_disk_checksum(section.bytes->data(), section.bytes->size(), checksum);
                } else {
                    section.buffer->visit(0, section.buffer->size(),
                                          [&](const uint8_t* data, size_t bytes, size_t) {
                                              if (ok && std::fwrite(data, 1, bytes, file) != bytes) ok = false;
                                              checksum = conversation_disk_checksum(data, bytes, checksum);
                                              return ok;
                                          });
                }
            };
            visit_payload(record, write_section);
            if (ok) {
                panel.payload_checksum = checksum;
                uint64_t total = sizeof(panel) + metadata.data.size() + payload_bytes;
                ok = std::fseek(file, -static_cast<long>(total), SEEK_CUR) == 0 &&
                     std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
                     std::fseek(file, static_cast<long>(total), SEEK_CUR) == 0;
            }
        }
    }
    if (std::fclose(file) != 0) ok = false;
    if (ok) {
        std::filesystem::rename(temporary, target, error_code);
        if (error_code) ok = false;
    }
    if (!ok) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        error = "cannot write " + name;
        return ConversationDiskStatus::failed;
    }
    Entry entry;
    entry.name = name;
    entry.cover = end;
    entry.stamp = ++stamp_;
    entry.cvec = cvec;
    entry.root = !name.empty() && name[0] == 'p';
    entry.refs = old_refs;
    entry.base = base_name;
    entry.bytes = std::filesystem::file_size(target);
    entry.points = points;
    bytes_ += entry.bytes;
    entries_.push_back(std::move(entry));
    if (!base_name.empty())
        if (Entry* base = find(base_name)) ++base->refs;
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::append(const std::string& name, ConversationDiskRecord&& record,
                                                     std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    if (record.stages.empty() || record.stages.front().live.ids.empty()) {
        error = "the record has no live token slice";
        return ConversationDiskStatus::invalid;
    }
    if (record.stages.front().cvec != entry->cvec) {
        error = "the record's cvec flag does not match the chain";
        return ConversationDiskStatus::invalid;
    }
    const uint64_t slice = (uint64_t) record.stages.front().live.ids.size();
    for (const auto& stage : record.stages)
        if ((uint64_t) stage.live.ids.size() != slice) {
            error = "the stages disagree on the token chain";
            return ConversationDiskStatus::invalid;
        }
    const uint64_t base = entry->cover;
    const uint64_t end = base + slice;
    if (end < base) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    // Validate the checkpoints against the chain's full ids (which the final point holds) BEFORE
    // any file write: an invalid delta must not leave the chain half written.
    std::vector<ConversationCheckpoint> points;
    {
        std::vector<int32_t> full = entry->points.front().ids;
        full.insert(full.end(), record.stages.front().live.ids.begin(), record.stages.front().live.ids.end());
        std::vector<ConversationImageKey> imgs = entry->points.front().imgs;
        imgs.insert(imgs.end(), record.stages.front().live.imgs.begin(), record.stages.front().live.imgs.end());
        points = chain_points(full, imgs, record.stages.front(), base, end, error);
        if (points.empty()) return ConversationDiskStatus::invalid;
        points.insert(points.begin() + 1, entry->points.begin() + 1, entry->points.end());
    }
    MetaWriter metadata;
    metadata.u32(static_cast<uint32_t>(record.stages.size()));
    for (const auto& stage : record.stages) metadata.write_stage(stage);
    uint64_t payload_bytes = 0;
    for (const uint64_t bytes : metadata.lengths)
        if (!add_overflow(payload_bytes, bytes)) {
            error = "the record size overflows";
            return ConversationDiskStatus::invalid;
        }
    uint64_t panel_bytes = kPanelHeaderBytes;
    if (!add_overflow(panel_bytes, metadata.data.size()) || !add_overflow(panel_bytes, payload_bytes)) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    const ConversationDiskStatus room = make_room(panel_bytes, name, false, error);
    if (room != ConversationDiskStatus::ok) return room;
    const std::filesystem::path target = path_for(name);
    // "r+b": appends must be able to seek back and patch the panel/chain headers in place (an
    // append-mode stream would force EVERY write to the end of the file).
    std::FILE* file = std::fopen(target.string().c_str(), "r+b");
    if (file == nullptr) {
        error = "cannot open " + target.string();
        return ConversationDiskStatus::failed;
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        error = "cannot seek " + target.string();
        return ConversationDiskStatus::failed;
    }
    const uint64_t start_bytes = std::filesystem::file_size(target);
    bool ok = true;
    {
        PanelHeader panel;
        panel.magic = kPanelMagic;
        panel.base_tokens = base;
        panel.end_tokens = end;
        panel.metadata_bytes = metadata.data.size();
        panel.metadata_checksum = conversation_disk_checksum(metadata.data.data(), metadata.data.size());
        panel.payload_bytes = payload_bytes;
        ok = std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
             (metadata.data.empty() ||
              std::fwrite(metadata.data.data(), 1, metadata.data.size(), file) == metadata.data.size());
        if (ok) {
            uint64_t checksum = conversation_disk_checksum_seed;
            auto write_section = [&](Section section) {
                if (!ok) return;
                if (section.bytes != nullptr) {
                    if (!section.bytes->empty() &&
                        std::fwrite(section.bytes->data(), 1, section.bytes->size(), file) != section.bytes->size())
                        ok = false;
                    checksum = conversation_disk_checksum(section.bytes->data(), section.bytes->size(), checksum);
                } else {
                    section.buffer->visit(0, section.buffer->size(),
                                          [&](const uint8_t* data, size_t bytes, size_t) {
                                              if (ok && std::fwrite(data, 1, bytes, file) != bytes) ok = false;
                                              checksum = conversation_disk_checksum(data, bytes, checksum);
                                              return ok;
                                          });
                }
            };
            visit_payload(record, write_section);
            if (ok) {
                panel.payload_checksum = checksum;
                uint64_t total = sizeof(panel) + metadata.data.size() + payload_bytes;
                ok = std::fseek(file, -static_cast<long>(total), SEEK_CUR) == 0 &&
                     std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
                     std::fseek(file, static_cast<long>(total), SEEK_CUR) == 0;
            }
        }
    }
    if (ok) {
        // Patch the chain header: one more panel, the new coverage.
        ChainHeader header;
        ok = std::fseek(file, 0, SEEK_SET) == 0 &&
             std::fread(&header, 1, sizeof(header), file) == sizeof(header) &&
             header.magic == kMagic && header.version == format_version;
        if (ok) {
            header.panel_count += 1;
            header.total_tokens = end;
            ok = std::fseek(file, 0, SEEK_SET) == 0 &&
                 std::fwrite(&header, 1, sizeof(header), file) == sizeof(header);
        }
    }
    if (std::fclose(file) != 0) ok = false;
    if (!ok) {
        // Roll the torn tail back so the chain stays at its previous coverage.
        std::error_code ignored;
        std::filesystem::resize_file(target, start_bytes, ignored);
        return ConversationDiskStatus::failed;
    }
    const uint64_t new_bytes = std::filesystem::file_size(target);
    bytes_ += new_bytes - entry->bytes;
    entry->bytes = new_bytes;
    entry->cover = end;
    entry->points = std::move(points);
    return ConversationDiskStatus::ok;
}

bool ConversationDiskStore::cover(const std::string& name, uint64_t& tokens) const {
    const Entry* entry = find(name);
    if (entry == nullptr) return false;
    tokens = entry->cover;
    return true;
}

ConversationDiskStatus ConversationDiskStore::remove(const std::string& name, std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    std::error_code error_code;
    std::filesystem::remove(path_for(name), error_code);
    if (error_code) {
        error = "cannot remove " + path_for(name).string();
        return ConversationDiskStatus::failed;
    }
    if (!entry->base.empty())
        if (Entry* base = find(entry->base)) --base->refs;
    bytes_ -= entry->bytes;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(entry - entries_.data()));
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::rewind_chain(const std::string& name, uint64_t at,
                                                           uint64_t& new_cover, std::string& error) {
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    if (at >= entry->cover) {
        new_cover = entry->cover;
        return ConversationDiskStatus::ok;
    }
    const std::filesystem::path target = path_for(name);
    std::FILE* file = std::fopen(target.string().c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open " + target.string();
        return ConversationDiskStatus::failed;
    }
    ChainHeader header;
    if (!read_exact(file, &header, sizeof(header))) {
        std::fclose(file);
        error = "corrupt conversation record: header";
        return ConversationDiskStatus::corrupt;
    }
    bool ok = header.magic == kMagic && header.version == format_version && header.base_kind <= 1;
    uint64_t pos = kChainHeaderBytes;
    uint64_t kept = 0;
    uint64_t kept_end = header.base_kind != 0 ? header.base_tokens : 0;
    for (uint64_t k = 0; ok && k < header.panel_count; ++k) {
        PanelHeader p{};
        if (std::fseek(file, static_cast<long>(pos), SEEK_SET) != 0 ||
            !read_exact(file, &p, sizeof(p)) || p.magic != kPanelMagic ||
            p.base_tokens != kept_end) {
            ok = false;
            break;
        }
        if (p.end_tokens > at) break;               // drop this panel and everything after it
        kept = k + 1;
        kept_end = p.end_tokens;
        pos += sizeof(p) + p.metadata_bytes + p.payload_bytes;
    }
    std::fclose(file);
    if (!ok) {
        error = "corrupt conversation record: panels";
        return ConversationDiskStatus::corrupt;
    }
    if (kept == header.panel_count) {
        new_cover = kept_end;
        return ConversationDiskStatus::ok;
    }
    if (kept == 0) {
        // A header without a panel is not a valid record. Leave the old chain
        // intact and let the caller seed the shorter branch as a new record.
        new_cover = 0;
        return ConversationDiskStatus::miss;
    }
    std::error_code ignored;
    std::filesystem::resize_file(target, pos, ignored);
    file = std::fopen(target.string().c_str(), "r+b");
    if (file == nullptr) {
        error = "cannot reopen " + target.string();
        return ConversationDiskStatus::failed;
    }
    ok = std::fseek(file, 0, SEEK_SET) == 0 && std::fread(&header, 1, sizeof(header), file) == sizeof(header);
    if (ok) {
        header.panel_count = kept;
        header.total_tokens = kept_end;
        ok = std::fseek(file, 0, SEEK_SET) == 0 && std::fwrite(&header, 1, sizeof(header), file) == sizeof(header);
    }
    std::fclose(file);
    if (!ok) {
        error = "cannot patch " + target.string();
        return ConversationDiskStatus::failed;
    }
    const uint64_t new_bytes = std::filesystem::file_size(target);
    bytes_ += new_bytes - entry->bytes;
    entry->bytes = new_bytes;
    entry->cover = kept_end;
    // Rebuild the points from the retained panels' metadata.
    Entry rebuilt;
    const ConversationDiskStatus status = index_file(target, entry->name, rebuilt, error);
    if (status != ConversationDiskStatus::ok) return status;
    entry->points = std::move(rebuilt.points);
    new_cover = kept_end;
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::get(const std::string& name, ConversationDiskRecord& record,
                                                  std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    const Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    ConversationDiskStatus status = read_chain(path_for(name), 0, record, error);
    if (status == ConversationDiskStatus::corrupt) {
        // A torn tail (a crash mid-append) truncates back to the last complete panel instead of
        // losing the whole conversation.
        if (truncate_torn_tail(name)) {
            std::string retry_error;
            status = read_chain(path_for(name), 0, record, retry_error);
            if (status == ConversationDiskStatus::ok) {
                error = std::move(retry_error);
                if (Entry* used = find(name)) {
                    used->stamp = ++stamp_;
                    const uint64_t bytes = std::filesystem::file_size(path_for(name));
                    bytes_ += bytes - used->bytes;
                    used->bytes = bytes;
                }
                return status;
            }
        }
        // Real corruption: drop the chain (its referenced base, if any, stays for the other chats).
        std::error_code remove_error;
        if (Entry* stale = find(name)) {
            if (!stale->base.empty())
                if (Entry* base = find(stale->base)) --base->refs;
            std::filesystem::remove(path_for(name), remove_error);
            bytes_ -= stale->bytes;
            entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(stale - entries_.data()));
        }
        ++corruptions_;
        return ConversationDiskStatus::corrupt;
    }
    if (status == ConversationDiskStatus::ok)
        if (Entry* used = find(name)) used->stamp = ++stamp_;
    return status;
}

// Verifies the LAST panel's payload checksum and truncates the file when it is torn, patching the
// chain header and re-indexing the entry.  Returns false when the tail is intact (or the file does
// not validate at all - a genuine corruption the caller removes).
bool ConversationDiskStore::truncate_torn_tail(const std::string& name) {
    const std::filesystem::path target = path_for(name);
    std::FILE* file = std::fopen(target.string().c_str(), "rb");
    if (file == nullptr) return false;
    ChainHeader header;
    if (!read_exact(file, &header, sizeof(header)) || header.magic != kMagic ||
        header.version != format_version || header.base_kind > 1 || header.panel_count > kMaxPanels ||
        header.panel_count == 0) {
        std::fclose(file);
        return false;
    }
    // Walk to the last panel; verify its payload checksum.
    uint64_t pos = kChainHeaderBytes;
    uint64_t kept_end = header.base_kind != 0 ? header.base_tokens : 0;
    bool ok = true;
    for (uint64_t k = 0; ok && k < header.panel_count; ++k) {
        PanelHeader p{};
        if (std::fseek(file, static_cast<long>(pos), SEEK_SET) != 0 ||
            !read_exact(file, &p, sizeof(p)) || p.magic != kPanelMagic ||
            p.base_tokens != kept_end) {
            ok = false;
            break;
        }
        if (k + 1 == header.panel_count) {
            if (std::fseek(file, static_cast<long>(p.metadata_bytes), SEEK_CUR) != 0) { ok = false; break; }
            uint64_t checksum = conversation_disk_checksum_seed;
            std::vector<uint8_t> scratch(1u << 20);
            uint64_t remaining = p.payload_bytes;
            while (ok && remaining != 0) {
                const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, scratch.size()));
                if (!read_exact(file, scratch.data(), chunk)) { ok = false; break; }
                checksum = conversation_disk_checksum(scratch.data(), chunk, checksum);
                remaining -= chunk;
            }
            if (!ok) break;
            if (checksum == p.payload_checksum) { std::fclose(file); return false; }   // tail intact
            std::fclose(file);
            // Truncate back to the start of the torn panel and patch the chain header.
            const uint64_t panel_at = pos;
            std::error_code ignored;
            std::filesystem::resize_file(target, panel_at, ignored);
            std::FILE* writer = std::fopen(target.string().c_str(), "r+b");
            if (writer == nullptr) return false;
            ChainHeader patched = header;
            patched.panel_count = k;
            patched.total_tokens = kept_end;
            bool patched_ok = std::fwrite(&patched, 1, sizeof(patched), writer) == sizeof(patched);
            std::fclose(writer);
            if (!patched_ok) return false;
            // Re-index the entry from the retained panels.
            Entry rebuilt;
            std::string index_error;
            if (index_file(target, name, rebuilt, index_error) != ConversationDiskStatus::ok) return false;
            if (Entry* used = find(name)) {
                bytes_ += rebuilt.bytes - used->bytes;
                used->bytes = rebuilt.bytes;
                used->cover = rebuilt.cover;
                used->points = std::move(rebuilt.points);
            }
            return true;
        }
        kept_end = p.end_tokens;
        pos += sizeof(p) + p.metadata_bytes + p.payload_bytes;
    }
    std::fclose(file);
    return false;
}

ConversationDiskStatus ConversationDiskStore::read_chain(const std::filesystem::path& path, int depth,
                                                         ConversationDiskRecord& out, std::string& error) const {
    return read_chain_impl(path, *this, depth, out, error);
}

ConversationDiskStatus ConversationDiskStore::read_chain_impl(const std::filesystem::path& path,
                                                              const ConversationDiskStore& self, int depth,
                                                              ConversationDiskRecord& out,
                                                              std::string& error) {

    if (depth > 1) {
        error = "corrupt conversation record: chain base depth";
        return ConversationDiskStatus::corrupt;
    }
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        if (errno == ENOENT) return ConversationDiskStatus::miss;
        error = "cannot open " + path.string();
        return ConversationDiskStatus::failed;
    }
    auto bad = [&](const char* why) {
        std::fclose(file);
        error = std::string("corrupt conversation record: ") + why;
        return ConversationDiskStatus::corrupt;
    };
    ChainHeader header;
    std::vector<ParsedPanel> panels;
    if (!read_chain_meta(file, self.options_.identity, self.options_.budget_bytes, header, panels, error))
        return bad(error.empty() ? "header" : error.c_str());
    const uint64_t base_tokens = header.base_kind != 0 ? header.base_tokens : 0;
    // Resolve the base (a 'p' chain; its own base must be none - enforced by index_file/seed).
    ConversationDiskRecord base;
    if (header.base_kind != 0) {
        const Entry* base_entry = self.find(header.base_name);
        if (base_entry == nullptr) return bad("missing base chain");
        const ConversationDiskStatus got = self.read_chain_impl(self.path_for(header.base_name), self,
                                                               depth + 1, base, error);
        if (got != ConversationDiskStatus::ok)
            return bad("missing base chain");
        if (base.stages.empty() || (uint64_t) base.stages.front().live.ids.size() != base_tokens)
            return bad("base coverage");
    }
    // Stage layout must be consistent across the base and every panel.
    const size_t stage_count = panels.front().delta.stages.size();
    if (!base.stages.empty()) {
        if (!stages_compatible(base, panels.front().delta, error)) return bad(error.c_str());
    }
    for (size_t k = 1; k < panels.size(); ++k)
        if (!stages_compatible(panels[0].delta, panels[k].delta, error)) return bad(error.c_str());
    // Build the merged image skeleton from the metadata (ids, imgs, checkpoint identities, shapes).
    out.stages.resize(stage_count);
    std::vector<std::vector<size_t>> cp_start(stage_count);   // per panel, per stage: checkpoint index
    std::vector<std::vector<std::array<size_t, 5>>> c_per(stage_count);   // bytes per unit, per stage/layer
    for (size_t si = 0; si < stage_count; ++si) {
        SavedConversation& st = out.stages[si];
        const SavedConversation* base_st = base.stages.empty() ? nullptr : &base.stages[si];
        const SavedConversation& first = panels.front().delta.stages[si];
        st.geometry = first.geometry;
        st.layer_lo = first.layer_lo;
        st.layer_hi = first.layer_hi;
        st.cvec = first.cvec;
        st.kv.resize(first.kv.size());
        (void) si;
        st.live.ids.clear();
        st.live.imgs.clear();
        if (base_st != nullptr) {
            st.live.ids = base_st->live.ids;
            st.live.imgs = base_st->live.imgs;
        }
        for (const auto& panel : panels) {
            const SavedConversation& stage = panel.delta.stages[si];
            if ((uint64_t) stage.live.ids.size() != panel.end_tokens - panel.base_tokens)
                return bad("token slice");
            st.live.ids.insert(st.live.ids.end(), stage.live.ids.begin(), stage.live.ids.end());
            st.live.imgs.insert(st.live.imgs.end(), stage.live.imgs.begin(), stage.live.imgs.end());
        }
        if ((uint64_t) st.live.ids.size() != header.total_tokens) return bad("lengths:staids");
        st.live.used = panels.back().delta.stages[si].live.used;
        // Checkpoints: the base's (fully loaded arrays), then the panels' (ids/imgs now, arrays
        // streamed below).
        st.checkpoints.clear();
        if (base_st != nullptr)
            for (auto& c : base_st->checkpoints) st.checkpoints.push_back(std::move(c));
        cp_start[si].resize(panels.size());
        for (size_t k = 0; k < panels.size(); ++k) {
            cp_start[si][k] = st.checkpoints.size();
            for (auto& c : panels[k].delta.stages[si].checkpoints) {
                if (!std::equal(c.ids.begin(), c.ids.end(), st.live.ids.begin()))
                    return bad("checkpoint prefix");
                ConversationCheckpoint out_c;
                out_c.ids = std::move(c.ids);
                out_c.imgs = std::move(c.imgs);
                out_c.used = c.used;
                st.checkpoints.push_back(std::move(out_c));
            }
        }
        // K/V shapes and buffer sizes: base bytes + every panel's tail lengths.
        for (size_t m = 0; m < first.kv.size(); ++m) {
            const ConversationKv& shape = first.kv[m];
            ConversationKv& out_kv = st.kv[m];
            out_kv.format = shape.format;
            out_kv.cells = panels.back().delta.stages[si].kv[m].cells;
            out_kv.heads = shape.heads;
            out_kv.head_dim = shape.head_dim;
            out_kv.page_size = shape.page_size;
            out_kv.pooled_rows = panels.back().delta.stages[si].kv[m].pooled_rows;
            out_kv.idx_dim = shape.idx_dim;
            // Each panel's tail starts at the byte offset of the PAGE containing its base and
            // runs to the panel's final (page-capped) cells: the overlap page is re-saved whole,
            // so a panel overwrites the previous content's partial page instead of duplicating
            // it.  The merged buffer totals come from the LAST panel's final cell count.
            const ConversationKv& last_kv = panels.back().delta.stages[si].kv[m];
            const auto last_panel_lens = panel_kv_lens(panels.back().delta, panels.back().lengths);
            if (last_panel_lens.size() <= si || last_panel_lens[si].size() <= m) return bad("lengths:kvlen");
            const std::array<size_t, 5> last_lens = last_panel_lens[si][m];
            const int64_t last_page = std::max<int64_t>(1, last_kv.page_size);
            // Each panel's tail covers the layout UNITS after its base: page-capped cells for the
            // K/V buffers, pooled rows for the indexer (see kv_units).  The merged buffer totals
            // come from the LAST panel's final unit count.
            std::array<size_t, 5> section_per{};      // bytes per unit of the merged layout
            std::array<size_t, 5> kv_total{};
            for (int e = 0; e < 5; ++e) {
                const int64_t first = last_kv.first_units[(size_t) e];
                const int64_t units = kv_units(first, (int64_t) panels.back().end_tokens, last_page, e);
                if (first < 0 || units <= 0) return bad("lengths:kvcell");
                if (last_lens[(size_t) e] % (uint64_t) units != 0) return bad("lengths:kvmod");
                section_per[(size_t) e] = last_lens[(size_t) e] / (uint64_t) units;
                kv_total[(size_t) e] =
                    (size_t) kv_end_units((int64_t) panels.back().end_tokens, last_page, e) *
                    section_per[(size_t) e];
            }
            // Validate every panel's tail lengths against the same per-unit size (the stage layout
            // is already validated identical across panels).
            for (auto& panel : panels) {
                const ConversationKv& kv_panel = panel.delta.stages[si].kv[m];
                const auto panel_lens = panel_kv_lens(panel.delta, panel.lengths);
                if (panel_lens.size() <= si || panel_lens[si].size() <= m) return bad("lengths:kvlen");
                const std::array<size_t, 5> lens = panel_lens[si][m];
                for (int e = 0; e < 5; ++e) {
                    const int64_t first = kv_panel.first_units[(size_t) e];
                    const int64_t units = kv_units(first, (int64_t) panel.end_tokens, last_page, e);
                    if (first < 0 || units <= 0 || lens[(size_t) e] != (uint64_t) units * section_per[(size_t) e])
                        return bad("lengths:kvlen");
                }
            }
            c_per[si].resize(first.kv.size());
            c_per[si][m] = section_per;
            const ConversationKv* base_kv = base_st != nullptr ? &base_st->kv[m] : nullptr;
            out_kv.k.resize(kv_total[0]);
            out_kv.v.resize(kv_total[1]);
            out_kv.k_scale.resize(kv_total[2]);
            out_kv.v_scale.resize(kv_total[3]);
            out_kv.pooled.resize(kv_total[4]);
            (void) m;
            // Copy the base's bytes into the merged buffers.
            if (base_kv != nullptr) {
                auto copy_from = [&](ConversationBuffer& dst, const ConversationBuffer& src) {
                    if (src.empty()) return true;
                    return dst.visit(0, src.size(), [&](uint8_t* p, size_t chunk, size_t at) {
                        return src.read(p, at, chunk);
                    });
                };
                if (!copy_from(out_kv.k, base_kv->k) || !copy_from(out_kv.v, base_kv->v) ||
                    !copy_from(out_kv.k_scale, base_kv->k_scale) || !copy_from(out_kv.v_scale, base_kv->v_scale) ||
                    !copy_from(out_kv.pooled, base_kv->pooled))
                    return bad("base copy");
            }
        }
    }
    // Stream every panel's payload in order.  Each panel's payload checksum is verified against its
    // own header; the file position was left at the end of the chain by read_chain_meta, so seek to
    // each payload.
    std::vector<uint8_t> scratch;
    for (size_t k = 0; k < panels.size(); ++k) {
        const ParsedPanel& panel = panels[k];
        if (std::fseek(file, static_cast<long>(panel.payload_at), SEEK_SET) != 0)
            return bad("payload position");
        const bool last = k + 1 == panels.size();
        uint64_t checksum = conversation_disk_checksum_seed;
        auto read_checksummed = [&](void* dst, uint64_t bytes) -> bool {
            if (!read_exact(file, dst, bytes)) return false;
            checksum = conversation_disk_checksum(static_cast<const uint8_t*>(dst),
                                                  static_cast<size_t>(bytes), checksum);
            return true;
        };
        size_t li = 0;
        auto length = [&]() -> uint64_t {
            return li < panel.lengths.size() ? panel.lengths[li++] : 0;
        };
        for (size_t si = 0; si < stage_count; ++si) {
            SavedConversation& st = out.stages[si];
            // Stage live run-state: only the LAST panel's lands in the merged image.
            std::array<std::vector<uint8_t>*, 5> live = live_arrays(st.live);
            for (int e = 0; e < 5; ++e) {
                const uint64_t bytes = length();
                std::vector<uint8_t>* dst = last ? live[(size_t) e] : nullptr;
                if (dst != nullptr) {
                    dst->resize(static_cast<size_t>(bytes));
                    if (!read_checksummed(dst->data(), bytes)) return bad("payload read");
                } else {
                    scratch.resize(static_cast<size_t>(bytes));
                    if (!read_checksummed(scratch.data(), bytes)) return bad("payload read");
                }
            }
            // Checkpoints: the panels' land in order after the base's.
            for (size_t j = 0; j < panel.delta.stages[si].checkpoints.size(); ++j) {
                std::array<std::vector<uint8_t>*, 5> dst = live_arrays(st.checkpoints[cp_start[si][k] + j]);
                for (int e = 0; e < 5; ++e) {
                    const uint64_t bytes = length();
                    dst[(size_t) e]->resize(static_cast<size_t>(bytes));
                    if (!read_checksummed(dst[(size_t) e]->data(), bytes)) return bad("payload read");
                }
            }
            // K/V tails into the pre-sized buffers at the page-aligned OFFSET of each panel's
            // base (tails overlap the previous content's partial page by design - they rewrite it).
            for (size_t m = 0; m < st.kv.size(); ++m) {
                ConversationKv& out_kv = st.kv[m];
                std::array<ConversationBuffer*, 5> dst = kv_arrays(out_kv);
                const int64_t page = std::max<int64_t>(1, out_kv.page_size);
                const ConversationKv& kv_panel = panel.delta.stages[si].kv[m];
                for (int e = 0; e < 5; ++e) {
                    const uint64_t bytes = length();
                    const int64_t first = kv_panel.first_units[(size_t) e];
                    const int64_t units = kv_units(first, (int64_t) panel.end_tokens, page, e);
                    if (first < 0 || units <= 0 || bytes != (uint64_t) units * c_per[si][m][(size_t) e])
                        return bad("lengths");
                    const size_t at = (size_t) first * c_per[si][m][(size_t) e];
                    if (!dst[(size_t) e]->visit(at, static_cast<size_t>(bytes),
                            [&](uint8_t* p, size_t chunk, size_t) {
                                return read_checksummed(p, chunk);
                            }))
                        return bad("payload read");
                }
            }
        }
        if (li != panel.lengths.size()) return bad("section sizes");
        if (checksum != panel.payload_checksum) return bad("payload checksum");
    }
    std::fclose(file);
    return ConversationDiskStatus::ok;
}
} // namespace strata::core
