# Upstream improvements survey — V100 (sm_70) benefit

Read-only review of the open pull requests and issues of the upstream repository
[Niko1221/Strata](https://github.com/Niko1221/Strata), surveyed 2026-10-04, for
changes that would benefit this fork's V100 cards. One file per item follows
this index.

Every file states the upstream item, what it does, its measured numbers from the
upstream report, the fork's current state (with local evidence), and a
recommendation. Items the fork already solves with its own, better solution are
recorded under `REVIEWED-*` files and should not be ported as-is.

Legend: **port** = implement as written; **test first** = implement behind a
switch and measure on the V100s before enabling; **conditional** = needs
another feature first; **skip** = already solved in the fork.

---
## Release v0.1.40.2 integration

See [RELEASE-v0.1.40.2-V100.md](RELEASE-v0.1.40.2-V100.md) for the selected
changes, local verification, operating limits, and the next integration steps.
This release review takes precedence over the older survey for these changes.


## Recommended implementation order

Do the items in this order. Effort: S = small (hours), M = a few days,
L = a week or more.

### 1. Port first — small, certain wins

| # | Item | Effort | Why first |
| --- | --- | --- | --- |
| 1 | [PR-800-Volta-prompt-attention-two-accumulator-chains.md](PR-800-Volta-prompt-attention-two-accumulator-chains.md) | S | Fixes the fork's own flagship V100 kernel; halves the attention error (5.9e-6 -> 3.0e-6) for ~2% time. Localized change; parity harness already exists. |
| 2 | [PR-792-zero-doorbell-batch-fix.md](PR-792-zero-doorbell-batch-fix.md) | S | A few lines in `run_slot_rows`; `--batch` on a 100%-resident stage otherwise hangs ("timed out at layer 0"). The fork's stages do reach full residency. |
| 3 | [PR-711-k8v4-KV-streaming.md](PR-711-k8v4-KV-streaming.md) | M | Removes the `--kv k8v4 --kv-resident` refusal (`generate.cpp:1948`). 816 B/cell KV + streamed VRAM = more expert slots on 16 GB cards. |
| 4 | [PR-743-coalesced-streaming-topk-long-prompts.md](PR-743-coalesced-streaming-topk-long-prompts.md) | M | 250K top-k 12.8 -> 1.13 ms, +23% 248K prefill. The fork's 262K/524K prompts hit the 135K-cell register limit today. |
| 5 | [PR-789-prefill-overlap-shared-expert-uploads.md](PR-789-prefill-overlap-shared-expert-uploads.md) | M | Schedule-only change in `prefill.cpp`; 4K prefill on the slow stage waits 2,037/5,266 ms in copy waits (PROGRESS.md:84). |
| 6 | [PR-796-startup-VRAM-plan.md](PR-796-startup-VRAM-plan.md) | M/L | Prices prefill staging before committing the expert cache; stops the 524K config from dying between load and READY. Replaces the crude `160 + chunk*680/1024` estimate (`generate.cpp:3185`). Also answers ISSUE-486. |
| 7 | [PR-802-IQ1_S-GPU-and-AVX2.md](PR-802-IQ1_S-GPU-and-AVX2.md) | L | The largest model-fit lever: IQ1_S fits a 16 GB V100; AVX-2 CPU expert kernels are 7.7-13.8x. Largest port (two kernel families + fixtures). |

### 2. Port, then measure on the V100s

| # | Item | Effort | Note |
| --- | --- | --- | --- |
| 8 | [PR-783-fused-decode-verify-MTP-kernels.md](PR-783-fused-decode-verify-MTP-kernels.md) | M/L | Verify latency -13..-18% upstream; decode here is verify-bound (27.9/30.6 ms per window). Keep the fork's own fused-GR kernel where it overlaps. |
| 9 | [PR-794-split-placement-per-stage-PCIe-link.md](PR-794-split-placement-per-stage-PCIe-link.md) | M | The auto search still uses one 190 ms miss constant while the rig's GPU0 is Gen3 x2. Runtime per-stage pcie_frac already exists. |
| 10 | [PR-742-FP32-tiled-QSA-block-scores-below-sm80.md](PR-742-FP32-tiled-QSA-block-scores-below-sm80.md) | M | Test first: the fork already rejected one Volta scorer candidate (1.8-2x slower, PROGRESS.md:34). This is a different kernel, but verify on sm_70. |
| 11 | [PR-699-Linux-read-ahead-at-startup.md](PR-699-Linux-read-ahead-at-startup.md) | S | Cold start 920 s -> 70 s upstream. Partial overlap exists (madvise WILLNEED, io_uring PLE); extend, do not duplicate. |
| 12 | [PR-773-file-tier-unbuffered-reads-Linux.md](PR-773-file-tier-unbuffered-reads-Linux.md) | M | Linux file tier still reads through the page cache today; reuse the fork's io_uring pattern or the PR's aio path. |
| 13 | [PR-733-pipeline-CPU-expert-work.md](PR-733-pipeline-CPU-expert-work.md) | M | Opt-in `STRATA_POOL_PIPELINE`; only helps when CPU experts run (small budget / long context). Modest, scheduling-only. |
| 14 | [PR-693-prefill-auto-finer-equal-chunks.md](PR-693-prefill-auto-finer-equal-chunks.md) | M | Test first: the upstream +21-38% was on a fast-link card; the fork's x2 link may blunt the gain. |

### 3. Conditional — blocked on another feature or a trial

| # | Item | Blocked on |
| --- | --- | --- |
| 15 | [PR-704-HC_Q8-int8-hyper-connections.md](PR-704-HC_Q8-int8-hyper-connections.md) | Upstream says "not with a layer split"; the fork's default is a split. Needs split support first. |
| 16 | [PR-732-overlap-streamed-KV-uploads-with-prefill.md](PR-732-overlap-streamed-KV-uploads-with-prefill.md) | Needs KV streaming in use (after PR-711); the fork's production config keeps KV in VRAM. |
| 17 | [PR-378-elastic-KV-kv-grow.md](PR-378-elastic-KV-kv-grow.md) | Needs a V100-specific trial: the fork already measured KV-residency reduction slowing long-prompt decode (PROGRESS.md:162). |
| 18 | [PR-793-pipelined-windows-over-active-slots.md](PR-793-pipelined-windows-over-active-slots.md) | Only matters when `--batch` slots are served; port part 1 (active-slot trim), skip part 2 (Prometheus). |
| 19 | [PR-599-Q4_0-Q4_1-native-experts.md](PR-599-Q4_0-Q4_1-native-experts.md) | Low: Q4_0 packs suit the 32 GB V100 more than 16 GB. Do when a plain-Q4_0 pack is on the roadmap. |
| 20 | [PR-478-Q6K-MMQ-prompt-kernels.md](PR-478-Q6K-MMQ-prompt-kernels.md) | Low: Q6_K is too large for useful 16 GB contexts; draft upstream with no numbers yet. |

### 4. Issues — investigate, do not port

| # | Item | Action |
| --- | --- | --- |
| 21 | [ISSUE-781-long-context-VRAM-stages-grow.md](ISSUE-781-long-context-VRAM-stages-grow.md) | Reproduce the stage split at 262K vs 524K on the V100s with `STRATA_DECODE_TIMING=1`; then apply the levers from items 8/3. |
| 22 | [ISSUE-486-2x-Turing-OOM-weight-arena.md](ISSUE-486-2x-Turing-OOM-weight-arena.md) | Covered by item 6 (#796). No separate work. |
| 23 | [ISSUE-771-Linux-arena-hugepages-slow-start.md](ISSUE-771-Linux-arena-hugepages-slow-start.md) | Workaround already exists (`STRATA_NO_LARGEPAGES`); measure once on the production rig. |
| 24 | [ISSUE-528-conversation-cache-decode-drop.md](ISSUE-528-conversation-cache-decode-drop.md) | Retest on the current build at ~100K conversations before any change. |
| 25 | [ISSUE-739-KVarN-KV-quantization-enhancement.md](ISSUE-739-KVarN-KV-quantization-enhancement.md) | Queue as research; do after items 1-7. |

### 5. Reviewed — already solved better in the fork, do not port

| File | Upstream | Fork's solution |
| --- | --- | --- |
| [REVIEWED-741-BF16-on-FP16-tensor-cores.md](REVIEWED-741-BF16-on-FP16-tensor-cores.md) | #741 | `STRATA_BF16_TC`, default-on for Volta |
| [REVIEWED-500-CPU-pool-quantization-barrier.md](REVIEWED-500-CPU-pool-quantization-barrier.md) | #500 | Quantization only for CPU-job tokens; no barrier wait |
| [REVIEWED-761-prefill-stage-pipelining.md](REVIEWED-761-prefill-stage-pipelining.md) | #761 | `next_run_` async hand-off in `prefill.cpp` |
| [REVIEWED-563-expert-cache-VRAM-release.md](REVIEWED-563-expert-cache-VRAM-release.md) | #563 | `--vram-elastic` + lazy vision, per GPU |
| [REVIEWED-726-live-expert-cache-resizing.md](REVIEWED-726-live-expert-cache-resizing.md) | #726 | `--vram-elastic` + `VRAM` command |
| [REVIEWED-390-multi-GPU-helper-expert-tier.md](REVIEWED-390-multi-GPU-helper-expert-tier.md) | #390 | Stage-owned weights automatic; helper tier measured slower |
| [REVIEWED-731-disjoint-adaptive-expert-duplicates.md](REVIEWED-731-disjoint-adaptive-expert-duplicates.md) | #731 | One expert, one owner per card |
| [REVIEWED-751-KV-persist-disk-LRU.md](REVIEWED-751-KV-persist-disk-LRU.md) | #751 | L3 disk store, longest-prefix restore |
| [REVIEWED-707-V100-community-bench-row.md](REVIEWED-707-V100-community-bench-row.md) | #707 | The fork's own V100 benchmark docs |

---

## Reference table

| File | Upstream | What | V100 benefit | Fork status | Verdict |
| --- | --- | --- | --- | --- | --- |
| [PR-800-Volta-prompt-attention-two-accumulator-chains.md](PR-800-Volta-prompt-attention-two-accumulator-chains.md) | PR #800 | Hi/lo accumulator chains in the m8n8k4 prompt attention | Error 5.9e-6 -> 3.0e-6, +2% time | absent | **port** |
| [PR-711-k8v4-KV-streaming.md](PR-711-k8v4-KV-streaming.md) | PR #711 (+ #705) | `--kv k8v4` streams with `--kv-resident` | More expert VRAM on 16 GB cards | absent | **port** |
| [PR-742-FP32-tiled-QSA-block-scores-below-sm80.md](PR-742-FP32-tiled-QSA-block-scores-below-sm80.md) | PR #742 | FP32 tiled block scorer below sm_80 | +22-33% long prefill (2080 Ti) | absent | test first |
| [PR-743-coalesced-streaming-topk-long-prompts.md](PR-743-coalesced-streaming-topk-long-prompts.md) | PR #743 (+ issue #669) | Coalesced streaming top-k | 250K top-k 12.8 -> 1.13 ms | absent | **port** |
| [PR-802-IQ1_S-GPU-and-AVX2.md](PR-802-IQ1_S-GPU-and-AVX2.md) | PR #802 | IQ1_S on GPU + IQ1_S/IQ1_M AVX-2 | Model fits 16 GB; CPU experts 7.7x | absent | **port** |
| [PR-796-startup-VRAM-plan.md](PR-796-startup-VRAM-plan.md) | PR #796 (+ issue #765) | One startup VRAM plan before cache commit | 524K config no longer dies | absent | **port** |
| [PR-792-zero-doorbell-batch-fix.md](PR-792-zero-doorbell-batch-fix.md) | PR #792 (fixes #776) | Batch window on 100% resident stage | `--batch` works at full residency | absent | **port** |
| [PR-783-fused-decode-verify-MTP-kernels.md](PR-783-fused-decode-verify-MTP-kernels.md) | PR #783 | Fused decode/verify/MTP kernels | Verify latency -13..-18% | partial | port + measure |
| [PR-789-prefill-overlap-shared-expert-uploads.md](PR-789-prefill-overlap-shared-expert-uploads.md) | PR #789 | Overlap shared expert with uploads | Short-chunk prefill waits | absent | **port** |
| [PR-699-Linux-read-ahead-at-startup.md](PR-699-Linux-read-ahead-at-startup.md) | PR #699 | Startup read-ahead on Linux | Cold start 920 s -> 70 s | partial | **port** |
| [PR-773-file-tier-unbuffered-reads-Linux.md](PR-773-file-tier-unbuffered-reads-Linux.md) | PR #773 | O_DIRECT file tier on Linux | Fast expert reads, less RAM | absent | **port** |
| [PR-733-pipeline-CPU-expert-work.md](PR-733-pipeline-CPU-expert-work.md) | PR #733 | Pipeline CPU expert Down per expert | CPU-expert decode overlap | absent | opt-in port |
| [PR-693-prefill-auto-finer-equal-chunks.md](PR-693-prefill-auto-finer-equal-chunks.md) | PR #693 | Finer auto chunks + equal chunks | +21-38% prefill from 20K | absent | test first |
| [PR-794-split-placement-per-stage-PCIe-link.md](PR-794-split-placement-per-stage-PCIe-link.md) | PR #794 | Auto split sees each PCIe link | x2-link GPU0 priced right | partial | **port** |
| [PR-732-overlap-streamed-KV-uploads-with-prefill.md](PR-732-overlap-streamed-KV-uploads-with-prefill.md) | PR #732 | Overlap KV uploads with prefill | Long-context streaming | absent | conditional |
| [PR-704-HC_Q8-int8-hyper-connections.md](PR-704-HC_Q8-int8-hyper-connections.md) | PR #704 | int8 hyper-connections | 525 MiB free per engine | absent | split-blocked |
| [PR-599-Q4_0-Q4_1-native-experts.md](PR-599-Q4_0-Q4_1-native-experts.md) | PR #599 | Q4_0/Q4_1 GPU kernels + PLE | Plain Q4_0 packs run | absent | low |
| [PR-478-Q6K-MMQ-prompt-kernels.md](PR-478-Q6K-MMQ-prompt-kernels.md) | PR #478 | Q6_K MMQ prompt kernels | Full K-quant MMQ coverage | absent | low |
| [PR-793-pipelined-windows-over-active-slots.md](PR-793-pipelined-windows-over-active-slots.md) | PR #793 | Trim batch windows to active slots | Batch pad-row waste | partial | low |
| [PR-378-elastic-KV-kv-grow.md](PR-378-elastic-KV-kv-grow.md) | PR #378 | Elastic KV (`--kv-grow`) | Cache keeps un-reached KV VRAM | absent | conditional |
| [ISSUE-781-long-context-VRAM-stages-grow.md](ISSUE-781-long-context-VRAM-stages-grow.md) | issue #781 | GDN/QSA VRAM hits grow 7-8x | Applies at 524K | n/a | investigate |
| [ISSUE-486-2x-Turing-OOM-weight-arena.md](ISSUE-486-2x-Turing-OOM-weight-arena.md) | issue #486 | 2x 2080 Ti 262K OOM | Same class as 2x V100 16 GB | n/a | monitor |
| [ISSUE-771-Linux-arena-hugepages-slow-start.md](ISSUE-771-Linux-arena-hugepages-slow-start.md) | issue #771 | MADV_HUGEPAGE 20x slower start | Linux arena fills | workaround present | low |
| [ISSUE-528-conversation-cache-decode-drop.md](ISSUE-528-conversation-cache-decode-drop.md) | issue #528 | 4-6x decode drop when caching | Agent workload on V100 | n/a | retest |
| [ISSUE-739-KVarN-KV-quantization-enhancement.md](ISSUE-739-KVarN-KV-quantization-enhancement.md) | issue #739 | KVarN KV quantization | KV VRAM on 16 GB | n/a | low |

## Reviewed and excluded (no file)

- **#803** (perplexity gap vs llama.cpp): output-quality/numerics, not speed. Out of scope.
- **#519 / fused prompt experts**: already shipped in the fork (moe_fused.cu), but the fused int8 kernels need sm_80+; they never run on V100 (`moe_fused.hpp`: "sm_80 and newer; sm_75 and HIP: MMQ").
- **Serve correctness PRs** (#787, #790, #762, #700, #525, #572, #454, #615, #652, ...): parser/API fixes, not performance.
- **AMD/HIP PRs** (#786, #766, #755, #720, #565, #778, #745, #758, ...): gfx11/gfx9 targets, not V100.
- **Benchmark-report PRs** (#698, #721, #674, #757, #777, #780, #791, ...): measurements, not improvements; the fork already publishes its own V100 benchmarks (docs/DETAILS.md, docs/NVIDIA_V100.md).
- **#689** (MTP/RTX PRO "buffer ownership"): PR body contains no concrete change or measurement; nothing verifiable to port.
- **Platform/serve features** (#768 Docker, #686 launcher, #680 web, ...): not V100 performance.
- **#691** (EcoQoS, Windows-only): this fork runs Linux.
- **#660, #625, #679** (vision CPU encoder/device role): vision already runs on the GPU encoder in the fork's V100 setup.
- **KV pointer**: the fork's production config keeps full GPU-resident KV at 262K/524K (PROGRESS.md), so KV-streaming items (#711, #732) only pay off once streaming is enabled.
