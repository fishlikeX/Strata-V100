# Dual-V100 IQ3_S: upstream v0.1.42 sync

## Scope

This change syncs the fork with upstream `Niko1221/Strata` release v0.1.42.
The fork keeps its Volta (sm_70) work. This document records the merge, the
resolutions, the test results and the performance comparison.

Date: 2026-10-10. Two Tesla V100-PCIE-16GB cards, compute capability 7.0,
80 SMs per card, 175 W power limits, PCIe Gen3 x4 and x16.
CPU: Ryzen 5 3600, six cores and 12 threads. RAM: 94 GiB.
Driver: 580.178.04. Compiler: CUDA 12.8.93 and GCC 13.3, Release, `sm_70`.
Model: Qwen3.8-Flash-Next IQ3_S native GGUF shards on local storage.

## The upstream history rewrite

Upstream rewrote its git history after v0.1.40.1. Old and new commit hashes do
not match. The fork branched from the old history, so a plain merge has no
merge base and reports 1,787 upstream commits and 1,283 fork commits.

The tree of upstream v0.1.40.1 is the same in both histories. The tree hash is
`cc527a200c3f4f5bb06a63456b1b8936987b1071`. The merge uses that tree as the
merge base:

1. A graft made the old v0.1.40.1 commit (`20c2dd31`) a child of the new
   v0.1.40.1 commit (`82f46a8c`).
2. `git merge v0.1.42` then performed a three-way merge against that base.
3. The graft was removed after the commit. The merge commit records both
   parents, so later syncs need no graft.

`git rev-list v0.1.42..HEAD` reports 0. The merge contains every upstream
v0.1.42 commit.

## Conflict set

The merge changed 2,275 files against the fork's `main`. The upstream delta
adds 2,021 files. 263 files were modified on both sides and 35 of them
conflicted.

The 35 files are: CMakeLists.txt, 4 documents, 4 serve files, 3 web files,
2 tools, 10 headers, 8 CUDA and CPU kernel files, 2 prefill files,
generate.cpp, 3 core files and 2 SYCL files.

## Resolution rules

1. A file that only one side changed took that side.
2. Where both sides added different things, both were kept.
3. Where both sides changed the same lines:
   * upstream won for a general improvement, a new feature, a bug fix or a
     version bump;
   * the fork won for Volta (sm_70) work and for a documented fork feature.

Two subagents resolved the mechanical files. The kernel, prefill and
`generate.cpp` files were resolved by hand.

## Notable resolutions

* `include/strata/core/expert_source.hpp` and `.cpp`: upstream's version is
  1,403 lines ahead. The fork's `pinned_blob()` was kept as a small extension,
  and the fork's `expert_pool_dispatch_multi` fixes were re-applied (the
  published-plan test and the per-token activation test).
* `src/kernels/cuda/*`: the fork's Volta kernels were kept. Upstream's newer
  code was added beside them.
* `src/prefill/prefill.cpp`: upstream's `arm_cpu_share`, `cpu_share_max`,
  `g_stream_min_share`, `cpu_dead` and `copy_blobs` were kept. The fork's
  compact tensor-core walk, ring budget, `quant_act_rows` path and
  per-stage CPU assist were kept. `STAGE` is gone: the fork's `stage_slots()`
  and `m.ring` replace it.
* `src/program/generate.cpp`: upstream's layer-split startability gate, the
  batch read pieces and the pin/checkpoint work were kept. The fork's unified
  VRAM plan replaced upstream's inline cache sizing, because the plan also
  prices the prompt path and the two cannot run together.
* `bytes_needed()` takes both an explicit ring budget and a source flag. Five
  call sites were updated.

## Defects found by the comparison

Three defects came out of the comparison runs. All three are fixed in this PR.

1. **The expert cache opened at half its planned size.** The VRAM plan filled
   the per-pair slot list, and upstream's block then appended the same list a
   second time. The cache asked for 25.67 GiB where the plan said 12.83 GiB,
   failed the allocation and fell back to 6,185 slots against the plan's 7,332.
   The block now runs only when the plan left the list empty.
   Effect at the time: 15% fewer expert slots, -22% prefill at 2,048 tokens.
2. **The staging loop counted each slot down twice.** Upstream and the fork
   each added a `--pending` for the same slot at a different place, and the
   merge kept both. `pending` then wrapped, the loop stopped staging ahead, and
   the GPU waited for expert copies. The GPU-timeline profile showed
   `wait copy` on CUDA1 rising from 241 ms to 470 ms. One decrement was removed.
   Effect at the time: -4% to -10% prefill at 2,048 tokens.
3. **The CPU worker thread lost its affinity restore.** The thread inherited
   the GPU driver's single-core pin, so the CPU's experts were computed on one
   core. The `restore_thread_affinity` call was restored.

## Test results

`ctest` with free GPUs: 131 of 132 tests pass. `ple_parity` fails because the
Q2_0 fixture is absent on this machine. The unmerged tree fails the same test
in the same way.

The 3 skipped tests are `prefill_fused_moe_test`, `prefill_fused_iq_test` and
`expert_multi_test`.

## Performance comparison

Both arms use `bench/run_v100_prefill.py`, direct GENI protocol with an empty
embedding file, seed 20261010, three repeats per length, temperature zero, and
at most 128 output tokens. Each arm starts a separate engine. Prompt and
conversation caches are off. All rows report zero reused prompt tokens.

Both arms use the same settings as the user service:
`STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_PREFILL_STREAM_MIN=3072`,
`STRATA_EMB_REUSE_ACCOUNT=1`, `STRATA_SM70_TABLE=1`, `STRATA_PF_WMMA=1`,
`STRATA_PF_SKIP_UNROUTED=1`, `STRATA_KV_GROW=1`, and the configuration's
`STRATA_EXPERT_PAIR=1` and `STRATA_SPLIT_RING=192`.

Baseline: the fork's `main` (`4400773d`). Candidate: this merge.

The prompt lengths are matched between the arms: the same seed produces the
same prompts, so row N of one arm uses the same token ids as row N of the
other.

| Prompt | Repeat | Baseline ms | Candidate ms | Change |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 2613.1 | 2611.3 | -0.1% |
| 2,048 | 2 | 2390.9 | 2394.7 | +0.2% |
| 2,048 | 3 | 2356.1 | 2372.5 | +0.7% |
| 8,192 | 1 | 6219.7 | 6246.6 | +0.4% |
| 8,192 | 2 | 6301.5 | 6324.5 | +0.4% |
| 8,192 | 3 | 6309.2 | 6308.0 | -0.0% |
| 32,768 | 1 | 16472.7 | 16488.1 | +0.1% |
| 32,768 | 2 | 16547.2 | 16556.6 | +0.1% |
| 32,768 | 3 | 16512.5 | 16543.2 | +0.2% |

The mean difference is +0.3%, +0.3% and +0.1%. The two engines are the same
within the measurement spread. There is no regression at these lengths.

Compare only rows from one session. An earlier session measured the baseline
4% faster at 2,048 tokens; that session had the baseline first and a warmer
card, and the same binary measured 2613 ms against 2726 ms for the same prompt.

## A/B decisions

Where upstream changed something the fork had already done, both versions were
measured. The winner was kept.

| Change | Fork | Upstream | Kept |
| --- | --- | --- | --- |
| Block scores below sm_80 | FP32 tiled, on by default | opt-in with `STRATA_SELECT_SIMT=1` | fork |
| CPU assist across stages | every stage uses the pool | one stage at a time (`PoolHold`) | fork |
| CPU assist gating | every eligible layer shares | alternating arms, then a gate | fork |
| CPU assist candidates | page-locked blobs only | page-locked or resident | fork |
| Startup cache sizing | one VRAM plan | inline sizing with platform guards | fork |
| Stager copy | one blob per request | batched `copy_blobs` | upstream |
| KV trim | idle-time heuristic | `STRATA_KV_GROW_HOLD` | fork |

The CPU assist gating row was measured twice with the same prompt pool: upstream's
gate was 1.1% and 1.5% slower at 2,048 tokens and equal at 8,192 and 32,768. The
fork's rule was kept, and upstream's now-unused gate machinery was removed.

The stager keeps upstream's batched copy: the batch is 1 unless the source
reads files, so the fork's behavior is unchanged for a mapped pack.

## Limits

* Decode is not compared. Upstream v0.1.42 changes the default routing and the
  draft path, so the two arms generate different token counts for the same
  prompt. A decode rate over different output lengths is not a comparison.
* The comparison uses one prompt per length per repeat, three repeats. The
  per-prompt spread is larger than the mean difference.
* The upstream delta adds 2,021 files, most of them community benchmark
  records. They are not reviewed here.
* No long-context, vision or conversation-cache test was run.
