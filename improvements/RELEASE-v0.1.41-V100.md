# Release v0.1.41: selected V100 changes

## Source and scope

Review date: 2026-10-08. Source: [upstream release v0.1.41](https://github.com/Niko1221/Strata/releases/tag/v0.1.41), commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
The fork baseline is `1cb28c34c3769dbf9e55c2077c94f76729714089`.
The current benchmark-only branch has the same engine source as this baseline.
The previous upstream integration used `e8ca9afd` (v0.1.40.2).
Use selected source changes. Do not replace the fork with an upstream engine.
Keep the fork's Volta attention, expert-transfer ownership, CPU-pool serialization,
and VRAM planner.

## Included source changes

| Upstream source | Change | Operating limit |
| --- | --- | --- |
| `ffac97f7`, `2d6c6ae5`, `e509ba5e`, `c85b7c87` | Interleaved Q8_1 activation storage and a Volta dense-matvec row table. Update the verify-window caller and scratch storage. | Volta requires `STRATA_SM70_TABLE=1`. Supported dense weight formats and windows of 2–4 tokens only. Other calls retain the old path. |
| `f6330507`, `c85b7c87` | Native expert mode 8 on CUDA Volta. | Volta requires `STRATA_SM70_TABLE=1`. An explicit `STRATA_EXP_MODE` takes precedence. |
| `aaa323fe`, `c85b7c87` | Use the latency-hidden HC up projection on Volta. Preserve the fork's norm-fused down projection. | Volta requires `STRATA_SM70_TABLE=1`. An explicit `STRATA_GR_FAST` takes precedence. |
| `ccd657a2` | Retain HC norm products in existing scratch. | The V100 default plain HC path does not use this split-norm kernel. |
| `199fd1d5`, `55dee8ed` | Reuse the dead embedding buffer for half outputs. Keep the previous planner accounting by default. | `STRATA_EMB_REUSE_ACCOUNT=1` permits the planner to use the saved space. Different cache loans or chunks can change rounding. |
| `83a7dbc0` | Select the RAM staging profile when at least 90% of sampled mapped GGUF expert pages are resident. | Linux mapped GGUF sources only. Unknown residency and direct reads retain the SSD profile. Explicit staging settings take precedence. |

The residency port uses the OS page size and retains one sample-vector allocation.
It does not classify direct reads as RAM copies. The normal arena configuration
has no mapped GGUF expert source, so this change does not affect the arena timings.

## Larger-chunk CPU assistance

Upstream `92bce20e` extends staged CPU-assisted chunks. Upstream later keeps this
extension optional (`162ce64e`). The fork already has a common planner and runtime
threshold: `STRATA_PREFILL_STREAM_MIN`. Do not add a second threshold API.

With `STRATA_PREFILL_CPU_SHARE=auto`, set `STRATA_PREFILL_STREAM_MIN=3072` to test
the staged CPU-assisted path for a 2,048-token prompt. The limit is exclusive.
The fork retains its immutable source views and serialized multi-GPU CPU pool.
Batch slots remain excluded from CPU assistance.

A matched three-repeat 2K comparison measured 419.80 versus 676.49 prompt tok/s
(+61.1%). Median decode rates were 44.59 versus 44.65 tok/s. These prompts differ
from the full-length matrix because the existing harness selects offsets from a
pool whose length depends on the requested target set. Compare only rows with
identical prompt hashes. This result does not establish a gain at every length.

## Changes not selected

- Stage-buffer pinning: the v0.1.41 release gate reports IQ3_S corruption after a
  long prompt. Do not enable `STRATA_STAGE_PIN` for this model.
- Chunked GDN: the upstream gate requires compute capability 8 or newer and at
  least 128 SMs. These V100 cards have compute capability 7.0 and 80 SMs. The
  forced SM-count option does not remove the architecture requirement.
- A Volta key-head recurrence experiment: not part of the released change set.
  The fork already uses a Volta fast recurrence with a different accumulation
  order. No experiment code is included.
- Prompt IQ tensor-core WMMA from upstream PR 1401: not in the v0.1.41 tag.
- No-P2P expert sums and residency-biased routing: optional changes can alter
  output or reduce decode speed. Not included.
- Multi-GPU batch-group defaults: not measured here. The current service uses one
  slot. No concurrency-default change is included.
- CPU-share defaults: the fork already supports its dual-GPU configuration.
  Do not replace that support with upstream's single-GPU default policy.
- AMD, Intel, and Pascal-only changes: not applicable to this hardware.

## Verification and limits

See the [benchmark report](../benchmarks/v100-iq3_s-v0141-2026-10-08.md) and its raw
result files. The Release build targets `sm_70` with CUDA 12.8.

The following focused tests passed with `STRATA_SM70_TABLE=1`:
`file_expert_source_test`, `expert_layout_test`, `mmvq_il_parity`, `gr_parity`,
`native_multi_parity`, and `gr_multi_parity`.
`native_grouped_parity` passed with `STRATA_EXP_MODE=8`.
`fused_gr_bench 50 1 8` reported zero differing values in all 32 configurations.
A real-model expert benchmark on layers 0, 15, 16, and 47 reported zero differing
values between modes 0 and 8. These are kernel parity results, not a model-quality
or full-suite claim.

The first residency assertion used the wrong fixture and failed. It was moved
from an experts.bin fixture to the actual mapped-GGUF fixture. Both source tests
then passed. The final port also excludes direct reads from residency detection.

The same baseline engine produced different greedy outputs on eight of nine
repeated prompts. Thus these end-to-end measurements do not establish byte-exact
output or a quality result. No service binary or private service setting is
changed by this PR. The user service must retain its existing binary until the
user selects and installs a reviewed build.
