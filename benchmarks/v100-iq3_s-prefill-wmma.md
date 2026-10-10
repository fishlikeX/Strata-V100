# Dual-V100 IQ3_S prefill runtime changes

## Goal and fixed settings

Measure prefill at 2,048, 4,096, 8,192, 16,384, and 32,768 tokens against
merged fork PR 47. The performance target is a 20% increase. Do not reduce
the logical context capacity or change the KV format.

Technical terms: KV means key/value cache. WMMA means warp matrix multiply
and accumulate. VMM means virtual memory management.

The runs use two Tesla V100-PCIE-16GB GPUs, compute capability 7.0, and
layer boundary 16. The CPU is a Ryzen 5 3600 with six cores and 12 threads.
RAM capacity is 94 GiB. The build uses CUDA 12.8.93, GCC 13.3, Release,
and `sm_70`. The driver is 580.178.04. The benchmark interpreter is Python
3.12.3. GPU 0 uses PCIe Gen3 x4. GPU 1 uses PCIe Gen3 x16.

The model is the native IQ3_S expert arena. Keep logical context 524,288,
INT8 KV, YARN scale 2, MTP window 8, VRAM reserve 750 MiB, expert cache
auto, prefill auto, and five CPU workers. Keep `STRATA_EXPERT_PAIR=1` and
`STRATA_SPLIT_RING=192`. The expert arena stays in RAM. KV data stays on
the GPUs. Prompt cache, conversation cache, and disk cache are disabled.

## Baseline provenance

Use the existing PR 47 results. Do not run those baselines again.

- The archived 2K, 8K, and 32K baseline is
  `bench/results/2026-10-08-v100-v0141/selected.json`.
  Its engine SHA-256 is
  `333626841e54408bd72a0aa226485f2a0a25a2d3ed19dae07507184792c53894`.
  The harness checkout is `2a63d29848dea55738b663412532d605b409a866`.
  The source base is `1cb28c34c3769dbf9e55c2077c94f76729714089`, with the
  uncommitted selected v0.1.41 port documented in the PR 47 report.
- The supplementary 4K and 16K baseline uses the installed merged-PR-47
  engine, version 0.1.40. Its engine SHA-256 is
  `f3eaf1453df41c4a12b5f2387435a104585902c53d152a0dbd4b44f03c3dbb82`.
  Its source commit is `4400773dba46a4a149292c9aa301d8029e3a5b32`.

These are different baseline binaries. The five-size table is not a
measurement against one binary. Each candidate uses the matching baseline
prompt pool for its lengths. A binary hash identifies the tested executable;
a harness checkout commit does not identify uncommitted engine source.

## Method

Use `bench/run_v100_prefill.py`, direct GENI protocol, seed 20261008,
three repeats per length, temperature zero, an empty embedding file, and
at most 128 output tokens. Use two matrices:

1. `--targets 2048,8192,32768`: pool of 106,662 tokens.
2. `--targets 4096,16384`: pool of 69,860 tokens.

The prompt hashes of all 15 no-growth candidate requests match their
respective baseline hashes. Every request reports zero prompt reuse and
no request error. Start each arm in a separate process. Do not run two
engines, GPU tests, or compilation during a measured arm.

Use `--no-settings-strict`. The existing harness checks Q2_0 reserve 700,
not this IQ3_S reserve 750. Its conversation-cache argument check sees an
earlier duplicate argument. The final argument and engine status both
report cache zero. These warnings do not identify prompt reuse.

Report the median of three per-request prefill rates. Decode rate is
`1000 * generated / decode_ms`. Decode output length can differ after EOS.
These runs are an ablation sequence, not a randomized statistical study.

## Runtime changes

- Add grouped FP16-input, FP32-accumulator WMMA expert GEMM for Volta.
  `STRATA_PF_WMMA=1` selects the supported native expert formats.
  Unsupported paths keep their existing implementation.
- Reuse raw expert ring slots until down projection completes. Use device
  expert-pointer tables to avoid a separate weight gather buffer.
- Use shared lookup tables and packed half operations for the supported
  quantized weight decoders. Keep the existing weight formats.
- Remove unused quantized activation scratch on the WMMA path. Reuse gated
  projection scratch after its last read. At chunk 8,192, this saves about
  320 MiB from each stage's scratch requirement.
- Queue shared projection after router readback. CPU route grouping can
  then overlap the independent GPU projection.
- `STRATA_PF_SKIP_UNROUTED=1` omits a pinned expert transfer only after the
  current layer's route counts are published. Unknown routes still copy.
  Do not wait for routing or reduce issuer lookahead. Keep transient
  staging jobs and ring event ordering unchanged.
- Restore the original CPU affinity mask for auxiliary asynchronous work.
  Keep the main GPU host-thread pin. Quantize activation rows through the
  existing CPU pool. Keep the serial quantizer's output bytes unchanged.
- Extend optional GPU-only `--kv-grow` to distinct layer-split stages.
  Each controller uses its own GPU pools, cache, residency table, and
  layer bounds. Reserve virtual addresses for the full logical context.
  Map physical GPU memory as context grows. Do not move KV data to RAM.
  A VMM range retains its owning device and allocation granularity.

The Volta tuning guide describes FP16 Tensor Core inputs, FP32 accumulation,
and the shared-memory and register limits used for this kernel design:
<https://docs.nvidia.com/cuda/volta-tuning-guide/index.html>.
The CUDA VMM API defines device-specific allocation properties and
allocation granularity. Mapping alone does not grant access:
<https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html>.

## Profile without GPU-only KV growth

Keep the original full physical GPU KV allocation. Add these two options
to the PR 47 runtime profile:

```sh
STRATA_PREFILL_CPU_SHARE=auto
STRATA_PREFILL_STREAM_MIN=3072
STRATA_EMB_REUSE_ACCOUNT=1
STRATA_SM70_TABLE=1
STRATA_PF_WMMA=1
STRATA_PF_SKIP_UNROUTED=1
```

The tested engine SHA-256 is
`f5d1a3f7ecc21176711b414cbb989a0cec573773e9f18e5a8e389bfb642e67eb`.
It contains the runtime changes before the per-stage KV-growth extension.

| Prompt tokens | PR 47 prefill tok/s | Candidate prefill tok/s | Change | Candidate prefill ms | Candidate decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 700.96 | 728.70 | +3.96% | 2,810.5 | 61.20 |
| 4,096 | 719.10 | 816.46 | +13.54% | 5,016.8 | 56.43 |
| 8,192 | 1,009.56 | 1,178.37 | +16.72% | 6,952.0 | 59.64 |
| 16,384 | 1,367.48 | 1,595.56 | +16.68% | 10,268.5 | 56.34 |
| 32,768 | 1,660.57 | 1,915.05 | +15.32% | 17,110.8 | 53.83 |

The 8K third repeat ends after 10 output tokens. The other 14 requests
produce 128 output tokens. Do not treat these decode medians as a
fixed-output-length comparison. This no-growth profile does not meet 20%.

## Initial GPU-only growth measurements

Add `--kv-grow` to the private engine arguments. Keep all other profile
settings unchanged. Physical GPU KV allocation starts at 16,384 cells.
The logical context capacity stays 524,288. The tested engine SHA-256 is
`1912b0731eddeb6b2e9ac81ec00c1de379c9ebcdc731441f5b104fe0491acba9`.
This run precedes the final guard that skips VMM capability queries when
KV growth is disabled.

| Prompt tokens | PR 47 prefill tok/s | Candidate prefill tok/s | Change | Candidate prefill ms | Candidate decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 700.96 | 859.71 | +22.65% | 2,382.2 | 66.68 |
| 4,096 | 719.10 | 1,018.22 | +41.60% | 4,022.7 | 60.11 |
| 8,192 | 1,009.56 | 1,302.70 | +29.04% | 6,288.5 | 63.65 |
| 16,384 | 1,367.48 | 1,706.86 | +24.82% | 9,598.9 | 67.95 |
| 32,768 | 1,660.57 | 2,003.40 | +20.65% | 16,356.2 | 65.16 |

All 15 prompt hashes match their baseline hashes. Every request completes
without an error or prompt reuse. The 8K third repeat generates 10 tokens;
the other requests generate 128. Startup takes 43.9 s and 44.1 s for the
two matrices. The measured prefill increase exceeds 20% at each length.
This is not a guarantee for a different workload or hardware.

The initial primary expert cache increases from 6,199 to 7,333 slots.
The secondary cache increases from 2,836 to 4,843 slots. During the 32K
requests, both devices grow their KV pools to 40,960 cells. GPU 0 releases
57 cache slots; GPU 1 releases 96. Both changes stay below the prefill
loan boundaries. See `gpu-growth-evidence.json` in the raw-result directory.
The largest measured request is 32,768 tokens, not 524,288 tokens.

## Selected final-source profile

Use the same GPU-only growth profile on the final runtime source. The
engine SHA-256 is
`e930ad63cd26b4442f4ab03c44fc00aee79fda304d93c9561283f596be41b631`.
All 15 requests match their baseline prompt hashes. All finish without a
request error or prompt reuse. Both matrices use this same executable.

| Prompt tokens | PR 47 prefill tok/s | Selected prefill tok/s | Change | Selected prefill ms | Selected decode tok/s | Request wall s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 700.96 | 858.88 | +22.53% | 2,384.5 | 58.49 | 4.484 |
| 4,096 | 719.10 | 1,016.81 | +41.40% | 4,028.3 | 60.23 | 6.162 |
| 8,192 | 1,009.56 | 1,305.73 | +29.34% | 6,273.9 | 65.36 | 8.210 |
| 16,384 | 1,367.48 | 1,705.10 | +24.69% | 9,608.8 | 71.76 | 11.395 |
| 32,768 | 1,660.57 | 2,003.57 | +20.66% | 16,354.8 | 63.36 | 18.386 |

The selected profile exceeds the 20% prefill target at all five lengths.
Request wall time includes generation and protocol work. It is not
prefill-only latency. The 8K third repeat generates 10 tokens; the other
14 requests generate 128. Decode and wall-time medians are observations,
not exact-output speed comparisons.

Startup takes 44.1 s and 43.9 s. The prefill improvement is not a claim of
faster cold engine startup. The sampler records 141 samples per GPU.
Maximum temperatures are 48 C and 58 C. Thermal slowdown sample counts
are zero. Maximum instantaneous draws are 191.30 W and 212.35 W.

### Reproduce the selected profile

Use a private copy of the original arena configuration. Add `--kv-grow`
to its engine argument list. Do not change its context, KV format, model,
split, reserve, ring, or worker settings. Do not publish credentials.
Set `PYTHON` to the benchmark interpreter and `PRIVATE_GROW_CONFIG` to
that private configuration.

```sh
cmake --build build-v100-prefill-opt --target strata -j 6

"$PYTHON" bench/run_v100_prefill.py --config "$PRIVATE_GROW_CONFIG" \
  --exe build-v100-prefill-opt/strata --arm-label selected-final \
  --targets 2048,8192,32768 --repeats 3 --seed 20261008 --max-new 128 \
  --no-settings-strict --env STRATA_PREFILL_CPU_SHARE=auto \
  --env STRATA_PREFILL_STREAM_MIN=3072 --env STRATA_EMB_REUSE_ACCOUNT=1 \
  --env STRATA_SM70_TABLE=1 --env STRATA_PF_WMMA=1 \
  --env STRATA_PF_SKIP_UNROUTED=1 --out /tmp/v100-selected-final

"$PYTHON" bench/run_v100_prefill.py --config "$PRIVATE_GROW_CONFIG" \
  --exe build-v100-prefill-opt/strata --arm-label selected-final-missing \
  --targets 4096,16384 --repeats 3 --seed 20261008 --max-new 128 \
  --no-settings-strict --env STRATA_PREFILL_CPU_SHARE=auto \
  --env STRATA_PREFILL_STREAM_MIN=3072 --env STRATA_EMB_REUSE_ACCOUNT=1 \
  --env STRATA_SM70_TABLE=1 --env STRATA_PF_WMMA=1 \
  --env STRATA_PF_SKIP_UNROUTED=1 --out /tmp/v100-selected-final-missing
```

## Rejected parameter changes

All values below are medians from the matched 2K/8K/32K matrix.

| Change from the no-growth profile | 2K tok/s | 8K tok/s | 32K tok/s | Decision |
| --- | ---: | ---: | ---: | --- |
| Ring 128 instead of 192 | 718.80 | 1,171.01 | 1,839.96 | Keep ring 192. |
| Eleven CPU workers instead of five | 709.88 | 1,178.13 | 1,916.55 | Keep five workers. |

The worker change reduces 2K throughput and does not materially improve
8K. The small 32K difference does not justify changing the worker count.

## Verification and limits

The integrated CUDA build passed seven focused CTest checks: `vmm_test`,
`vmm_device_local_test`, `iq_parity_fixtures`, `iq_parity`,
`pf_wmma_parity`, `pool_native_concurrency_test`, and
`cpu_pool_quant_rows_test`. The new VMM test checks physical allocation
locality on both GPUs and data preservation after unmap/remap. The WMMA
test checks direct and indirect expert inputs, sparse row IDs, unequal
expert row counts, and output guards. The activation-row test checks
byte equality with the serial quantizer, including skipped and guard rows.

These checks are not a full-suite result or a model-quality assessment.
WMMA uses FP16 activations instead of the previous Q8 activation path.
Greedy token sequences can differ. The archived baseline also differs
between repeats. Do not claim bit-exact model output.

Across the 15 no-growth requests, the sampler records 155 samples per GPU.
GPU 0 reaches 46 C; GPU 1 reaches 56 C. Neither GPU reports a thermal
slowdown sample. Recorded maximum instantaneous draws are 210.59 W and
227.79 W. A separate current device query reports 175 W power limits.
The raw baseline does not independently record historical power limits.
Do not infer a historical limit from this current query.

## GPU-growth transition checks

Use the final binary for two diagnostic long, short, long sequences:
32,768, 8,192, and 32,768 tokens. Each request generates at most one token.
The prompt pool and output cap differ from the selected matrix. Do not
compare these rates with the baseline table.

- With requested initial cells 512 and cache retention floor 100,000,
  allocation granularity gives an initial 4,096 cells. Both GPUs grow to
  40,960 cells using new GPU memory, with zero cache slots released.
  This high-floor case produces no shrink record.
- With the normal profile, both GPUs grow to 40,960 cells. GPU 0 releases
  57 cache slots; GPU 1 releases 96. The short request trims both pools
  to 16,384 cells and refills those slots from the profile. The next long
  request grows both pools to 40,960 cells again.

All six requests complete without errors or prompt reuse. Within each
sequence, the repeated long prompt has the same hash and first output
token. This is not a full-output quality result. The raw directory contains
the request records and exact growth log excerpts for both checks.

## Original service restoration

The original service is active, running, and enabled after the measurements.
Its installed engine is unchanged, with SHA-256
`f3eaf1453df41c4a12b5f2387435a104585902c53d152a0dbd4b44f03c3dbb82`.
No private service configuration or installed executable is replaced.
The selected runtime profile is an opt-in setting for the candidate build.

The restored service passes authenticated HTTP checks. Health, model
discovery, and chat return status 200. The loaded engine reports context
524,288 and INT8 KV. The short chat request returns `Hi`, with two output
tokens. This is a functional restoration check, not a baseline benchmark.

## Raw evidence

See the [raw-result index](../bench/results/2026-10-09-v100-prefill-wmma/README.md)
for the selected measurements, baseline provenance, parameter ablations,
and transition checks. Published records omit private credentials and home
paths. The source-state fields identify uncommitted builds; binary hashes
identify the tested executables.

## CPU and SSD study, 2026-10-10

This study uses a later baseline than the preceding study. Its installed
baseline engine has SHA-256
`c6e0694c7b3c65709e7c298eb55fdfc5be4d719275da38dafabb19823db53281`.
Do not add the gains from the preceding study to these results.

The machine has two 16 GiB V100 GPUs. GPU 0 uses PCIe Gen3 x4.
GPU 1 uses PCIe Gen3 x16. The CPU is a Ryzen 5 3600 with six physical
cores and 12 logical processors. The GPU power limit stays at 175 W.
The model stays IQ3_S, with its existing mixed expert formats.

### Protected HTTP comparison

Run three repeats at each target, with seed 20261010 and 256 output
tokens. Start each arm with a fresh test-owned conversation directory
on the same NVMe device. Stop the service for isolated measurements.
Do not run builds or other benchmarks during an arm.

Both arms keep INT8 KV, context capacity 524,288, and KV growth.
They keep MTP window 8 and minimum probability 0.70, image support,
the 15,360 MiB RAM conversation cache with 32 slots, and the 50 GiB
NVMe conversation cache. The PLE table stays on NVMe. These capacity
settings do not mean that this study tests a 524,288-token prompt.

All 18 requests generate 256 tokens. All requests have zero prefix reuse.
The table shows the median rate in tokens per second.

| Target | Baseline prefill | Candidate prefill | Change | Baseline decode | Candidate decode | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 558.6 | 599.1 | +7.3% | 62.7 | 59.0 | -5.9% |
| 8,192 | 1,134.7 | 1,168.4 | +3.0% | 61.6 | 63.5 | +3.1% |
| 32,768 | 1,832.3 | 1,729.2 | -5.6% | 57.6 | 64.2 | +11.5% |

**This candidate does not meet the 20% prefill and decode target.**
It is not a general throughput improvement. Three repeats do not give
a confidence interval for small changes. Do not use a selected context
or a selected repeat to claim that the target is met.

### Retained changes and rejected experiments

The candidate makes selected CPU gate/up formats match the GGML AVX2
reference. It uses packed index reads for IQ2_S and IQ3_S. It uses
single-token CPU dispatch by default only on the selected AVX2 path.
An explicit user threshold still takes precedence.

The CPU pool places overflow workers on allowed SMT siblings.
The main GPU host thread stays pinned. The two background conversation
writers restore the original allowed CPU set before they write to NVMe.
The PLE reader copies a row across two cached pages without another SSD
read. A partial cache hit leaves the output unchanged until the SSD read.
The QSA prefill selector can use active block bounds on SM70.
Decode graph selection keeps its capacity-based rule.

Measure total CPU compute budgets from 6 through 10. The direct protocol
disables conversation caching, so its results are supplementary.
Keep the default budget of six: five workers and the host thread.
No tested budget gives a consistent 20% gain in both phases.
Some arms stop before 256 output tokens. Mark these arms as not comparable
at a fixed output length.

Reject the larger WMMA tile, vector staging, smaller half-block tile,
secondary PCIe-share overrides, changed layer split, changed stream
threshold, larger PLE row cache, and disabled WMMA experiments.
Remove their temporary implementation changes. Keep the existing WMMA,
cache budgets, layer split, and automatic PCIe-share policy.

### Correctness evidence

The expanded CPU reference test fails in nine groups when linked to the
original CPU library. The candidate passes the CPU parity tests.
The cases cover token-group widths 1 through 8, cancellation, and
subnormal products. The CPU affinity test passes sparse allowed masks
and both host-reservation modes.

An actual RAM-cache A/B/A smoke restores 4,615 of 4,620 prompt tokens
with both the baseline and candidate. A syscall trace observes two NVMe
writers in each arm. The original writers inherit CPU mask `[0]` and
do not restore it. Both candidate writers successfully restore `[0..11]`
before their conversation-file open. The file rename operations succeed.
These trace results prove the affinity change. They are not timing evidence.

The complete built CTest run has 100 passes, two skips, and three failures.
The skips require newer GPU instructions. One failure needs a missing
legacy Q2_0 PLE fixture. One requires AVX512 instructions that this CPU
does not have. The third is a K8V4 test-harness defect: the test uses
the FP16 zeroing branch for hybrid pools and passes a null pointer to
`cudaMemset`. Its hybrid batch-append and comparison helpers also lack
hybrid branches. The protected INT8 streamed-KV cases pass 306 random
batches, ring restore, and batch-versus-step checks bitwise. No KV
runtime change is made for this unrelated test defect. Do not report
the full suite as passed.

The real-model native expert smoke passes at layers 0, 16, 20, 24,
28, 32, and 47. It covers IQ3_XXS, IQ2_S, IQ3_S and IQ4_XS gate/up
weights, with IQ4_NL and Q2_0 down weights. The selected gate/up
formats match the GGML dot reference. One-token and three-token
gate/up and down outputs have no differing rows in this smoke.
Full expert operations pass the existing tolerance; they are not a
bit-exact model-output test.

The PLE selftest also passes on the production NVMe mount, in an owned
temporary directory. It checks both 90-byte and 110-byte rows, with
IQ4_NL, Q5_0, F8_E4M3, Q5_1, Q8_0, Q4_0 and BF16 dequantizers.

The installed candidate identifies a generated solid-red PNG as `Red`.
The text question does not name the color. The request generates 112
tokens, with 111 speculative drafts offered and 81 accepted.
The restart smoke restores 4,615 tokens from NVMe. The following A/B/A
smoke records two RAM restores. The final NVMe counters show two restores
and 9,230 restored tokens, with zero corruptions and zero write failures.
The service reports context 524,288, INT8 KV, image support, MTP
window 8, minimum probability 0.70, and the unchanged cache budgets.
Startup reports elastic KV on both GPUs: 16,384 resident cells out of
524,288 logical cells each. The PLE reader keeps the NVMe SSD awake
with page reads. It does not use a whole-table RAM override.

The installed engine has SHA-256
`c2256dd20f00df2bde1def55b8c33b01566a67d68a8ccea0789b1cb451a362c6`.
The original executable is saved outside the repository. The candidate
service is active after the checks.

See the [machine-readable measurements](../bench/results/2026-10-10-v100-cpu-ssd/measurements.json)
for the direct arms, fixed-output comparison flags, protected HTTP
records, and verification evidence. The published data has no private
configuration, credentials, request text, or home paths.

