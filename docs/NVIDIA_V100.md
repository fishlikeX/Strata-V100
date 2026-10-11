# NVIDIA Tesla V100 / Titan V (Volta, sm_70): the community build

Strata's ready-made engine and the support matrix start at compute capability 7.5 (RTX 20). A V100 (7.0) runs the same
engine compiled for it, through the experimental build that issue #236 added. It is **community-tested, not part of the
supported cards**: no ready-made engine is published for it and the numbers below come from one machine.

## Build

CUDA 13 dropped Volta, so the build needs a **CUDA 12.x** toolkit (12.8 was used) and a host GCC that toolkit accepts.

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON
cmake --build build --target strata -j
```

Setup does the same by itself for a model on a V100 (the experimental CUDA 12 engine, in `engine-cuda12/`: ready-made
on Windows, compiled with a CUDA 12.x toolkit on Linux; [OLDER_GPUS.md](OLDER_GPUS.md)). For a PC with a V100 and a newer card, a build for both architectures (`-DCMAKE_CUDA_ARCHITECTURES="70;86"`) is the
natural choice (a community report in issue #509 ran an RTX 3080 beside a V100 that way; it was not repeated here). On the
measuring PC a Quadro RTX 4000 (sm_75) sits beside the V100s and was hidden with `CUDA_VISIBLE_DEVICES` because the sm_70-only
build has no code for it.

## What runs on Volta

The flag only lowers the floor (CMake and the runtime device check) and replaces two intrinsics Volta lacks. It does not
select "old" kernels everywhere: which kernel runs is decided per architecture when the engine is built.

| Part of the prompt path | On sm_70 |
| --- | --- |
| MoE experts (ggml MMQ) | `dp4a` kernels (Volta has no int8 tensor cores) |
| Dense projections (dequantized weights) | FP16 tensor-core GEMMs (cuBLAS / CUTLASS `s884`) |
| BF16 projections (hyper-connection, router, indexer, ...) | converted to FP16 and run on the FP16 tensor cores (#655, #540; `STRATA_BF16_TC=0`: cuBLAS BF16, an FP32 SIMT kernel on Volta) |
| QSA attention for decode and verify windows (and prompts with `STRATA_PROMPT_ATTN_OLD=1`) | #540's kernel (fewer shuffles, bit-exact; `STRATA_ATTN_PRE75=0`: the one other cards run) |
| QSA prompt attention, int8 / FP16 / K8V4 KV | `prompt_attn_v70_kernel` on `mma.m8n8k4` (`STRATA_PROMPT_ATTN_OLD=1`: the decode kernel, one query at a time) |
| QSA prompt attention, Q4_0 KV | the decode kernel |
| QSA block scores | the warp kernel (the tensor-core scorer needs sm_80) |

| Part of a decode / verify window | On sm_70 |
| --- | --- |
| Routed experts in VRAM | gfx906's expert mode 8: the codebook grid and the group's activations in shared memory, SwiGLU fused (`STRATA_EXP_MODE=0`: the CUDA layout other cards run) |
| Dense 2-4 column GEMVs and the head | the interleaved `native_mmvq_il` with a Volta rows table (`STRATA_MMVQ_IL=0`: `native_mmvq`) |
| Hyper-connection read, up projection | gfx906's latency-hidden up kernel (`STRATA_GR_FAST=0`: the plain one) |

All three are bitwise the kernels they replace; on a V100-SXM2 they take a verify window's GPU work from 22.8 to 21.2 ms
([bench/results/2026-10-07-v100-decode-kernels](../bench/results/2026-10-07-v100-decode-kernels/README.md)).

## Measured

One V100-PCIE-32GB, i9-7960X, Unsloth `UD-IQ4_XS` (not a setup model), int8 KV, MTP `--spec 2`; prompt read speed of random-word prompts, tokens/s,
with and without the Volta attention kernel (`STRATA_PROMPT_ATTN_OLD=1`):

| Prompt tokens | old attention path | `prompt_attn_v70_kernel` |
| --- | ---: | ---: |
| 7,194 | 1,010 | 1,164 |
| 28,650 | 1,063 | 1,251 |
| 114,338 | 973 | 1,123 |

Decode was 38-51 tok/s in every run and is not affected. The profile, the parity numbers, the needle test and the limits are in
[bench/results/2026-10-03-v100-prompt-attn](../bench/results/2026-10-03-v100-prompt-attn/README.md).

What the kernel does and does not guarantee: against an FP64 reference its error is about 2-3e-6 (output scale 3.6; the FP32 kernel it replaces: 2e-6; 5e-6 before the hi and lo halves got separate accumulator chains); 15 of 15
needles were found; greedy output was identical on the 28,650- and 114,338-token prompts and differed late in the answer on the 7,194-token one (another
summation order).

## Checking and benchmarking the Volta kernels

```sh
cmake -S . -B build -DSTRATA_PARITY_PROMPT_ATTN=ON ... && cmake --build build --target qsa_prompt_attn_parity
CUDA_VISIBLE_DEVICES=0 ./build/qsa_prompt_attn_parity 32768 2048 5     # synthetic, no model; exit 0 = PASS
```

It compares the tensor-core attention with the FP32 kernel and an FP64 host reference and times both. The Q4_0 cases are skipped below
sm_80.

## Limits

- A layer split over two V100 was not measured with this kernel. The cards work in turn on a single request, so a split adds expert-cache room, not prompt speed.
- sm_60 (Pascal) is covered by the same flag, and community runs exist: two Tesla P40 with a layer split (#1028, bench
  [2026-10-05-community-2x-p40](../bench/results/2026-10-05-community-2x-p40/README.md)), a Tesla P100 alone and with an RTX 2070 SUPER as
  the expert-helper card (#1069: 30.5 tok/s alone, 35.8-38.4 with the helper), and a V100 + P100 pair (#1079: +6-11% decode from 0.1.39 to
  0.1.40 with the P100 as helper cache). One reporter's numbers each; this page's own kernel measurements are V100 only.
- Volta's tensor cores take FP16 only: a model's BF16 weights are converted, and a weight outside FP16's range would saturate (none did in
  the check above).

## Selected v0.1.41 kernels

Set `STRATA_SM70_TABLE=1` to select the interleaved dense-matvec table,
native expert mode 8, and latency-hidden HC up projection on Volta.
The interleaved 2-4 column matvec with a Volta rows table and the
latency-hidden norm/up are bitwise the default kernels' output (GPU work
per verify window 22.8 -> 21.2 ms on a V100-SXM2 in the contributor's
measurement); they become the default once the contributor has confirmed
them on a V100 with the 0.1.41 code. The individual settings
`STRATA_MMVQ_IL`, `STRATA_EXP_MODE`, and `STRATA_GR_FAST` take precedence.
Without the master setting, Volta retains the previous decode selection
(what 0.1.40.3 ran).

Set `STRATA_EMB_REUSE_ACCOUNT=1` to let the prompt planner use the embedding
buffer space reused by half outputs. Default accounting remains conservative.
On the measured 8K chunk, this option reduces each stage's prompt loan by
about 80 MiB. It does not establish a general prompt-speed gain.

For the measured dual-V100 IQ3_S arena, `STRATA_PREFILL_CPU_SHARE=auto`
and `STRATA_PREFILL_STREAM_MIN=3072` select CPU-assisted staging below
3,072 tokens. The larger-chunk option can change prompt rounding.
Do not enable it for batch slots or assume the same gain on other models.

**Retained CPU and Volta changes.** These changes keep the model formats, the
int8 KV cache at a 524,288-token context, vision, MTP, the L2/L3 conversation
tiers and the NVMe-backed n-gram table. No 20% end-to-end gain is proved yet,
and no performance guarantee follows.

On sm_70 the prompt path's QSA top-k may use the active-block bound of a
prefill. The bound selects the register kernel when the prompt's active blocks
fit it, even when a long context capacity makes the score-row stride large.
Decode and captured graphs pass no bound and keep the capacity rule.

On a CPU with AVX2, no AVX-512 and no AVX-VNNI, the AVX-2 expert kernel serves
IQ3_XXS, IQ3_S and IQ2_S rows for one token (NT1) by default. NT means the
tokens in one group; NT1 is a group of one token. The IQ3_S and IQ2_S decodes
use 32-bit loads to read packed index bytes, which shortens the grid assembly.
These rows reduce in ggml's own float order and apply the format normalization
factor after the reduction, so a row is ggml's dot product bit for bit,
including subnormal products. IQ4_NL down rows are unchanged. An explicit
`STRATA_IQ_MT_MIN` keeps its rule.

After each non-host physical core has a worker, the pool pins additional
workers to the allowed SMT siblings of the worker cores. SMT (simultaneous
multithreading) means a physical core's second logical processor. The host
core's sibling comes last. The default placement is unchanged: five workers and
the host thread on the measured six-core CPU.

See the [release selection](../improvements/RELEASE-v0.1.41-V100.md)
and [prefill/decode measurements](../benchmarks/v100-iq3_s-v0141-2026-10-08.md).
The options do not change an installed service until its environment and
engine are changed. Do not enable stage-buffer pinning for IQ3_S.

## Shared-root return latency

The dual-V100 protected study measures repeat shared-root HTTP latency
at 1,799.7 ms before the cache changes and 902.6 ms after them.
The engine retains a disk-restored root in the bounded RAM cache.
The unrelated small-chat and live-continuation cases do not improve
in this sample. Do not treat this result as a general throughput gain.
See the [agentic latency measurements](../benchmarks/v100-iq3_s-prefill-wmma.md#agentic-short-turn-latency-2026-10-10).
