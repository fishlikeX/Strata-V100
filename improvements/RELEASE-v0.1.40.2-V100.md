# Release v0.1.40.2: V100 integration

## Source and scope

Review date: 2026-10-07. Source: [upstream release v0.1.40.2](https://github.com/Niko1221/Strata/releases/tag/v0.1.40.2).
The release tag points to `e8ca9af`.

The upstream repository changed its history. The fork at `d8a6925` and the new
release have no common ancestor. Do not reset this fork to upstream main.
Use selected changes. Keep the fork's Volta kernels, staged prefill pipeline,
expert-transfer ownership, cache handoff, and accepted VRAM plan.
This integration does not change the setup download version. An upstream binary
cannot replace the fork's locally built engine.

## Changes included

| Source commit | Change | Operating limit |
| --- | --- | --- |
| `08cdd8b` | When auto-split decode estimates differ by at most 0.000001 ms, select the placement with the shortest slowest stage. | Applies to auto split. Does not change the service's explicit split of 16. |
| `bc089d5`, `2a8cded` | Let prefill staging threads sleep on ring waits and DMA events. | Linux default. `STRATA_STAGER_SLEEP=0` selects the old waits; `1` selects sleeping waits. |
| `5208b32` | Use an AVX2 Q8_K activation quantizer for native CPU experts. Include byte-parity checks. | Requires AVX2. `STRATA_NO_Q8K_AVX2=1` selects the old quantizer. |
| `d6850f3` | Reduce index and sign decoding work in IQ3_S and IQ2_S AVX2 kernels. | Does not change the arithmetic. |
| `978d355`, `667f2ec`, `4322e24` | Let the CPU compute selected nonresident experts during short-prompt prefill. Capture source pointers before the worker starts. Exclude CPU-only scratch rows from NaN checks. | Opt-in, one GPU, no batch slots. |

The CPU path uses `STRATA_PREFILL_CPU_SHARE=auto` or a fraction. It applies to
chunks below 1,024 tokens. It selects experts with at most `MAXT` routed tokens.
The GPU computes the other experts. The combine step applies the original
routing weights to both sets of rows. `auto` adjusts the share from measured
CPU and GPU expert times. Unset or `0` keeps CPU sharing off.

The fork has no `ExpertSource::blob_stable` API. The port selects only pinned,
non-transient blobs and captures them with the existing `blob` API on the
prefill thread. The CPU worker makes no source calls. This keeps the source
read counter and staging state out of the worker thread.

Conflict resolution keeps the fork's ring-budget argument, compact tensor-core
scratch checks, and runtime VRAM contract checks.

## Local verification

Hardware: two Tesla V100 16 GB cards and a Ryzen 5 3600 CPU. Model: IQ3_S.
Build: CUDA 12.8, target `sm_70`, Release.

The engine and the focused CPU test targets built successfully. These checks passed:

- `iq_avx2_parity`.
- `iq_avx2_parity_iq3s_mt1`.
- `q8k_quant_parity`.
- `q8k_quant_parity_avx`.

`expert_multi_test` stopped at its AVX-512 capability check. This CPU has no
AVX-512. This result is not a passing test. The native IQ3_S runtime uses AVX2.

The Q8_K quantizer benchmark measured 3,250 ns per 2,560-value row for ggml
and 1,062 ns for AVX2, about 3.1 times faster. This is a kernel measurement,
not an end-to-end decode speed claim.

The direct engine protocol used identical seeded prompt IDs, greedy generation,
four output tokens, and no prompt reuse. Each arm ran once. NaN diagnostics
were enabled. The two-GPU arms used the service's explicit split and ring of
192. The one-GPU arms used GPU 1 and a 32K context. Do not compare the one-GPU
and two-GPU timings as a device-count benchmark.

| Runtime | Prompt tokens | CPU share | Prompt latency, ms |
| --- | ---: | --- | ---: |
| Existing two-GPU engine | 600 | Off | 5398.6 |
| Selected release changes, two GPUs | 600 | Off | 5278.9 |
| Existing two-GPU engine | 4096 | Off | 32596.9 |
| Selected release changes, two GPUs | 4096 | Off | 31741.6 |
| Selected release changes, one GPU | 256 | Off | 2068.4 |
| Selected release changes, one GPU | 256 | Auto | 1441.8 |
| Selected release changes, one GPU | 600 | Off | 4574.6 |
| Selected release changes, one GPU | 600 | Auto | 3853.5 |

All four compared output-token sequences matched. CPU sharing reduced prompt
latency by about 30.3% and 15.8% in the one-GPU smoke runs. CPU and GPU rounding
can still produce different answers on other prompts. The release reports this
limit too. The two-GPU timings are diagnostic evidence only. They do not prove
a stable performance gain. No full-context or full-suite result is claimed.

Additional runtime checks:

- Two-GPU auto split reached READY, selected layer boundary 25 for a 32K
  context, and completed a 256-token prompt with four generated tokens.
  This check exercises the search; it does not prove a tie occurred.
- Standalone one-GPU generation completed with CPU sharing on. Its prefill
  statistics reported 4,638 CPU expert evaluations and a measured share of 0.56.
  Residual, PLE history, and GDN state diagnostics reported zero non-finite values.
- The new engine was installed locally. The user service was restarted with
  its existing two-GPU configuration. The models API returned HTTP 200.
  A chat request returned `READY` with finish reason `stop`. A second request
  used 54 cached prompt tokens.
- The previous engine was saved outside the repository before installation.
  Local benchmark files and private service configuration were not committed.

## Service policy


Keep the current two-GPU service split and cache settings. Do not enable CPU
prefill sharing for that service: the upstream attachment condition excludes
multi-GPU execution. The fork's stages can overlap and share one non-reentrant
CPU expert pool. Removing that condition can cause concurrent pool use.
The new Linux staging wait policy requires no service environment change.

For a separate one-GPU runtime with no batch slots, set
`STRATA_PREFILL_CPU_SHARE=auto` to try CPU prefill sharing. Measure representative
chat and tool prompts before selecting it as a default. Remove the variable or
set it to `0` to restore GPU-only prefill.

## Next integration steps

1. Measure auto split against the explicit split with interleaved runs. Use
   matched GPU temperatures and cache budgets. The release's reported multi-GPU
   gain is from an unbalanced auto placement on AMD cards. It is not a forecast
   for the fork's asymmetric PCIe links.
2. Extend CPU sharing to staged prefill only with exclusive pool ownership or
   separate stage pools. Price extra CPU scratch memory. Check overlapping
   chunks, stage-helper use, batch decode, source lifetimes, and request failure.
   Then measure short tool turns on both V100s. The current port does not provide
   CPU-assisted two-GPU prefill.
3. Review late-stage dense-weight pricing (`e6ca0a0`) against the fork's accepted
   VRAM contract before porting it. Keep the fork's post-touch validation.
4. Test page-cache file reads and optional SSD read-ahead only for a RAM-limited
   or file-tier configuration. The current arena-backed service is not the
   low-RAM setup used for the release's drive-I/O gains.
5. Review pinned shared-prefix research support separately. It changes request
   and cache policy; it is not a multi-GPU kernel change.

Do not port the RTX 30-series interleaved MMVQ path as a V100 default. V100 is
`sm_70`, below that path's supported architecture. Intel Arc and AMD kernel
changes do not apply to this CUDA V100 service. Keep sampled-draft changes
separate: they change sampling behavior and have no verified V100 gain here.
