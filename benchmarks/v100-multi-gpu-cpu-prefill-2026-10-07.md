# V100 multi-GPU CPU-assisted prefill

## Purpose

Allow each layer-split prefill stage to give selected short-chunk experts to
one shared CPU expert pool. Keep the GPU stages overlapped. Keep CPU assistance
opt-in with `STRATA_PREFILL_CPU_SHARE=auto` or a fraction. The default is off.

## Ownership and lifetime

Every eligible stage uses the existing decode pool. A mutex protects the complete
`run_split_multi_native` call: job and format pointers, gate/up work, activation
quantization, down work, statistics, and all scratch-capacity subbatches.
The mutex is not held during GPU work, source lookup, or stage-future waits.
The existing CPU-share timing includes time waiting for pool ownership. The
automatic share can therefore decrease when another stage uses the pool.

No extra expert pools or permanent worker sets are created. Each stage keeps
its existing CPU-share activation and result buffers. The public prefill run
drains all stages before decode starts. Batch-slot decoding and `--no-pool`
runtimes do not attach the pool. Other pool entry points must not overlap a
native batch; this change does not make all pool methods generally thread-safe.

CPU selection uses `ExpertSource::pinned_blob`, a const lookup that does not
change counters, file mappings, staging state, or prefetch state. The arena
uses its immutable registered memory. The file source uses its current pinned
resident ownership table. Source ownership changes and closure must wait for
prefill to drain. Unpinned and file-tier experts stay on the GPU. Sources that
do not support this contract return null and stay on the GPU.

CPU assistance still requires a native expert format and a chunk below 1,024
tokens. An active peer-expert path is excluded by the existing peer guard.
This is CPU assistance for a layer split, not a new CPU/GPU peer-compute mode.

## Regression evidence

Before the pool mutex was added, the concurrent native-batch regression exited
with signal 11. After the fix, it passed. The regression compares concurrent
results with serial results using different IQ formats, one- and two-token
jobs, and a batch of 97 experts that crosses the 96-expert scratch limit.
It repeats concurrent submission and then checks pool reuse.

The following focused checks passed:

- `pool_native_concurrency_test`.
- `iq_avx2_parity` and `iq_avx2_parity_iq3s_mt1`.
- `q8k_quant_parity` and `q8k_quant_parity_avx`.
- `file_expert_source_test`, with new pinned-view checks for ownership rotation,
  unpinned and GPU-only experts, invalid geometry, read accounting, and closure.
  The arena fixture also checks the registered-memory boundary.

Driver: 580.178.04. Linux kernel: 7.0.0-34-generic. The GPU power limit was
175 W per card. Temperatures and throttling were not sampled during the paired
arms, so thermal equivalence is not established. Prompt seed: 20261008.
The engine built with CUDA 12.8 for `sm_70`. No full-suite result is claimed.

## Performance experiment

Hardware: two V100 PCIe 16 GB cards and a Ryzen 5 3600 CPU. Model: native IQ3_S.
The runtime used the existing explicit layer split of 16, ring of 192, 524K
context capacity, and automatic prefill chunk size. Both arms used the same
candidate engine and configuration. The only arm change was CPU sharing:
`0` versus `auto`. Prompt caches and conversation reuse were disabled.

Three interleaved pairs ran in off/auto, auto/off, off/auto order. Each engine
read the same seeded prompt IDs at 256, 600, and 2,048 tokens. Generation requested
64 tokens after each prompt. Timing and NaN diagnostics were off for these arms.
The service was stopped to avoid GPU contention and restarted in a finally block.

| Prompt tokens | CPU off, median ms | CPU auto, median ms | Prompt latency reduction |
| ---: | ---: | ---: | ---: |
| 256 | 1667.5 | 1088.9 | 34.7% |
| 600 | 2456.0 | 1536.4 | 37.4% |
| 2048 | 4872.3 | 4879.4 | -0.15% |

These are local measurements for this configuration. They do not establish a
universal speed gain or a gain for full-context prompts. The 2K control was
approximately unchanged. Do not force small chunks to extend the short-prompt
speed claim to long prompts.

All nine paired first output tokens matched. The complete output sequences
were not identical. Automatic CPU sharing can change the allocation of experts
between CPU and GPU as timing changes. The CPU and GPU use different activation
formats and rounding. Some GPU-only repeated outputs also differed in this
experiment. No byte-identical output, deterministic automatic sharing, or model
quality equivalence is claimed. One 600-token auto run stopped after 39 generated
tokens; the other timing requests produced 64. The table measures prefill only,
not decode, so output length is not used to compute the reported gain.

The [public raw measurements](../bench/results/2026-10-07-multi-gpu-cpu-prefill/results.json)
include settings and output token IDs. The original local evidence remains in
`/tmp/mgpu-prefill-results.json`. Logs are local and are not published because
runtime paths can contain private data.

## Overlap and cancellation experiment

A separate diagnostic arm used `--prefill 512`, CPU sharing `auto`, timing,
and NaN checks. It read two distinct 2,048-token prompts and generated 64 tokens
for each. Both GPU stages reported CPU expert work. After the first prompt,
CUDA0 reported 2,532 cumulative CPU expert evaluations and CUDA1 reported 15,400.
These counts show that both stages used the CPU, not only the primary stage.

The forced small-chunk prompts took about 14.3 seconds with diagnostics. This
was a concurrency stress configuration, not a recommended long-prompt setting
or a speed comparison with the clean automatic-chunk arms.

A STOP request during a 4K prompt completed with finish reason `cancel`, zero
generated tokens, and 1,024 prompt tokens read. The same engine then completed
a fresh 256-token prompt and generated 32 tokens. Residual, PLE history, and GDN
state diagnostics reported zero non-finite values. No whole-buffer GU/H NaN
failure was reported. The test covers stage draining and pool reuse after
cancellation; it does not inject a GPU fault.


An additional 32K-context smoke used a fixed share of `0.25` and 512-token
chunks. A 1,536-token request completed with 32 generated tokens. CUDA0 reported
306 CPU expert evaluations and CUDA1 reported 4,535, both with share 0.25.
The same requested share with `--batch 2` completed a solo GEN request with
32 generated tokens; both stages reported zero CPU experts and share zero.
This verifies the batch attachment guard. It is not a concurrent BGEN benchmark.

## Local service and rollback

The tested candidate engine is installed locally. The existing user service is
active with its original two-GPU configuration. CPU assistance remains off by
default; no private service settings were changed. A live chat request returned
`READY` with finish reason `stop`. The running engine process hash matched the
tested build. The previous engine is saved at
`/tmp/strata-before-mgpu-cpu-prefill` for local rollback.

For an operator trial, add `Environment=STRATA_PREFILL_CPU_SHARE=auto` to a user
service override, reload the user service manager, and restart the service.
Remove that override to restore GPU-only prefill. Do not enable batch slots
for this trial. Keep automatic prefill chunk sizing and the existing layer split.
