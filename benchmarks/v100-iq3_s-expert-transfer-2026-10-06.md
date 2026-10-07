# V100 IQ3_S expert transfer tests

## Purpose

Find a safe transfer setting for two Tesla V100-PCIE-16GB cards. GPU 0 has a
PCIe Gen3 x4 link. GPU 1 has a PCIe Gen3 x16 link. Keep both cards and the
524,288-token context capacity.

This report tests runtime settings. It does not change engine defaults,
kernels, the installed binary, or the private server configuration. The PR
contains the findings and the settings for review. Do not apply a setting
without a test on the required workload.

## System and method

- Model: Qwen3.8-Flash-Next-GSQ-RCO, native IQ3_S expert pack.
- GPUs: two Tesla V100-PCIE-16GB cards. No NVLink.
- CPU: AMD Ryzen 5 3600, six physical cores. RAM: 94 GiB as reported by Linux.
- Linux kernel: 7.0.0-34-generic. NVIDIA driver: 580.178.04. CUDA build: 12.8.
- Model storage: NVMe SSD, ext4. The expert arena is loaded before request timing.
- Both GPU power limits: 175 W. No power or clock setting was changed.
- Source commit: `077ebae6536772ea252521aa30fb1f6777b0084b`.
- Engine SHA-256: `903982948bdf67650c9474329dffcb915750e0bf31a043499b3dcd9ead2a38db`.
- Device order: GPU 0, then GPU 1. Layer split: 16. GPU 1 also holds the output
  head and the MTP draft layer.
- KV: INT8. Context: 524,288 tokens. RoPE: YaRN, scale 2. Vision reservation on.
- Expert cache: auto. Resident expert slots: 6,203 on GPU 0 and 2,841 on GPU 1.
- Host expert arena: 46.84 GiB, pinned. Five CPU pool workers plus the host thread.
- VRAM reserve: 750 MiB. Speculation: 8. Minimum draft probability: 0.70.
- Each arm starts a new engine. Prompt and conversation reuse are disabled.
- `bench/run_v100_prefill.py` sends exact token IDs. Seed: 617. Each prompt has
  4,096 or 32,768 tokens. Each request generates 128 tokens at temperature 0.
- Three repeats per length. Prompt hashes match across arms. Every row has zero
  reused tokens. Report median engine times, not HTTP wall times.
- The harness waits for GPU temperatures at or below 55 C for ten seconds before
  each request. During requests, the recorded maximum temperatures are 50 C on
  GPU 0 and 63 C on GPU 1. No thermal-slowdown samples were recorded. Power-cap
  samples can occur. These tests do not remove the card power limits.
- Direct-engine results and live HTTP results use different prompts. Do not
  compare their absolute rates as an A/B result.

The harness's strict-settings template is for an older Q2_0 configuration.
These IQ3_S arms use `--no-settings-strict`. The reserve is 750 MiB, not the
700 MiB template value. The last `--conversation-cache-mib 0` argument disables
reuse; the harness's first-value settings check also reports the earlier
15,360 MiB argument. The engine reports conversation cache 0 and disk store 0.

## Initial matched results

All rates below are tokens/s. The decode rate is generated tokens divided by
engine decode time. Small changes in decode rate are not evidence of a decode
improvement: the prompt changes can change output and draft acceptance.

| Arm | Ring slots | Prompt chunk | 4K prefill | 32K prefill | 4K decode | 32K decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 384 | 8,192 | 676.73 | 1,633.22 | 49.59 | 52.20 |
| Smaller ring | 96 | 8,192 | 700.16 | 1,556.62 | 49.25 | 52.05 |
| Balanced ring | 192 | 8,192 | 708.41 | 1,637.69 | 50.25 | 53.11 |
| Balanced ring, larger chunk ceiling | 192 | 10,752 | 713.46 | 1,501.63 | 52.70 | 54.86 |
| Balanced ring, `--adapt-swaps 32` | 192 | 8,192 | 710.01 | 1,640.26 | 41.98 | 48.10 |
| Balanced ring, `--pcie-mode dma` | 192 | 8,192 | 713.10 | 1,633.26 | 45.05 | 46.14 |

### Findings

1. A 192-slot ring improves initial 4K prefill by 4.68%. The initial 32K change
   is only +0.27%; treat it as unchanged, not a long-prompt speedup.
2. A 96-slot ring improves 4K but reduces 32K prefill by 4.69%. Do not use it as
   a general default.
3. A larger auto chunk ceiling selects 10,752 tokens, not 16,384. It gives up
   more resident experts and reduces 32K prefill by 8.31% against the 192-slot
   arm. Keep `--prefill auto` with its 8,192-token ceiling on this configuration.
4. Fewer adaptive swaps reduce decode throughput. Against the 192-slot arm,
   `--adapt-swaps 32` loses 16.46% at 4K and 9.43% at 32K. Keep the default 96
   swaps and four-window interval.
5. DMA decode is slower than the automatic in-graph copy mechanism. It loses
   10.35% at 4K and 13.12% at 32K against the 192-slot arm. Keep `--pcie-mode auto`.
6. Keep automatic per-card PCIe shares. The engine probes each link separately.
   An explicit `--pcie-frac` overrides both cards and removes that distinction.

### Confirmation

After all six initial arms and the first numerical check, repeat the default
and 192-slot arms in that order. Keep the same seed, lengths, and three repeats.

| Arm | 4K prefill | 32K prefill | 4K decode | 32K decode |
| --- | ---: | ---: | ---: | ---: |
| Default confirmation | 676.62 | 1,630.18 | 55.66 | 52.46 |
| 192-slot confirmation | 713.08 | 1,633.55 | 50.94 | 49.94 |

The confirmation gives +5.39% at 4K and +0.21% at 32K. The short-prompt
prefill gain is repeatable. The long-prompt change remains too small to claim.

Decode is not a controlled teacher-forced comparison. Even the unchanged
default's 4K median moves from 49.59 to 55.66 tokens/s between sessions. The
192-slot confirmation is slower than that default confirmation. Thus these
tests prove neither a decode gain nor absence of a decode regression. Use the
192-slot setting only as an opt-in short-prompt prefill candidate. Keep the
default when decode performance has priority.


## Why the ring matters

The ring is a GPU buffer for streamed experts. A slot holds a whole expert
blob. This IQ3_S pack has a largest blob of 2,662,400 bytes. A 384-slot ring
uses about 975 MiB per stage; a 192-slot ring uses about 488 MiB.

The prompt buffers borrow the tail of each stage's resident expert cache.
The smaller ring therefore leaves more experts resident during prefill:

| Ring | GPU 0 borrowed slots | GPU 1 borrowed slots | GPU 1 slots left resident | Loan per stage |
| --- | ---: | ---: | ---: | ---: |
| 384 | 2,341 | 2,071 | 770 | 4.12 GiB |
| 192 | 2,072 | 1,833 | 1,008 | 3.64 GiB |
| 96 | 1,938 | 1,712 | 1,129 | 3.40 GiB |

A smaller ring reduces the required transfers but also reduces transfer
lookahead. The 96-slot result shows the cost of too little lookahead. There is
no evidence here for a universal byte-budget or smallest-ring default.

The engine already uses pinned host memory, a separate copy stream, event
ordering, grouped gathers, and expert lookahead. These methods agree with the
[NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device).
The remaining problem is the balance between transfer buffers and resident
experts, not a missing asynchronous-copy flag.

The stage hand-off uses a single device residual buffer. Moving its copy to
another stream without a dependency would permit the next chunk to overwrite
that buffer. Keeping the dependency gives no useful device overlap. A second
device buffer would use more cache capacity. No hand-off code change is included.

## Candidate setting for review

For this IQ3_S configuration, test `STRATA_SPLIT_RING=192` in the run
configuration's existing `env` object. Keep its other entries:

```json
"env": {
  "STRATA_EXPERT_PAIR": "1",
  "STRATA_SPLIT_RING": "192"
}
```

The value takes effect when the engine starts. It applies to a layer split.
`STRATA_PREFILL_RING`, if set, takes precedence. `STRATA_SPLIT_RING=0` removes
the split override and uses the normal ring rule; it does not disable streaming.
Remove the new entry to return to the original configuration.

This is an opt-in prefill setting, not a demonstrated decode optimization.
The live configuration was unchanged during the benchmarks. The deployment
follow-up below records its later activation. No universal automatic ring
change is included. Do not extrapolate these measurements to Q2_0, another GPU, a
different context capacity, or prompts longer than the measured lengths.


## Reproduce

Use a private IQ3_S run configuration with the settings above. Do not copy an
API key into this report or a tracked file. Unload the live model before a
direct-engine arm. Load it again after the tests. Do not start the systemd unit
if a separate server already owns the port.

```sh
.venv/bin/python bench/run_v100_prefill.py \
  --config strata-iq3_s-arena.json --exe engine/strata \
  --arm-label ring-default --targets 4096,32768 --repeats 3 \
  --max-new 128 --seed 617 --no-settings-strict \
  --out bench/results/2026-10-06-expert-transfer/ring-default

.venv/bin/python bench/run_v100_prefill.py \
  --config strata-iq3_s-arena.json --exe engine/strata \
  --arm-label ring-192 --targets 4096,32768 --repeats 3 \
  --max-new 128 --seed 617 --no-settings-strict \
  --env STRATA_SPLIT_RING=192 \
  --out bench/results/2026-10-06-expert-transfer/ring-192
```

For the 96-slot arm, change only the ring environment value. For the larger
chunk arm, change `--prefill auto` to `--prefill auto:16384` in a private copy of
the configuration. For each decode arm, keep the 192-slot ring and add only
`--adapt-swaps 32` or `--pcie-mode dma`.

Do not require identical free-running output for a different resident/streamed
expert mix. Floating-point accumulation order can change. The first generated
token matches the default in all six requests for the 96-slot and 192-slot
arms, but the full sequences are not identical. This check alone does not prove
numerical equivalence or general model quality.

## Numerical check

`bench/check_v100_ring.py` starts three fresh serve engines: the default, a
second default run as a control, and the requested split ring. It uses the
same token IDs in each arm. A long head uses batched prefill and the expert
cache loan. A 600-token tail is then scored with `STRATA_LOGPOS` through verify
windows over the state that prefill produced.

The check sets `--prompt-cache 1` to enable the turn split, but disables the
other reuse tiers and requires zero reused tokens. It sets `--short-read 640`
so that only the tail uses the verify windows. It freezes adaptive swaps with
`--adapt-every 100000`. It requires progress for the batched head, 600
contiguous scored positions, and a matching target token at each position.

This checks the effect of batched prefill on the tail. It does not check
free-running decode quality or all prompt positions. `--dump-logits` is not
used: the native IQ3_S path does not reach that per-token dump site.

The published benchmark ran on both lengths with seed 617. All coverage and
ring-resolution checks passed. The engines reported rings 384, 384, and 192:

| Head length | Default vs control top-1 agreement | Default vs ring 192 top-1 agreement | Default tail PPL | Control tail PPL | Ring 192 tail PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 100% | 100% | 4.2667 | 4.2667 | 4.2667 |
| 32,768 | 99.1667% | 99.1667% | 1.1020 | 1.1161 | 1.1156 |

PPL means perplexity, from the actual target-token log probabilities. The
32K unchanged control is not identical. Thus the ring difference cannot be
separated from run-to-run numerical differences in this test. Do not claim
equivalence or an improvement in model quality. These are 600 tail positions
per head length, not a general quality evaluation.
The five changed top-1 positions at 32K are the same in the control and
candidate comparisons. This is not evidence of a ring-specific error.

No KL divergence is reported. The first exploratory probe used an invalid
top-k estimate with a residual probability bucket that could round to zero.
That estimate was removed from the published check.

```sh
.venv/bin/python bench/check_v100_ring.py \
  --config strata-iq3_s-arena.json --exe engine/strata \
  --target 4096 --seed 617 --ring 192 \
  --out bench/results/2026-10-06-expert-transfer/numerical-4k

.venv/bin/python bench/check_v100_ring.py \
  --config strata-iq3_s-arena.json --exe engine/strata \
  --target 32768 --seed 617 --ring 192 \
  --out bench/results/2026-10-06-expert-transfer/numerical-32k
```

## Evidence

The compact numeric evidence is in
[`expert-transfer-measurements.json`](../bench/results/2026-10-06-expert-transfer/expert-transfer-measurements.json).
It includes matched prompt hashes, per-request times, generated counts, output
hashes, cache-loan observations, and GPU statistics. Private configuration,
authentication data, and machine-specific paths are not included.
The numerical results and coverage checks are in
[`expert-ring-numerical.json`](../bench/results/2026-10-06-expert-transfer/expert-ring-numerical.json).

After the tests, the original server was loaded again. `/health` reported
`status: ok`, `loaded: true`, and context 524,288. An authenticated chat request
returned HTTP 200 and generated eight tokens. The installed engine hash did
not change. The systemd unit was inactive before the tests; it was not started
or modified. The tests used the separate IQ3_S server that was already running.

## Deployment follow-up (2026-10-06)

The installed `strata-v100.service` unit was updated after the measurements
above. It now starts the IQ3_S deployment: the description names IQ3_S, the
unit sets `STRATA_SPLIT_RING=192`, the server reads the private IQ3_S run
configuration, and a mount-point guard precedes the start.

The unit was enabled and started as a user service. The server and engine ran
in the unit's cgroup. The engine log identified the native IQ3_S pack and an
8,192-token chunk with a 192-slot ring. `/health` reported the 524,288-token
context loaded. An authenticated request read 4,852 fresh prompt tokens, with
zero reused tokens, and returned `OK`.

A normal `systemctl --user restart strata-v100` was also tested. The restarted
engine retained the 192-slot setting, health was loaded, and another generation
returned `OK`. The mount-point check exited successfully. The unit is enabled,
and lingering was already enabled, so startup does not require a login.
The model mount has a persistent filesystem entry. No reboot or power cycle
was performed.

The engine binary, kernels, and the expert routing context are unchanged from
the measured configuration. The historical result in this report stays as
measured: the systemd unit was inactive during the tests, and they used the
separate IQ3_S server that was already running. Process IDs and private
configuration values are not recorded here.
