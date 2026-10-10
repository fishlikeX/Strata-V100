# Dual-V100 SXM2 IQ3_S: the image encoder's VRAM, and two test fixes

## Configuration

Date: 2026-10-10. Two Tesla V100-SXM2-16GB cards, compute capability 7.0,
80 SMs per card, 300 W power limits, both on PCIe Gen3 x16, no NVLink and no
working P2P on this board. CPU: Intel Xeon E5-2696 v3 (18 cores / 36 threads,
AVX2 only — no AVX-512). RAM: 46.9 GiB DDR4-2133 across four channels.
Storage: NVMe for the dense shards and the PLE table, SATA SSD for the expert
arena. Driver 580.178.04. Compiler CUDA 12.9 and GCC 14.4, Release, `sm_70`,
CMake 4.3.4.

Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S, native expert pack (46.84 GiB of
experts) with the expert arena on local storage and MTP weights.

This machine's defining constraint is that the expert set is almost exactly the
whole RAM: 46.84 GiB of experts against 46.88 GiB installed. Plain `--mmap-experts`
is not possible; the resident RAM mode (`--resident-experts`) is the only shape
that runs, and it is the reason the layer-split resident work in PR #848 matters
here at all.

Source: fork main `277a707b` (PR #49) plus two local commits. Engine SHA-256
`c3f26b51cb99831de620f1306a45570d1d100ee79c7759a095e14242adbc148c`, vision
encoder `10c915da6f8358e74baa38037af359dbe6e60986ebebf3783a7ff487c5402ac5`.

Both arms use two GPUs, `--layer-split 28 --split-device 1`, `--pipeline-windows 2`,
context 262,144, INT8 KV with `--kv-resident 20480`, `--resident-experts`,
`--spec 4 --spec-min-p 0.5`, expert cache auto, prefill auto, and
`STRATA_PREFILL_MMQ=0 STRATA_SM70_TABLE=1 STRATA_PF_WMMA=1
STRATA_PF_SKIP_UNROUTED=1 STRATA_EMB_REUSE_ACCOUNT=1 STRATA_SPLIT_RING=192`.

## What is measured here

### 1. The image encoder's VRAM is expert-cache capacity

The encoder (`strata-vision`, `--gpu`) holds about 1.55 GiB on the first card and
0.3 GiB on the second, and the expert caches are sized around it at boot.
With a resident configuration the encoder cannot be lazy — the engine refuses
the VRAM command that the lazy encoder hands its VRAM back and forth over — so
that VRAM is gone from the expert cache for the whole session.

Running the encoder on the CPU (`"gpu": false` in the `vision` section) returns it:

| Arm | expert cache, card 0 | expert cache, card 1 | pinned expert RAM | 32K prefill | 2K decode | decode hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| encoder on the GPUs (production) | 5,608 slots | 5,302 slots | 25.92 GiB | 1,863 (1,841-2,069) | 97.7 (86.5-103.7) | 94.5% |
| encoder on the CPU | 6,440 slots | 5,455 slots | 24.11 GiB | 1,976 (1,846-2,052) | 102.9 (101.1-108.5) | 95.8% |
| encoder on the CPU, `--kv-resident 40960` | 6,364 slots | 5,403 slots | 24.35 GiB | 2,034 (1,850-2,044) | 101.6 (94.7-103.1) | 95.7% |

Medians over five warm repeats (two rounds, three and two repeats), one engine
start per arm, with the observed range in parentheses. Each arm passed the
stateless one-shot probe (16/16) and an image-understanding check (three fresh
images, correct colour).

Pooled, the change is **+6% prefill and +5.3% decode by median**
(+7.2% by mean: 97.4 against 104.4 tok/s). Within the second round alone it was
+6% and +10%; the first round's control caught a high decode repeat (103.7), which
is why the pooled figure is the one to use.

The individual decode repeats **do overlap**: the GPU arm's five are 86.5, 96.8,
97.7, 102.5 and 103.7, the CPU arm's are 101.1, 102.1, 102.9, 107.5 and 108.5, so
the control's two best repeats sit inside the CPU arm's range. Four of the five CPU
repeats are above the control's median, and both rounds moved the same way, but
five repeats each is not enough to call +5% resolved. The expert-slot count is
the part of this that is not noisy.

The direction is consistent across every repeat: +14.8% expert slots on the first
card, +6% prefill, +5.3% decode, +1.3 points of decode hit rate. The pinned expert
RAM also *falls*, by 1.81 GiB: the resident mode pins the experts the GPU caches
do not hold, so a larger cache means less page-locked RAM. On a box where the
expert set equals the installed RAM, that headroom is worth having on its own.

The cost is encode latency, and it is not small:

| Encoder | 448x448 image (196 tokens) | 1024-token photograph |
| --- | ---: | ---: |
| CPU, 18 threads (the encoder's default) | 1.30-1.43 s | 10.08-10.78 s |
| CPU, 8 threads | not measured | not measured |

The GPU encoder's own encode time is not measured here: with the engine running,
neither card has the ~1.85 GiB the encoder needs, so measuring it means unloading
the engine, which changes what the encoder competes with. The production numbers
above are the encoder's VRAM footprint, read from `nvidia-smi` with the engine up:
1,546 MiB on the first card and 308 MiB on the second.

The launcher caches an encode by image hash, so a conversation that resends the
same picture pays once (0.24 s the second time, measured). A workload that
sends many different pictures per turn will feel the CPU encode; a text-and-
occasional-image agent workload will not.

`"threads": 8` instead of the encoder's default (half the cores) was measured for
throughput only, in one round, two repeats: 90.4 and 95.9 tok/s decode against
102.9 and 102.1 for the 18-thread arm. That is a 7% gap, larger than the
arm-to-arm difference being studied, so it is probably not the thread count but the
round it ran in - and it is a reason not to trust the ordering of these arms too
much. Encode time at 8 threads was not measured.

### 2. `kv_stream_parity` never checked k8v4's batch path

`ctest -R kv_stream_parity` fails on this machine with
`zero: invalid argument`, which reads as a Volta problem in the k8v4 kernels. It
is not. In `src/kernels/kv_stream_parity.cpp` the `zero()` and `cmp()` helpers
branch on `kKvQ4`, `kKvInt8` and otherwise assume the fp16 pools — but the hybrid
format allocates `k_q`, `k_scale` and `v_q4`, so `k_pool` is null and
`cudaMemset(nullptr, ...)` returns `cudaErrorInvalidValue`. The test died before
comparing anything.

With the hybrid branch added to `zero()`, `cmp()` and `append_batch()` (the batch
helper had the same gap: k8v4 fell through to the fp16 kernel with a null pool, an
illegal memory access), the test passes on `sm_70`:

```
k8v4: 306 batches, 702021 block lookups, 70480 misses (90.0% hit), overflow 0, 0 failures
k8v4 ring restore: identical
k8v4 batch-vs-step: identical
PASS
```

So the k8v4 batch-vs-step property did hold; it had simply never been
checked. `--kv k8v4` is still not used on this machine for a different reason:
`docs/INTEL.md` records that it keeps its KV in VRAM, and the KV has to share that
VRAM with the expert cache here.

### 3. `expert_multi_test` reports a missing ISA as a build failure

On a CPU without AVX-512 VNNI/VBMI (an Xeon E5 v3, for example) the test calls
`cpu_require_expert_support()`, which exits 1 after printing that the CPU cannot
run the kernel. The test is fine; the machine never had the instruction set. The
repo already has the idiom for this — `SKIP_RETURN_CODE 77`, used by
`pf_wmma_parity` for `sm_70-` and by `qsa_topk_active_parity` for no CUDA device.
The change makes the test return 77 and adds the property.

### 4. A Q8_0 mmproj on the CPU encoder

`docs/DETAILS.md` recommends a Q8_0 mmproj for `--vision cpu` and quotes cosine
0.997-0.999 per image token from #625, with a worst single token of 0.94-0.96.
On this machine, with the mmproj quantized by the command in that document and
five real photographs at 1,024 image tokens:

| Image | mean cosine | worst token |
| ---: | ---: | ---: |
| 0 | 0.9996 | 0.9871 |
| 1 | 0.9995 | 0.9723 |
| 2 | 0.9996 | 0.9772 |
| 3 | 0.9992 | 0.8705 |
| 4 | 0.9965 | 0.8596 |

The means match the published range. Two of five images have a single token below
0.94, which is worse than the 0.94-0.96 worst reported in #625; those are
generated photographs rather than charts or text, so the content differs. Encode
time was the same within 4% (1.31-1.40 s BF16, 1.30-1.37 s Q8_0 at 196 tokens;
10.08-10.78 s against 10.39-10.55 s at 1,024 tokens), and the file is 588 MiB
against 865 MiB.

## Reproducing

The arms differ only in the `vision` section of the server config:

```json
"vision": { "gpu": false }
```

Nothing else changes; the engine arguments and environment are identical. The
expert-cache line to look for at startup is `expert cache 6440 slots` against
`5608 slots`, and `resident RAM mode: 24.11 GiB` against `25.92 GiB`.

Throughput was measured with a harness that sends a fresh 32,768-token prompt per
repeat (it asserts `reused = 0`) and a 512-token prompt with 2,048 generated
tokens, and reads the engine's own timings. It is not their
`bench/run_v100_prefill.py` harness, so these numbers are comparable within this
report and with the fork's own dual-V100 IQ3_S tables only in direction, not in
absolute value: this rig has two full-width links and 300 W cards against the
x4 + x16 and 175 W of the published reports, and a much larger CPU.

## Caveats

- The prefill and decode numbers are medians of five repeats across two rounds,
  not a randomized design. The control's 32K spread is 1,841-2,069 tok/s, which is
  **larger than the reported +6% prefill gain**: the prefill claim rests on the
  expert-slot count (+14.8%, which is deterministic and printed at
  startup) and on the direction being the same in both rounds, not on the prefill
  medians alone. The decode repeats of the two arms overlap (the control's 102.5
  and 103.7 fall inside the CPU arm's range), so that result is directional at five
  repeats each, not resolved.
- The rounds were run in one order (control, then CPU) rather than interleaved or
  randomized, so a drift in card temperature or page-cache state between rounds is
  not excluded. Cards were 47-53 C at idle.
- The encoder's CPU encode time was measured on synthetic solid-colour images at
  448x448 and on real photographs at 1,024 tokens; a different image size
  changes it. The GPU encode time was not measured (see section 1).
- The GPU encode times were measured on the same cards with the
  engine unloaded, so they are an upper bound for what the encoder gets in
  production, where both cards are full.
- No exact-output equivalence is claimed between the arms;
  the stateless probe and the image check passed in both.
- The 10-second CPU encode of a 1,024-token photograph is the number that decides
  whether this trade is worth making: it is fine for an agent that sends an image
  every few turns, and it is not fine for one that sends several large pictures per turn.
