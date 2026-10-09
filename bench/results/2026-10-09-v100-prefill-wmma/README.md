# Dual-V100 prefill runtime evidence

The [benchmark report](../../../benchmarks/v100-iq3_s-prefill-wmma.md)
contains the fixed settings, method, measured results, and limits.

## Selected final source

- [2K, 8K, and 32K](selected-2k-8k-32k.json)
- [4K and 16K](selected-4k-16k.json)

Both files use engine SHA-256
`e930ad63cd26b4442f4ab03c44fc00aee79fda304d93c9561283f596be41b631`.
Each contains three requests per length. All prompt hashes match the
corresponding baseline. All requests complete without errors or prompt reuse.
The logical context is 524,288. KV remains INT8 and GPU-only. Layer split
is 16. The selected prefill increase exceeds 20% at each measured length.

## Baselines

- [Archived PR 47 2K, 8K, and 32K](../2026-10-08-v100-v0141/selected.json)
- [Supplementary merged-PR-47 4K and 16K](baseline-4k-16k.json)

These baseline files use different engine binaries. See the report for
source provenance. No baseline was run again for the final confirmation.

## Ablations

- [WMMA without KV growth, 2K/8K/32K](wmma-no-grow-2k-8k-32k.json)
- [WMMA without KV growth, 4K/16K](wmma-no-grow-4k-16k.json)
- [Initial GPU-growth run, 2K/8K/32K](gpu-grow-initial-2k-8k-32k.json)
- [Initial GPU-growth run, 4K/16K](gpu-grow-initial-4k-16k.json)
- [Rejected ring 128](ring128.json)
- [Rejected eleven-worker setting](workers11.json)
- [Observed initial two-GPU KV growth](gpu-growth-evidence.json)
- [Fresh-allocation diagnostic requests](gpu-grow-transition-smoke.json)
- [Fresh-allocation execution excerpts](gpu-grow-fresh-evidence.json)
- [Cache-donation transition requests](gpu-grow-cache-transition-smoke.json)
- [Cache-donation growth, shrink, and refill excerpts](gpu-grow-cache-transition-evidence.json)

The transition diagnostics use one repeat and one output token per request.
Their prompt pool differs from the selected matrix. Do not use them for
baseline performance comparisons.

The source base is fork main `4400773dba46a4a149292c9aa301d8029e3a5b32`.
The engines were built from uncommitted source changes. Each record contains
its binary hash and source-state description. The harness checkout fields
do not identify the complete built engine source.

Home paths and private model-root paths are replaced with `$REPO` and
`$MODEL_ROOT`. No private API key or private configuration is included.
Prompt token hashes and measured row data are unchanged.
