# SMPL Cache And Padding Policy Implementation

The optimized SMPL runtime uses explicit cache and padding policy controls through `CachePolicy`.
For practical callers, `smpljax.create_optimized(...)` and `smpljax.create_runtime(..., mode="optimized")`
can now synthesize a policy through `CachePolicy.recommended(...)` instead of forcing every caller
to build the dataclass manually.

Key controls:

- `max_compiled`: bound the number of retained compiled variants
- `batch_buckets`: bucket repeated calls onto a small set of padded batch sizes
- `enable_batch_bucketing`: allow or disable bucketed padding behavior
- `fixed_padded_batch_size`: force a single padded compile shape for repeated smaller requests
- `forbid_new_compiles`: raise on unexpected new compile keys once the intended compile surface is established
- `dtype`: fix runtime precision policy

Recommended policy behavior:

- CPU defaults to a tighter bucket set oriented around smaller practical workloads.
- GPU and TPU defaults keep a wider bucket ladder and a larger default compile-cache budget.
- `batch_size_hint` trims or extends the bucket ladder so the runtime does not over-compile for obviously smaller jobs.
- `prefer_fixed_padding=True` converts the hint into a single padded capacity and, by default, enables strict no-new-compile mode.
- `fixed_padded_batch_size` and `max_compiled` can still be forced explicitly through the API surface when a deployment envelope is already known.

Robustness rules:

- If a requested batch is larger than the largest configured bucket, the runtime now uses the exact request size instead of truncating to the largest bucket.
- Fixed padded capacity is still the only mode that rejects larger requests outright, because that is an explicit compile-envelope contract.

Related IO policy:

- model IO is separately cached through the bounded IO cache exposed by `smpljax.io` and `common.smpl`

Diagnostics policy:

- runtime diagnostics expose compile counts, cache hits and misses, compiled key history, warmup coverage, backend metadata, and padding-related state
