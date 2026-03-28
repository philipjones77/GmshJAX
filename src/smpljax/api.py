from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .body_models import SMPLJAXModel
    from .optimized import CachePolicy, OptimizedSMPLJAX


RuntimeMode = Literal["plain", "uncached", "optimized"]


def create(model_path: str | Path, use_cache: bool = True, max_entries: int = 2) -> SMPLJAXModel:
    """Create a baseline model from a model package (.npz or .pkl)."""
    from .body_models import SMPLJAXModel
    from .io import load_model

    return SMPLJAXModel(data=load_model(model_path, use_cache=use_cache, max_entries=max_entries))


def create_uncached(model_path: str | Path) -> SMPLJAXModel:
    """Create a baseline model while bypassing the IO cache."""
    return create(model_path, use_cache=False)


def create_optimized(
    model_path: str | Path,
    *,
    use_io_cache: bool = True,
    io_cache_entries: int = 2,
    dtype: str | None = None,
    cache_policy: CachePolicy | None = None,
    backend: str | None = None,
    batch_size_hint: int | None = None,
    prefer_fixed_padding: bool = False,
    fixed_padded_batch_size: int | None = None,
    forbid_new_compiles: bool | None = None,
    max_compiled: int | None = None,
) -> OptimizedSMPLJAX:
    """Create an optimized runtime with compile caching left enabled."""
    from .io import load_model
    from .optimized import CachePolicy, OptimizedSMPLJAX

    resolved_policy = cache_policy
    if resolved_policy is None:
        resolved_policy = CachePolicy.recommended(
            dtype="float32" if dtype is None else dtype,
            backend=backend,
            batch_size_hint=batch_size_hint,
            prefer_fixed_padding=prefer_fixed_padding,
            fixed_padded_batch_size=fixed_padded_batch_size,
            forbid_new_compiles=forbid_new_compiles,
            max_compiled=max_compiled,
        )
    return OptimizedSMPLJAX(
        data=load_model(model_path, use_cache=use_io_cache, max_entries=io_cache_entries),
        dtype=dtype,
        cache_policy=resolved_policy,
    )


def create_runtime(
    model_path: str | Path,
    *,
    mode: RuntimeMode = "plain",
    io_cache_entries: int = 2,
    dtype: str | None = None,
    cache_policy: CachePolicy | None = None,
    backend: str | None = None,
    batch_size_hint: int | None = None,
    prefer_fixed_padding: bool = False,
    fixed_padded_batch_size: int | None = None,
    forbid_new_compiles: bool | None = None,
    max_compiled: int | None = None,
) -> SMPLJAXModel | OptimizedSMPLJAX:
    """Create a runtime by mode while preserving explicit cached and non-cached paths."""
    if mode == "plain":
        return create(model_path, use_cache=True, max_entries=io_cache_entries)
    if mode == "uncached":
        return create(model_path, use_cache=False)
    if mode == "optimized":
        return create_optimized(
            model_path,
            use_io_cache=True,
            io_cache_entries=io_cache_entries,
            dtype=dtype,
            cache_policy=cache_policy,
            backend=backend,
            batch_size_hint=batch_size_hint,
            prefer_fixed_padding=prefer_fixed_padding,
            fixed_padded_batch_size=fixed_padded_batch_size,
            forbid_new_compiles=forbid_new_compiles,
            max_compiled=max_compiled,
        )
    raise ValueError(f"Unsupported runtime mode: {mode}. Expected 'plain', 'uncached', or 'optimized'.")
