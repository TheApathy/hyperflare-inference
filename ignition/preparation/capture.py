# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile prepared model forwards and capture CUDA graphs for their shapes."""

from contextlib import nullcontext
from time import monotonic

import torch

from ignition.progress import graph_catalog, phase

from .aot import (
    BindingMismatch, BoundForward, ForwardCache, _completed, _guard_code_cache,
    allocation_order, aot_forward,
)
from .collectives import warmup_wait
from .compilation import _defer_full_collection
from .runtime_core import capture_graph, instantiate_graph, record_export_graph

_REAL_FORWARD_CACHE = ForwardCache


_warmup_streams = {}
_warmup_pools = {}


def _warmup_stream(device):
    """Return the per-device stream that eager warmups run on.

    PyTorch caches one cuBLAS/cuBLASLt workspace per (handle, stream).
    Warmups allocate inside transient storage, so their workspaces must
    stay off the capture stream: the first capture then allocates the
    stream's workspace inside the recording pool, and the cache retains
    it for every graph the stream captures afterward.
    """
    # The stream must be created outside PyTorch's round-robin pool: a
    # caller's later torch.cuda.Stream can alias a pool stream, and
    # workspaces are cached per CUDA stream, so an aliased capture stream
    # would reuse a warmup workspace's transient address. The stream and
    # its CUDA handle live for the life of the process.
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        stream = _warmup_streams.get(index)
        if stream is None:
            from cuda.bindings import driver as cuda

            status, handle = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cannot create warmup stream: {status}")
            stream = _warmup_streams[index] = torch.cuda.ExternalStream(int(handle), device=device)
    return stream


def _seed_warmup_workspaces(stream, device):
    """Seed the calling thread's warmup-stream workspace in a dedicated pool.

    Workspaces are cached per thread's cuBLAS handle, so seeding runs on
    each capture call under the ambient BLAS backend, the backend the
    warmups use. A workspace created inside transient storage keeps its
    address in the cache after the transient backing is released, and a
    later call could reuse it; a dedicated pool keeps the seed's backing
    mapped for the life of the cache entry, outside transient storage,
    any caller-owned pool, and the exported graph pool.
    """
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        pool = _warmup_pools.get(index)
        if pool is None:
            pool = _warmup_pools[index] = torch.cuda.MemPool()
    with torch.cuda.device(index), torch.cuda.stream(stream), torch.cuda.use_mem_pool(pool):
        operand = torch.empty((512, 512), dtype=torch.bfloat16, device=device)
        torch.mm(operand, operand, out=torch.empty_like(operand))
    stream.synchronize()


def _is_integrated_device(device, warmup_storage=None):
    """Report whether the target CUDA device shares unified host/device memory."""
    integrated = getattr(warmup_storage, "integrated", None)
    if integrated is not None:
        return bool(integrated)
    target = torch.device(device)
    if target.type != "cuda":
        return False
    try:
        return bool(torch.cuda.get_device_properties(target).is_integrated)
    except Exception:
        return False


def _capture_order(names, compiled, coordinator):
    sizes = {name: getattr(forward, "capture_allocation_sizes", ()) for name, forward in compiled.items()}
    return allocation_order(names, sizes, coordinator)


def _capture_one(name, function, *, is_compiled, device, stream, warm, pool, session,
                 collective, coordinator, retain_graphs, warmup_storage, before_run,
                 retain_forwards, progress, stage):
    if coordinator is not None:
        coordinator.barrier(f"capture {name}", coordinator.ranks)
    started = monotonic()
    stream.wait_stream(torch.cuda.current_stream(device))
    exporting = retain_graphs and warmup_storage is not None
    graph = torch.cuda.CUDAGraph(keep_graph=True) if exporting else capture_graph(retain_graphs=retain_graphs)
    communication = nullcontext() if collective is None else collective.capture(stream=stream)

    def memory_detail():
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        return (f"{free / (1 << 20):.0f} MiB free, "
                f"{allocated / (1 << 20):.0f} MiB allocated, "
                f"{reserved / (1 << 20):.0f} MiB reserved, "
                f"{(total - free - reserved) / (1 << 20):.0f} MiB outside PyTorch")

    try:
        if progress is not None:
            progress(f"capturing {name}: {memory_detail()}")
        if before_run is not None:
            warm.wait_stream(stream)
            with torch.cuda.stream(warm):
                before_run(name)
            stream.wait_stream(warm)
        with communication, session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
            function()
        if exporting:
            graph = record_export_graph(graph, warmup_storage, owners=(function,))
        elif retain_graphs:
            if progress is not None:
                progress(f"instantiating {name}: {memory_detail()}")
            instantiate_graph(graph)
    except Exception as error:
        error.add_note(f"Graph {name!r} on {device}: {memory_detail()}")
        raise
    if is_compiled and not retain_forwards:
        retire = getattr(function, "retire_after_capture", None)
        if retire is not None:
            retire()
    stage.update(advance=1)
    if progress is not None:
        cached = " (compiled cache hit)" if is_compiled and function.cache_hit else ""
        progress(f"captured {name} in {monotonic() - started:.2f}s{cached}")
    return graph


def _stream_compiled_capture(cache, functions, forwards, signatures, names, *, device, stream, warm,
                             pool, session, collective, materialize, progress, label, retain_graphs,
                             warmup_storage, warmup, before_run):
    """Compile the catalog, then load, warm, capture, and retire one compiled forward at a time.

    Integrated UMA hosts share physical memory between Python guard trees, transient warmup pools,
    and the retained CUDA graph pool. Streaming keeps at most one compiled forward's guards and
    Python wrapper live during capture while preserving allocation-descending capture order,
    warmup/capture stream separation, and canonical Inductor kernel selection sharing.
    """
    from .compile_pool import CompilePool
    from .host_memory import release_unused
    from .inductor_tuning import InductorSelections

    graphs, compiled = {}, {}
    cache_hits, paths, pending = {}, {}, []
    with cache._publication_lock(), torch._inductor.config.patch(compile_threads=1):
        with phase(progress, "Compile model forwards", total=len(forwards)) as stage, CompilePool(
            cache.device, headroom_bytes=cache.headroom_bytes, options=cache.options,
            compile_workers=cache.compile_workers,
        ) as compiler:
            def finish():
                name, bound, path = _completed(pending)
                try:
                    with _guard_code_cache(cache.root / "guards"):
                        if bound.pending is not None:
                            bound.pending.result()
                            cache_hits[name] = False
                        else:
                            cache_hits[name] = bool(bound.cache_hit)
                        paths[name] = path
                except Exception as error:
                    raise RuntimeError(f"failed to prepare compiled forward {name}") from error
                stage.update(advance=1)

            for name, forward in forwards.items():
                warm.wait_stream(torch.cuda.current_stream(device))
                warm.wait_stream(stream)
                with _guard_code_cache(cache.root / "guards"), torch.cuda.stream(warm):
                    bound, path = cache._prepare(
                        forward, name=name, signature=signatures[name], compiler=compiler,
                        deferred=materialize is not None,
                        before_run=None if before_run is None else lambda name=name: before_run(name),
                    )
                if bound.pending is None:
                    # Drop the restored wrapper immediately; streaming reloads one forward at a time.
                    bound = BoundForward(None, forward, bound.cache_hit)
                    release_unused(generation=0)
                pending.append((name, bound, path))
                if len(pending) >= compiler.workers:
                    finish()
            while pending:
                finish()
        release_unused()
        if materialize is not None:
            materialize()
        sizes = {name: cache._allocation_sizes(paths[name]) for name in forwards}
        ordered_names = allocation_order(names, sizes)
        uncompiled_names = tuple(name for name in names if name not in forwards)
        if uncompiled_names:
            with phase(progress, "Warm serving graphs", total=len(uncompiled_names)) as stage:
                for name in uncompiled_names:
                    warm.wait_stream(torch.cuda.current_stream(device))
                    warm.wait_stream(stream)
                    with warmup(), torch.cuda.stream(warm):
                        if before_run is not None:
                            before_run(name)
                        functions[name]()
                    warm.synchronize()
                    stage.update(advance=1)
        selections = InductorSelections(cache.device, lock_gpu_clocks=cache.lock_gpu_clocks,
                                        source_identity=cache.root.name)
        with phase(progress, label, total=len(names)) as stage:
            for name in ordered_names:
                if name in forwards:
                    forward = forwards[name]
                    run_before = None if before_run is None else lambda name=name: before_run(name)
                    try:
                        with _guard_code_cache(cache.root / "guards"):
                            warm.wait_stream(torch.cuda.current_stream(device))
                            warm.wait_stream(stream)
                            with torch.cuda.stream(warm):
                                try:
                                    bound = cache._load(forward, paths[name], required=True,
                                                        before_run=run_before)
                                except BindingMismatch:
                                    if materialize is None:
                                        raise
                                    bound = BoundForward(aot_forward(forward, options=cache.options), forward, False)
                                else:
                                    bound.cache_hit = cache_hits[name]
                            with warmup_wait(collective, None), torch.cuda.stream(warm):
                                bound = cache._warm(bound, paths[name], name=name, progress=None,
                                                    selections=selections, warmup=warmup,
                                                    before_run=run_before)
                            warm.synchronize()
                            bound.prepare_capture()
                            selections.release(bound)
                    except Exception as error:
                        if materialize is None:
                            raise RuntimeError(f"failed to prepare compiled forward {name}") from error
                        raise
                    stream.wait_stream(warm)
                    graphs[name] = _capture_one(
                        name, bound, is_compiled=True, device=device, stream=stream, warm=warm,
                        pool=pool, session=session, collective=collective, coordinator=None,
                        retain_graphs=retain_graphs, warmup_storage=warmup_storage,
                        before_run=before_run, retain_forwards=False, progress=progress, stage=stage,
                    )
                    compiled[name] = bound
                else:
                    stream.wait_stream(warm)
                    graphs[name] = _capture_one(
                        name, functions[name], is_compiled=False, device=device, stream=stream, warm=warm,
                        pool=pool, session=session, collective=collective, coordinator=None,
                        retain_graphs=retain_graphs, warmup_storage=warmup_storage,
                        before_run=before_run, retain_forwards=False, progress=progress, stage=stage,
                    )
    torch.cuda.synchronize(device)
    release_unused()
    return graphs, compiled


@_defer_full_collection()
def capture_forwards(functions, signatures, *, session, device, stream, pool,
                     collective=None, coordinator=None, compile_decoder=True, materialize=None,
                     progress=None, label="Prepare serving graphs", cache_root=None, retain_graphs=False,
                     warmup_storage=None, capture_names=None, before_run=None,
                     report_catalog=True, retain_forwards=False, compile_options=None,
                     compile_kinds=("prefill", "terminal_prefill", "scored_prefill", "decode", "resume", "speculative", "mixed", "vision")):
    """Retain compiled resources and capture with an optional b12x IPC runtime.

    Persistent state and graph outputs must live in caller-owned storage.
    Only graph-local intermediates may use the shared graph memory pool;
    serving replays variants serially and may choose them in any order.
    On discrete GPUs, all eager warmups precede capture so their transient
    workspaces do not coexist with retained graph intermediates. On single-rank
    integrated UMA GPUs when ``retain_forwards`` is false, compiled forwards
    stream one at a time (load, warm inside transient storage, capture, and
    retire guards) so peak host/UMA memory does not retain every forward's
    guard tree simultaneously.
    Graphs captured on one stream share that stream's cached workspace;
    resetting any of them frees the workspace the others still reference.
    Callers must reset all of a capture stream's graphs together or keep
    separately reset groups on separate streams.
    ``retain_graphs`` keeps templates and executables for export when
    ``warmup_storage`` supplies external CUDA pool owners.
    ``before_run`` restores caller-owned trial inputs outside compiled warmup
    and graph capture; it is unused for the ordinary serving catalog.
    ``retain_forwards`` preserves guarded Python execution for eager diagnostics.
    ``compile_options`` overrides Inductor configuration passed to ``ForwardCache``
    without mutating the module defaults.
    """
    if (report_catalog and progress is not None and not getattr(session, "_graph_catalog_reported", False)
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
        graph_catalog(progress, (signatures[name] for name in functions))
        session._graph_catalog_reported = True
    names = tuple(functions) if capture_names is None else tuple(capture_names)
    if len(set(names)) != len(names) or not set(names) <= functions.keys():
        raise ValueError("capture selection must name distinct prepared forwards")
    graphs, compiled = {}, {}
    warmup = nullcontext if warmup_storage is None else warmup_storage.transient
    warm = _warmup_stream(device)
    _seed_warmup_workspaces(warm, device)
    if compile_decoder:
        cache = ForwardCache(namespace=session.namespace.to_dict(), device=device, root=cache_root,
                             headroom_bytes=getattr(session, "headroom_bytes", 0),
                             options=compile_options,
                             compile_workers=getattr(session, "requested_compile_workers", None),
                             lock_gpu_clocks=getattr(session, "lock_gpu_clocks", False) and session.autotune)
        forwards = {name: function for name, function in functions.items()
                    if name in names and signatures[name].kind in compile_kinds}
        if (forwards and coordinator is None and not retain_forwards
                and isinstance(cache, _REAL_FORWARD_CACHE)
                and _is_integrated_device(device, warmup_storage)):
            return _stream_compiled_capture(
                cache, functions, forwards, signatures, names, device=device, stream=stream,
                warm=warm, pool=pool, session=session, collective=collective,
                materialize=materialize, progress=progress, label=label,
                retain_graphs=retain_graphs, warmup_storage=warmup_storage,
                warmup=warmup, before_run=before_run,
            )
        warm.wait_stream(torch.cuda.current_stream(device))
        warm.wait_stream(stream)
        retention = nullcontext() if warmup_storage is None else warmup_storage.retain_transient_backing()
        with retention, warmup_wait(collective, coordinator):
            with torch.cuda.stream(warm):
                compiled = cache.prepare_many(forwards, signatures, progress=progress,
                                              coordinator=coordinator, materialize=materialize, warmup=warmup,
                                              before_run=before_run)
            warm.synchronize()
        with phase(progress, "Release serialized compiler state", total=len(compiled)) as stage:
            released, counted = 0, set()
            for forward in compiled.values():
                artifact = id(getattr(forward, "artifact", forward))
                size = forward.prepare_capture()
                if artifact not in counted:
                    released += size
                    counted.add(artifact)
                stage.update(advance=1)
            stage.update(f"{released >> 20} MiB; guards and kernel owners retained")
        from .host_memory import release_unused

        release_unused()
    elif materialize is not None:
        materialize()
    with phase(progress, "Warm serving graphs", total=len(names)) as stage:
        for name in names:
            function = functions[name]
            if coordinator is not None:
                coordinator.barrier(f"capture warmup {name}", coordinator.ranks)
            warm.wait_stream(torch.cuda.current_stream(device))
            warm.wait_stream(stream)
            with warmup(), torch.cuda.stream(warm):
                if before_run is not None:
                    before_run(name)
                if name not in compiled:
                    function()
            warm.synchronize()
            stage.update(advance=1)
    stream.wait_stream(warm)
    ordered_names = _capture_order(names, compiled, coordinator)
    with phase(progress, label, total=len(names)) as stage:
        for name in ordered_names:
            function = compiled.get(name, functions[name])
            graphs[name] = _capture_one(
                name, function, is_compiled=name in compiled, device=device, stream=stream, warm=warm,
                pool=pool, session=session, collective=collective, coordinator=coordinator,
                retain_graphs=retain_graphs, warmup_storage=warmup_storage,
                before_run=before_run, retain_forwards=retain_forwards, progress=progress, stage=stage,
            )
    torch.cuda.synchronize(device)
    return graphs, compiled
