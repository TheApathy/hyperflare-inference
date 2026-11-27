# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming compiled-forward capture on integrated UMA GPUs."""

from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import weakref

import pytest

from ignition.preparation import aot, capture, compile_pool, inductor_tuning


def _setup_streaming_environment(monkeypatch, tmp_path, *, integrated, sizes, cache_hits):
    """Install CPU mocks for ForwardCache, CompilePool, InductorSelections, and CUDA streams."""
    events = []
    live_loaded = set()
    peak_live = [0]
    entered_streams = []
    captured_streams = []
    captured_graphs = []
    selection_instances = []

    stream = SimpleNamespace(
        name="capture",
        wait_stream=lambda other: None,
        synchronize=lambda: None,
    )
    warm = SimpleNamespace(
        name="warm",
        wait_stream=lambda other: None,
        synchronize=lambda: events.append("warm_sync"),
    )

    class FakePool:
        workers = 2

        def __init__(self, *args, **kwargs):
            events.append("compile_pool_open")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("compile_pool_close")

        def submit(self, forward, path):
            events.append(("compile_submit", forward.name))
            future = Future()
            future.set_result(None)
            return future

    class FakeSelections:
        def __init__(self, *args, **kwargs):
            self.winners = {}
            self.released = []
            selection_instances.append(self)

        def release(self, bound):
            self.released.append(bound.owner.name)
            events.append(("selections_release", bound.owner.name))

    class FakeBound:
        def __init__(self, owner, *, cache_hit, pending=None):
            self.owner = owner
            self.cache_hit = cache_hit
            self.pending = pending
            self.capture_allocation_sizes = sizes[owner.name]
            self.prepared = False
            self.retired = False
            if pending is None:
                token = (owner.name, id(self))
                live_loaded.add(token)
                peak_live[0] = max(peak_live[0], len(live_loaded))
                self._live_token = token
                weakref.finalize(self, live_loaded.discard, token)
            else:
                self._live_token = None

        def __call__(self):
            assert not self.retired
            events.append(("run", self.owner.name))
            return self.owner()

        def prepare_capture(self):
            self.prepared = True
            events.append(("prepare_capture", self.owner.name))
            return 0

        def retire_after_capture(self):
            assert not self.retired
            self.retired = True
            if self._live_token is not None:
                live_loaded.discard(self._live_token)
            events.append(("retire", self.owner.name))

    class FakeGraph:
        def __init__(self, keep_graph=False):
            self.keep_graph = keep_graph
            self.captured_name = None

    @contextmanager
    def record_stream(value):
        entered_streams.append(value)
        yield

    @contextmanager
    def record_graph(graph, *, pool, stream):
        captured_streams.append(stream)
        events.append(("graph_begin", pool))
        yield
        events.append("graph_end")
        captured_graphs.append(graph)

    @contextmanager
    def transient_scope():
        events.append("transient_enter")
        yield
        events.append("transient_exit")

    @contextmanager
    def retention_scope():
        events.append("retention_enter")
        yield
        events.append("retention_exit")

    def fake_cache_init(self, *, namespace, device, root=None, headroom_bytes=0, options=None,
                        compile_workers=None, lock_gpu_clocks=False):
        self.device = device
        self.headroom_bytes = headroom_bytes
        self.compile_workers = compile_workers
        self.lock_gpu_clocks = lock_gpu_clocks
        self.options = MappingProxyType({})
        self.root = tmp_path / "cache"
        self.root.mkdir(parents=True, exist_ok=True)

    def fake_prepare(self, forward, *, name, signature, compiler=None, progress=None,
                     deferred=False, before_run=None):
        path = self.root / f"{name}.bin"
        if cache_hits.get(name, False):
            if before_run is not None:
                before_run()
            return FakeBound(forward, cache_hit=True), path
        return FakeBound(forward, cache_hit=False, pending=compiler.submit(forward, path)), path

    def fake_load(self, forward, path, *, required=False, before_run=None):
        events.append(("load", forward.name))
        if before_run is not None:
            before_run()
        return FakeBound(forward, cache_hit=True)

    def fake_warm(self, bound, path, *, name, progress=None, selections=None, warmup=nullcontext,
                  before_run=None):
        events.append(("warm", name))
        with warmup():
            if before_run is not None:
                before_run()
            bound()
        if selections is not None:
            selections.winners[name] = f"winner:{name}"
        return bound

    monkeypatch.setattr(compile_pool, "CompilePool", FakePool)
    monkeypatch.setattr(inductor_tuning, "InductorSelections", FakeSelections)
    monkeypatch.setattr(capture, "_guard_code_cache", lambda path: nullcontext())
    monkeypatch.setattr(aot, "_guard_code_cache", lambda path: nullcontext())
    monkeypatch.setattr(capture, "_warmup_stream", lambda device: warm)
    monkeypatch.setattr(capture, "_seed_warmup_workspaces", lambda stream, device: None)
    monkeypatch.setattr(capture.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(capture.torch.cuda, "stream", record_stream)
    monkeypatch.setattr(capture.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(capture.torch.cuda, "graph", record_graph)
    monkeypatch.setattr(capture.torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(capture.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(aot.ForwardCache, "__init__", fake_cache_init)
    monkeypatch.setattr(aot.ForwardCache, "_publication_lock", lambda self, coordinator=None: nullcontext())
    monkeypatch.setattr(aot.ForwardCache, "_prepare", fake_prepare)
    monkeypatch.setattr(aot.ForwardCache, "_allocation_sizes", lambda self, path: sizes[Path(path).stem])
    monkeypatch.setattr(aot.ForwardCache, "_load", fake_load)
    monkeypatch.setattr(aot.ForwardCache, "_warm", fake_warm)
    monkeypatch.setattr(capture, "record_export_graph",
                        lambda graph, storage, *, owners: SimpleNamespace(graph=graph, owners=owners))

    warmup_storage = SimpleNamespace(
        integrated=integrated,
        transient=transient_scope,
        retain_transient_backing=retention_scope,
    )
    session = SimpleNamespace(
        namespace=SimpleNamespace(to_dict=lambda: {}),
        capture=nullcontext,
        autotune=False,
    )
    return SimpleNamespace(
        events=events,
        peak_live=peak_live,
        live_loaded=live_loaded,
        entered_streams=entered_streams,
        captured_streams=captured_streams,
        selection_instances=selection_instances,
        stream=stream,
        warm=warm,
        warmup_storage=warmup_storage,
        session=session,
    )


def _make_forward(name, outputs, events=None):
    def forward():
        outputs.append(name)
        if events is not None:
            events.append(("call_forward", name))
        return name

    forward.name = name
    return forward


def test_integrated_uma_streams_load_warm_capture_and_retire_one_forward_at_a_time(tmp_path, monkeypatch):
    sizes = {"small": (16,), "large": (128,), "medium": (64,)}
    cache_hits = {"small": True, "large": False, "medium": True}
    env = _setup_streaming_environment(monkeypatch, tmp_path, integrated=True, sizes=sizes, cache_hits=cache_hits)
    outputs = []
    functions = {
        "small": _make_forward("small", outputs, env.events),
        "large": _make_forward("large", outputs, env.events),
        "medium": _make_forward("medium", outputs, env.events),
        "init": _make_forward("init", outputs, env.events),
    }
    signatures = {
        "small": SimpleNamespace(kind="decode"),
        "large": SimpleNamespace(kind="prefill"),
        "medium": SimpleNamespace(kind="speculative"),
        "init": SimpleNamespace(kind="init"),
    }
    materialized = []

    graphs, compiled = capture.capture_forwards(
        functions,
        signatures,
        session=env.session,
        device="cuda:0",
        stream=env.stream,
        pool="graph_pool",
        materialize=lambda: (materialized.append(True), env.events.append("materialize")),
        warmup_storage=env.warmup_storage,
        retain_graphs=True,
        before_run=lambda name: env.events.append(("before_run", name)),
    )

    assert materialized == [True]
    # Peak concurrently loaded compiled forwards stays bounded to 1 on integrated UMA.
    assert env.peak_live[0] == 1
    assert not env.live_loaded
    assert "retention_enter" not in env.events

    # Capture order matches allocation_order (large -> medium -> small -> uncompiled init).
    assert tuple(graphs) == ("large", "medium", "small", "init")
    assert tuple(compiled) == ("large", "medium", "small")
    assert {name: value.cache_hit for name, value in compiled.items()} == {
        "large": False,
        "medium": True,
        "small": True,
    }
    assert all(value.retired for value in compiled.values())
    assert all(graphs[name].owners == (compiled[name],) for name in compiled)

    # Every compiled forward completes load -> warm -> transient exit -> prepare_capture -> graph capture -> retire
    # before the next compiled forward loads.
    for name in ("large", "medium", "small"):
        idx_load = env.events.index(("load", name))
        idx_warm = env.events.index(("warm", name))
        idx_prepare = env.events.index(("prepare_capture", name))
        idx_retire = env.events.index(("retire", name))
        assert idx_load < idx_warm < idx_prepare < idx_retire
    assert env.events.index(("retire", "large")) < env.events.index(("load", "medium"))
    assert env.events.index(("retire", "medium")) < env.events.index(("load", "small"))

    # Uncompiled forward warms before any graph capture starts.
    first_graph = env.events.index(("graph_begin", "graph_pool"))
    assert env.events.index(("call_forward", "init")) < first_graph

    # Stream isolation and shared InductorSelections across streamed forwards.
    assert env.entered_streams and all(value is env.warm for value in env.entered_streams)
    assert env.captured_streams and all(value is env.stream for value in env.captured_streams)
    assert len(env.selection_instances) == 1
    assert env.selection_instances[0].released == ["large", "medium", "small"]
    assert set(env.selection_instances[0].winners) == {"large", "medium", "small"}


def test_streaming_and_discrete_capture_produce_identical_order_and_results(tmp_path, monkeypatch):
    sizes = {"a": (32,), "b": (128,), "c": (64,)}
    cache_hits = {"a": True, "b": False, "c": True}
    results = {}

    for mode, integrated in (("discrete", False), ("integrated", True)):
        env = _setup_streaming_environment(
            monkeypatch, tmp_path / mode, integrated=integrated, sizes=sizes, cache_hits=cache_hits,
        )
        outputs = []
        functions = {name: _make_forward(name, outputs) for name in ("a", "b", "c", "snapshot")}
        signatures = {
            "a": SimpleNamespace(kind="decode"),
            "b": SimpleNamespace(kind="prefill"),
            "c": SimpleNamespace(kind="speculative"),
            "snapshot": SimpleNamespace(kind="snapshot"),
        }
        graphs, compiled = capture.capture_forwards(
            functions,
            signatures,
            session=env.session,
            device="cuda:0",
            stream=env.stream,
            pool="graph_pool",
            materialize=None,
            warmup_storage=env.warmup_storage,
        )
        results[mode] = (
            tuple(graphs),
            tuple(compiled),
            {name: value.cache_hit for name, value in compiled.items()},
            {name: value.retired for name, value in compiled.items()},
            outputs,
            env.peak_live[0],
            "retention_enter" in env.events,
        )

    assert results["integrated"][:4] == results["discrete"][:4]
    assert results["integrated"][0] == ("b", "c", "a", "snapshot")
    assert results["integrated"][2] == {"b": False, "c": True, "a": True}
    assert sorted(results["integrated"][4]) == sorted(results["discrete"][4])
    assert results["integrated"][5] == 1 and results["integrated"][6] is False
    assert results["discrete"][5] == 3 and results["discrete"][6] is True


def test_integrated_uma_retain_forwards_keeps_guarded_callables_unretired(tmp_path, monkeypatch):
    sizes = {"first": (64,), "second": (32,)}
    env = _setup_streaming_environment(
        monkeypatch, tmp_path, integrated=True, sizes=sizes, cache_hits={"first": True, "second": True},
    )
    outputs = []
    functions = {name: _make_forward(name, outputs) for name in ("first", "second")}
    signatures = {name: SimpleNamespace(kind="decode") for name in functions}

    graphs, compiled = capture.capture_forwards(
        functions,
        signatures,
        session=env.session,
        device="cuda:0",
        stream=env.stream,
        pool="graph_pool",
        materialize=lambda: None,
        warmup_storage=env.warmup_storage,
        retain_forwards=True,
    )

    assert tuple(graphs) == ("first", "second")
    assert all(not value.retired for value in compiled.values())
    assert "retention_enter" in env.events
