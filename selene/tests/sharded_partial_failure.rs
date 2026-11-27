/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Multi-process fault-injection test for sharded-store partial store failures.
//!
//! Spawns separate peer rank OS processes communicating with rank zero over loopback TCP.
//! When one rank fails a checkpoint store asynchronously while the remaining ranks write
//! their record files to disk, rank zero unlinks the surviving ranks' files on disk before
//! reporting the store failure.

use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use blake2::{Blake2b, Digest, digest::consts::U16};
use flare_selene::{
    CacheGeometry, CaptureRole, CheckpointId, Direction, Error, HASH_BLOCK_TOKENS, History,
    HostRegistrar, HostStore, InboundTicket, Job, Layout, PageCopier, PageId, Plane, PrefixKey,
    PublishedCheckpoint, Registration, Reply, Result, SharedCopier, SplitMapping, StateSlotId,
    StoreEvent, StoreOutcome,
    offload::{
        self, CopierFactory, PrefixCacheOffload, Preparation,
        sharded::{join_peer, join_rank_zero},
    },
    record, slab,
};

const MODEL_IDENTITY: [u8; 32] = [0x33; 32];
const WORLD: usize = 3;
const PAGE_TOKENS: u64 = 64;
const PAGE_BYTES: u64 = 4096;
const ROW_BYTES: u64 = 3 * 4096;

fn local_geometry() -> CacheGeometry {
    CacheGeometry {
        placement: Default::default(),
        codec_fields: Vec::new(),
        snapshot_pages: Vec::new(),
        page_tokens: PAGE_TOKENS,
        page_capacity: 4096,
        row_capacity: 64,
        row: Plane {
            name: "state".into(),
            address: 0x7000_0100_0000,
            unit_bytes: ROW_BYTES,
        },
        planes: vec![Plane {
            name: "kv".into(),
            address: 0x7000_0000_0000,
            unit_bytes: PAGE_BYTES,
        }],
    }
}

struct IdentityRegistrar;
struct IdentityRegistration(usize);

impl Registration for IdentityRegistration {
    fn device_address(&self) -> u64 {
        self.0 as u64
    }
}

impl HostRegistrar for IdentityRegistrar {
    fn register(&self, _: usize, address: usize, _: usize) -> Result<Box<dyn Registration>> {
        Ok(Box::new(IdentityRegistration(address)))
    }

    fn map_split(&self, _: &[(usize, i32, usize)]) -> Result<Box<dyn SplitMapping>> {
        Err(Error::new("split mapping is not provided by this test"))
    }
}

#[derive(Clone, Default)]
struct SimulatedMemory {
    pages: Arc<Mutex<HashMap<u32, Vec<u8>>>>,
    rows: Arc<Mutex<HashMap<u32, Vec<u8>>>>,
}

struct FaultInjectingCopier {
    layout: Layout,
    rank: usize,
    memory: SimulatedMemory,
    fail_on_out_store: Option<usize>,
    out_store_count: usize,
    replies: Vec<Reply>,
}

impl PageCopier for FaultInjectingCopier {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn submit(&mut self, job: Job) -> Result<()> {
        if job.direction == Direction::Out {
            self.out_store_count += 1;
            if self.fail_on_out_store == Some(self.out_store_count) {
                self.replies.push(Reply {
                    id: job.id,
                    window: job.window,
                    direction: job.direction,
                    result: Err(Error::new(format!(
                        "injected store failure on rank {}",
                        self.rank
                    ))),
                    bytes: 0,
                    elapsed: Duration::from_micros(1),
                });
                return Ok(());
            }
        }
        let page = self.layout.page_record_bytes as usize;
        let row = self.layout.row_record_bytes as usize;
        let (mut pages, mut rows) = (
            self.memory.pages.lock().expect("pages lock"),
            self.memory.rows.lock().expect("rows lock"),
        );
        for (id, address) in &job.pages {
            // SAFETY: The host address names a mapped staging slot retained for the job.
            let host = unsafe { std::slice::from_raw_parts_mut(*address as *mut u8, page) };
            match job.direction {
                Direction::Out => {
                    let fallback = vec![0u8; page];
                    host.copy_from_slice(pages.get(&id.0).unwrap_or(&fallback));
                }
                Direction::In => {
                    pages.insert(id.0, host.to_vec());
                }
            }
        }
        if let Some((slot, slots)) = &job.row {
            let mut data = rows.get(&slot.0).cloned().unwrap_or_else(|| vec![0u8; row]);
            for (address, offset, len) in slab::row_pieces(&self.layout, slots) {
                // SAFETY: Each row piece names a mapped staging slot retained for the job.
                let host = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, len) };
                let bytes = &mut data[offset as usize..offset as usize + len];
                match job.direction {
                    Direction::Out => host.copy_from_slice(bytes),
                    Direction::In => bytes.copy_from_slice(host),
                }
            }
            if job.direction == Direction::In {
                rows.insert(slot.0, data);
            }
        }
        self.replies.push(Reply {
            id: job.id,
            window: job.window,
            direction: job.direction,
            result: Ok(()),
            bytes: job.pages.len() as u64 * self.layout.page_record_bytes
                + u64::from(job.row.is_some()) * self.layout.row_record_bytes,
            elapsed: Duration::from_micros(1),
        });
        Ok(())
    }

    fn poll(&mut self, replies: &mut Vec<Reply>) {
        replies.append(&mut self.replies);
    }

    fn in_flight(&self) -> usize {
        self.replies.len()
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct FaultCopiers {
    fail_on_out_store: Option<usize>,
    memories: Mutex<HashMap<usize, SimulatedMemory>>,
}

impl FaultCopiers {
    fn new(fail_on_out_store: Option<usize>) -> Self {
        Self {
            fail_on_out_store,
            memories: Mutex::new(HashMap::new()),
        }
    }
}

impl CopierFactory for FaultCopiers {
    fn mapped(
        &self,
        contexts: &[(usize, i32)],
        geometry: Vec<CacheGeometry>,
        _: &slab::SlabPool,
    ) -> Result<SharedCopier> {
        let rank = contexts[0].0;
        let memory = SimulatedMemory::default();
        for page in 1..=4u32 {
            memory.pages.lock().expect("pages lock").insert(
                page,
                vec![(rank as u8).wrapping_add(page as u8); PAGE_BYTES as usize],
            );
        }
        for slot in 1..=2u32 {
            memory.rows.lock().expect("rows lock").insert(
                slot,
                vec![(rank as u8).wrapping_add(slot as u8 + 10); ROW_BYTES as usize],
            );
        }
        self.memories
            .lock()
            .expect("memories lock")
            .insert(rank, memory.clone());
        Ok(Arc::new(Mutex::new(FaultInjectingCopier {
            layout: Layout::new(&geometry)?,
            rank,
            memory,
            fail_on_out_store: self.fail_on_out_store,
            out_store_count: 0,
            replies: Vec::new(),
        })))
    }

    fn unmapped(&self, _: &[(usize, i32)], _: Vec<CacheGeometry>) -> Result<Box<dyn PageCopier>> {
        panic!("sharded disk store test uses mapped staging only")
    }
}

struct SimpleHistory {
    identity: [u8; 32],
    boundaries: Vec<PrefixKey>,
    tokens: Vec<u32>,
}

impl SimpleHistory {
    fn new(identity: [u8; 32]) -> Self {
        let mut hash = Blake2b::<U16>::new();
        hash.update(identity);
        Self {
            identity,
            boundaries: vec![PrefixKey {
                consumed_tokens: 0,
                digest: hash.finalize().into(),
            }],
            tokens: Vec::new(),
        }
    }

    fn append(&mut self, tokens: &[u32]) {
        for &token in tokens {
            self.tokens.push(token);
            if self.tokens.len().is_multiple_of(HASH_BLOCK_TOKENS) {
                let end = self.tokens.len();
                let key = self.digest(end, self.boundaries.last().expect("root").digest);
                self.boundaries.push(key);
            }
        }
    }

    fn digest(&self, end: usize, parent: [u8; 16]) -> PrefixKey {
        let start = if end.is_multiple_of(HASH_BLOCK_TOKENS) {
            end - HASH_BLOCK_TOKENS
        } else {
            end - end % HASH_BLOCK_TOKENS
        };
        let mut hash = Blake2b::<U16>::new();
        hash.update(self.identity);
        hash.update(parent);
        hash.update((end as u64).to_le_bytes());
        for token in &self.tokens[start..end] {
            hash.update(token.to_le_bytes());
        }
        PrefixKey {
            consumed_tokens: end as u64,
            digest: hash.finalize().into(),
        }
    }
}

impl History for SimpleHistory {
    fn token_count(&self) -> u64 {
        self.tokens.len() as u64
    }

    fn key_at(&self, consumed_tokens: u64) -> Option<PrefixKey> {
        let end = usize::try_from(consumed_tokens).ok()?;
        if end > self.tokens.len() {
            return None;
        }
        let boundary = self.boundaries[end / HASH_BLOCK_TOKENS];
        if end.is_multiple_of(HASH_BLOCK_TOKENS) {
            return Some(boundary);
        }
        Some(self.digest(end, boundary.digest))
    }

    fn aligned_boundaries(&self) -> &[PrefixKey] {
        &self.boundaries[1..]
    }
}

fn write_blob(stream: &mut TcpStream, bytes: &[u8]) {
    stream
        .write_all(&(bytes.len() as u32).to_le_bytes())
        .expect("write length");
    stream.write_all(bytes).expect("write payload");
    stream.flush().expect("flush payload");
}

fn read_blob(stream: &mut TcpStream) -> Vec<u8> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).expect("read length");
    let mut bytes = vec![0u8; u32::from_le_bytes(len) as usize];
    stream.read_exact(&mut bytes).expect("read payload");
    bytes
}

fn shard_record_dir(rank_root: &Path, rank: usize, world: usize, local: &CacheGeometry) -> PathBuf {
    rank_root.join(record::directory_name(
        &MODEL_IDENTITY,
        &flare_selene::geometry::shard_fingerprint(local, rank, world),
    ))
}

fn completed_event(store: &mut dyn HostStore) -> StoreEvent {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let mut events = Vec::new();
        store.poll(&mut events).expect("poll store");
        if !events.is_empty() {
            assert_eq!(events.len(), 1, "unexpected batch: {events:?}");
            return events.remove(0);
        }
        assert!(Instant::now() < deadline, "sharded store transfer stalled");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn probe_direct_io_dir(name: &str) -> Option<PathBuf> {
    let directory = std::env::temp_dir().join(format!(
        "flare-selene-partial-failure-{name}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).expect("create test dir");
    let probe = directory.join(".probe");
    match flare_eos::direct_file(&probe, true) {
        Ok(_) => {
            fs::remove_file(&probe).expect("remove probe");
            Some(directory)
        }
        Err(error) if cfg!(unix) && error.raw_os_error() == Some(libc::EINVAL) => {
            eprintln!(
                "skipping {name}: {} refuses direct I/O",
                directory.display()
            );
            let _ = fs::remove_dir_all(&directory);
            None
        }
        Err(error) => panic!("probe {}: {error}", probe.display()),
    }
}

fn run_worker(args: &[String]) {
    let mut rank: Option<usize> = None;
    let mut world: Option<usize> = None;
    let mut dir: Option<PathBuf> = None;
    let mut exchange: Option<SocketAddr> = None;
    let mut fail_on_out_store: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rank" => {
                rank = Some(args[i + 1].parse().expect("rank"));
                i += 2;
            }
            "--world" => {
                world = Some(args[i + 1].parse().expect("world"));
                i += 2;
            }
            "--dir" => {
                dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--exchange" => {
                exchange = Some(args[i + 1].parse().expect("exchange addr"));
                i += 2;
            }
            "--fail-store" => {
                fail_on_out_store = Some(args[i + 1].parse().expect("fail store index"));
                i += 2;
            }
            other => panic!("unknown worker argument: {other}"),
        }
    }

    let rank = rank.expect("--rank");
    let world = world.expect("--world");
    let dir = dir.expect("--dir");
    let exchange_addr = exchange.expect("--exchange");
    let local = local_geometry();
    let options = PrefixCacheOffload {
        memory_bytes: Some(0),
        disk: Some(dir.clone()),
        disk_limit: Some(1 << 20),
        ..Default::default()
    };
    let preparation = offload::prepare_with_inputs(
        &options,
        &dir,
        MODEL_IDENTITY,
        vec![local],
        vec![(rank, 0)],
        8,
        0,
        Some((rank, world)),
        &IdentityRegistrar,
    )
    .expect("prepare peer disk store");
    let Preparation::Store(_) = &preparation else {
        panic!("expected disk store preparation");
    };
    let shared = preparation.into_shared().expect("shared peer preparation");
    let factory = FaultCopiers::new(fail_on_out_store);

    let service = join_peer(Some(&shared), rank, &factory, |record| {
        let mut stream = TcpStream::connect(exchange_addr).expect("connect to coordinator");
        stream
            .write_all(&(rank as u32).to_le_bytes())
            .expect("send rank");
        write_blob(&mut stream, record);
        let mut all = Vec::with_capacity(world);
        for _ in 0..world {
            all.push(read_blob(&mut stream));
        }
        Ok(all)
    })
    .expect("peer join")
    .expect("peer service");

    service.join().expect("peer service clean shutdown");
}

struct ChildGuard(Vec<Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn run_partial_failure_scenario(name: &str, fail_rank: usize) {
    let Some(base_dir) = probe_direct_io_dir(name) else {
        return;
    };
    let local = local_geometry();
    let rank_roots: Vec<PathBuf> = (0..WORLD)
        .map(|rank| base_dir.join(format!("rank-{rank}")))
        .collect();
    let record_dirs: Vec<PathBuf> = (0..WORLD)
        .map(|rank| shard_record_dir(&rank_roots[rank], rank, WORLD, &local))
        .collect();

    let zero_options = PrefixCacheOffload {
        memory_bytes: Some(0),
        disk: Some(rank_roots[0].clone()),
        disk_limit: Some(1 << 20),
        ..Default::default()
    };
    let zero_prep = offload::prepare_with_inputs(
        &zero_options,
        &base_dir,
        MODEL_IDENTITY,
        vec![local.clone()],
        vec![(0, 0)],
        8,
        0,
        Some((0, WORLD)),
        &IdentityRegistrar,
    )
    .expect("prepare rank 0 store")
    .into_shared()
    .expect("rank 0 shared preparation");

    let exchange_listener = TcpListener::bind("127.0.0.1:0").expect("bind exchange listener");
    let exchange_addr = exchange_listener.local_addr().expect("exchange local addr");
    let exe = std::env::current_exe().expect("current test executable");

    let mut children = ChildGuard(Vec::new());
    for rank in 1..WORLD {
        let mut cmd = Command::new(&exe);
        cmd.arg("--worker")
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--world")
            .arg(WORLD.to_string())
            .arg("--dir")
            .arg(&rank_roots[rank])
            .arg("--exchange")
            .arg(exchange_addr.to_string())
            .stdin(Stdio::null());
        if rank == fail_rank {
            cmd.arg("--fail-store").arg("2");
        }
        children
            .0
            .push(cmd.spawn().expect("spawn peer rank process"));
    }

    let loopback: SocketAddr = "127.0.0.1:1".parse().expect("loopback addr");
    let peer_addresses = vec![(loopback, loopback); WORLD - 1];
    join_rank_zero(Some(&zero_prep), &peer_addresses, |offers| {
        let mut streams: Vec<Option<TcpStream>> = (1..WORLD).map(|_| None).collect();
        let mut replies: Vec<Vec<u8>> = vec![Vec::new(); WORLD];
        for _ in 1..WORLD {
            let (mut stream, _) = exchange_listener.accept().expect("accept peer");
            let mut rank_buf = [0u8; 4];
            stream.read_exact(&mut rank_buf).expect("read peer rank");
            let rank = u32::from_le_bytes(rank_buf) as usize;
            assert!((1..WORLD).contains(&rank), "unexpected peer rank {rank}");
            replies[rank] = read_blob(&mut stream);
            streams[rank - 1] = Some(stream);
        }
        for rank in 1..WORLD {
            let stream = streams[rank - 1].as_mut().expect("peer stream");
            write_blob(stream, &offers[rank]);
            for peer_reply in &replies[1..] {
                write_blob(stream, peer_reply);
            }
        }
        Ok(replies)
    })
    .expect("rank 0 join");

    let zero_factory = FaultCopiers::new((fail_rank == 0).then_some(2));
    let mut store =
        offload::open(zero_prep.take().expect("take rank 0 prep"), &zero_factory).expect("open");

    let mut history = SimpleHistory::new(MODEL_IDENTITY);
    history.append(&[11; 257]);
    let key_ok = history.key_at(128).expect("key_ok");
    let key_fail = history.key_at(256).expect("key_fail");

    // First checkpoint succeeds on all ranks and leaves record files in every rank directory.
    let ok_checkpoint = PublishedCheckpoint {
        capture_role: CaptureRole::Endpoint,
        id: CheckpointId::new(1),
        key: key_ok,
        ancestors: Vec::new(),
        bucket: key_ok.digest,
        chain_pages: vec![PageId(1), PageId(2)],
        tail_pages: 2,
        slot: StateSlotId(1),
    };
    assert_eq!(
        store.store(&ok_checkpoint).expect("queue ok store"),
        StoreOutcome::Queued
    );
    assert!(matches!(
        completed_event(store.as_mut()),
        StoreEvent::Stored { result: Ok(_), .. }
    ));
    let ok_file = record::file_name(&key_ok);
    for (rank, dir) in record_dirs.iter().enumerate() {
        assert!(
            dir.join(&ok_file).is_file(),
            "{name}: rank {rank} missing committed file {}",
            dir.join(&ok_file).display()
        );
    }

    // Second checkpoint fails on `fail_rank` while the other two ranks commit files to disk.
    // Rank zero must unlink the surviving ranks' files before reporting the store error.
    let fail_checkpoint = PublishedCheckpoint {
        capture_role: CaptureRole::Endpoint,
        id: CheckpointId::new(2),
        key: key_fail,
        ancestors: vec![key_ok],
        bucket: key_fail.digest,
        chain_pages: vec![PageId(1), PageId(2), PageId(3), PageId(4)],
        tail_pages: 2,
        slot: StateSlotId(2),
    };
    assert_eq!(
        store.store(&fail_checkpoint).expect("queue failing store"),
        StoreOutcome::Queued
    );
    match completed_event(store.as_mut()) {
        StoreEvent::Stored {
            result: Err(error), ..
        } => {
            let message = error.to_string();
            assert!(
                message.contains(&format!("rank {fail_rank}")),
                "{name}: error should name failing rank {fail_rank}: {message}"
            );
            assert!(
                message.contains("injected store failure"),
                "{name}: error should preserve injected failure: {message}"
            );
            assert!(
                !message.contains("cleanup unlink failed"),
                "{name}: cleanup unlink must succeed cleanly: {message}"
            );
        }
        other => panic!("{name}: expected store failure, got {other:?}"),
    }

    let fail_file = record::file_name(&key_fail);
    for (rank, dir) in record_dirs.iter().enumerate() {
        assert!(
            !dir.join(&fail_file).exists(),
            "{name}: rank {rank} leaked partial store file {}",
            dir.join(&fail_file).display()
        );
        assert!(
            dir.join(&ok_file).is_file(),
            "{name}: rank {rank} lost earlier committed file {}",
            dir.join(&ok_file).display()
        );
    }

    // The index must only expose the earlier committed checkpoint, which remains fetchable.
    let hit = store
        .lookup(&history, 0, u64::MAX)
        .expect("earlier checkpoint must remain indexed");
    assert_eq!(hit.consumed_tokens, key_ok.consumed_tokens);
    store
        .fetch(
            hit,
            InboundTicket {
                key: key_ok,
                pages: vec![PageId(11), PageId(12)],
                slot: StateSlotId(9),
            },
        )
        .expect("fetch surviving checkpoint");
    assert!(matches!(
        completed_event(store.as_mut()),
        StoreEvent::Fetched { result: Ok(_), .. }
    ));

    store.close().expect("close rank 0 store");
    drop(store);

    for (index, child) in children.0.drain(..).enumerate() {
        let mut child = child;
        let status = child.wait().expect("wait peer child");
        assert!(
            status.success(),
            "{name}: peer rank {} exited with {status}",
            index + 1
        );
    }
    fs::remove_dir_all(&base_dir).expect("cleanup test directory");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--worker") {
        run_worker(&args[1..]);
        return;
    }

    run_partial_failure_scenario("peer-fails-unlinks-zero-and-sibling", 2);
    run_partial_failure_scenario("zero-fails-unlinks-all-peers", 0);
}
