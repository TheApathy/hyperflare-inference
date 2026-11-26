/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use flare_selene::CacheGeometry;
use flare_selene::pool::{PoolPeers, PreparedPool};
use crate::offload::PoolOptions;
use flare_selene::{pool::PoolStatus, testing::TrackingRegistrar as Registrar};
use selene_manager::testing::{Manager, wait};
use std::{sync::Arc, thread};
fn unlimited() -> Budget {
    Budget {
        host: Ok(u64::MAX),
        devices: vec![(u64::MAX, false)],
    }
}
#[test]
fn peer_registration_budget_fails_startup() {
    use super::super::shard::{PoolOffer, pool_peer, pool_rank_zero};
    use crate::execution::RankStream;
    use std::net::{TcpListener, TcpStream};

    let mut manager = Manager::new();
    manager.run();
    for rejection in 0..3 {
        let reject = rejection != 2;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let peer_stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let zero_stream = listener.accept().unwrap().0;
        let options = PoolOptions { address: manager.address(), ca: None };
        let peer_registrar = Arc::new(Registrar::default());
        let registered = peer_registrar.clone();
        let offer = |rank| PoolOffer {
            enabled: true,
            identity: [3; 32],
            layout: [4; 16],
            codec_profile: None,
            rank,
            world: 2,
            gpu_uuid: [rank as u8 + 1; 16],
        };
        let peer = thread::spawn(move || {
            let peers = pool_peer(offer(1), &mut RankStream::Tcp(peer_stream)).unwrap();
            let budget = match rejection {
                0 => Budget {
                    host: Ok(1),
                    ..unlimited()
                },
                1 => Budget {
                    devices: vec![(1, false)],
                    ..unlimited()
                },
                _ => unlimited(),
            };
            start_with_registrar(
                &options,
                [3; 32],
                &[flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)],
                vec![[2; 16]],
                0,
                budget,
                &StartupProgress::silent(),
                registered,
                peers,
                false,
            )
        });
        let peers = pool_rank_zero(offer(0), &mut [RankStream::Tcp(zero_stream)]).unwrap();
        let (mut zero, _) = start_with_registrar(
            &PoolOptions { address: manager.address(), ca: None },
            [3; 32],
            &[flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)],
            vec![[1; 16]],
            0,
            unlimited(),
            &StartupProgress::silent(),
            Arc::new(Registrar::default()),
            peers,
            false,
        )
        .unwrap();
        let result = peer.join().unwrap();
        if reject {
            let reason = result
                .err()
                .expect("an unapproved peer must fail startup")
                .to_string();
            assert!(reason.contains("budget"), "{reason}");
            assert!(reason.contains("remove --prefix-cache-pool"), "{reason}");
            assert!(peer_registrar.calls.lock().unwrap().is_empty());
            wait("startup", || {
                matches!(
                    zero.client.as_ref().unwrap().status(),
                    PoolStatus::Disabled(_)
                )
            });
            zero.close().unwrap();
            continue;
        }
        let (peer, _) = result.unwrap();
        assert!(peer.agent.is_some() && peer.client.is_none() && peer.records.is_none());
        assert!(peer.approved());
        wait("startup", || zero.client.as_ref().unwrap().status() == PoolStatus::Ready);
        wait("startup", || peer.agent.as_ref().unwrap().status() == PoolStatus::Ready);
        assert_eq!(
            zero.identity.layout,
            flare_selene::geometry::fingerprint(&vec![
                flare_selene::testing::geometry(
                    64,
                    &[("kv", 4096)],
                    4096
                );
                2
            ])
        );
        zero.close().unwrap();
        let options = PoolOptions { address: manager.address(), ca: None };
        let options = super::super::PrefixCacheOffload {
            pool: Some(options),
            ..Default::default()
        };
        let mut prepared = super::super::prepare_with_inputs(
            &options,
            std::path::Path::new("model.safetensors"),
            [3; 32],
            vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)],
            vec![(0, 0)],
            1,
            0,
            Some((1, 2)),
        )
        .unwrap();
        let super::super::Preparation::Store(resources) = &mut prepared else {
            panic!("no pool resources")
        };
        resources.pool = Some(peer);
        let shared = prepared.into_shared().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let zero = listener.accept().unwrap().0;
        let service = thread::spawn(move || {
            super::super::shard::join_peer(
                Some(&shared),
                Vec::new(),
                &mut RankStream::Tcp(stream),
                1,
                2,
            )
                .unwrap()
                .unwrap()
        });
        assert!(
            super::super::shard::join_rank_zero(None, &mut [RankStream::Tcp(zero)])
                .unwrap()
                .is_none()
        );
        service.join().unwrap().join().unwrap();
    }
}

#[test]
fn multi_host_pool_only_rank_zero_and_peer_join_and_are_ready() {
    use super::super::shard::{PoolOffer, pool_peer, pool_rank_zero};
    use crate::execution::RankStream;
    use std::net::{TcpListener, TcpStream};

    let pair = || {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let server = listener.accept().unwrap().0;
        (server, client)
    };

    let mut manager = Manager::new();
    manager.run();
    let (a, b) = pair();
    let mut streams = [RankStream::Tcp(a)];
    let mut peer_stream = RankStream::Tcp(b);
    let identity = [11; 32];
    let geometry = vec![flare_selene::testing::geometry(64, &[("kv", 4096)], 4096)];
    let layout = flare_selene::geometry::fingerprint(&geometry);
    let peer_offer = PoolOffer {
        enabled: true,
        identity,
        layout,
        codec_profile: None,
        rank: 1,
        world: 2,
        gpu_uuid: [2; 16],
    };
    let zero_offer = PoolOffer {
        rank: 0,
        gpu_uuid: [1; 16],
        ..peer_offer.clone()
    };
    let peer_thread = thread::spawn(move || {
        pool_peer(peer_offer, &mut peer_stream).unwrap()
    });
    let zero_peers = pool_rank_zero(zero_offer, &mut streams).unwrap();
    let peer_peers = peer_thread.join().unwrap();
    let options = PoolOptions {
        address: manager.address(),
        ca: None,
    };
    let peer_options = options.clone();
    let peer_geometry = geometry.clone();
    let peer_start = thread::spawn(move || {
        start_with_registrar(
            &peer_options,
            identity,
            &peer_geometry,
            vec![[2; 16]],
            0,
            unlimited(),
            &StartupProgress::silent(),
            Arc::new(Registrar::default()),
            peer_peers,
            false,
        )
    });
    let (zero_pool, _) = start_with_registrar(
        &options,
        identity,
        &geometry,
        vec![[1; 16]],
        0,
        unlimited(),
        &StartupProgress::silent(),
        Arc::new(Registrar::default()),
        zero_peers,
        false,
    )
    .unwrap();
    let (peer_pool, _) = peer_start.join().unwrap().unwrap();
    wait("startup", || zero_pool.client.as_ref().unwrap().status() == PoolStatus::Ready);
    wait("startup", || peer_pool.agent.as_ref().unwrap().status() == PoolStatus::Ready);

    let offload_options = super::super::PrefixCacheOffload {
        pool: Some(options),
        ..Default::default()
    };
    let mut zero_prep = super::super::prepare_with_inputs(
        &offload_options,
        std::path::Path::new("model.safetensors"),
        identity,
        geometry.clone(),
        vec![(0, 0)],
        1,
        0,
        Some((0, 2)),
    )
    .unwrap();
    let super::super::Preparation::Store(zero_res) = &mut zero_prep else {
        panic!("expected pool-only rank-0 resources");
    };
    zero_res.pool = Some(zero_pool);
    let zero_shared = zero_prep.into_shared().unwrap();

    let mut peer_prep = super::super::prepare_with_inputs(
        &offload_options,
        std::path::Path::new("model.safetensors"),
        identity,
        geometry,
        vec![(0, 0)],
        1,
        0,
        Some((1, 2)),
    )
    .unwrap();
    let super::super::Preparation::Store(peer_res) = &mut peer_prep else {
        panic!("expected pool-only peer resources");
    };
    peer_res.pool = Some(peer_pool);
    let peer_shared = peer_prep.into_shared().unwrap();

    let (join_zero, join_peer) = pair();
    let peer_join = thread::spawn(move || {
        super::super::shard::join_peer(
            Some(&peer_shared),
            Vec::new(),
            &mut RankStream::Tcp(join_peer),
            1,
            2,
        )
        .unwrap()
        .unwrap()
    });
    let summary = super::super::shard::join_rank_zero(
        Some(&zero_shared),
        &mut [RankStream::Tcp(join_zero)],
    )
    .unwrap();
    assert_eq!(
        summary.as_deref(),
        Some("pool only across rank processes: restores only")
    );
    let binding = super::super::Binding::new(
        flare_selene::offload::Binding::new(zero_shared, 0),
        Vec::new(),
    );
    assert!(!binding.sharded(), "pool-only mode has no disk shards");
    assert!(
        binding.multi_host_ready(),
        "multi-host pool-only rank-0 store must be accepted for separate rank processes"
    );
    #[cfg(feature = "server")]
    crate::server::runtime::validate_multi_host_offload(true, Some(&binding)).unwrap();
    let (mut prepared, _) = binding.take().unwrap();
    prepared.pool.take().unwrap().close().unwrap();
    peer_join.join().unwrap().join().unwrap();
}

fn start_with_registrar(
    options: &PoolOptions,
    identity: [u8; 32],
    geometry: &[CacheGeometry],
    gpu_uuids: Vec<[u8; 16]>,
    registration_context: usize,
    budget: Budget,
    _: &StartupProgress,
    registrar: Arc<Registrar>,
    peers: Option<PoolPeers>,
    disk: bool,
) -> flare_selene::Result<(PreparedPool, Vec<u64>)> {
    PoolBuilder {
        options,
        identity,
        geometry,
        gpu_uuids,
        registration_context,
        budget,
        peers,
        disk,
    }
    .start(registrar)
}
