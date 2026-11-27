/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::Write;
use std::time::Duration;

use super::*;
use crate::execution::{CompletionRecord, Dtype};

#[test]
fn logical_layout_excludes_process_addresses_and_detects_contract_changes() {
    let manifest = super::super::tests::manifest();
    let expected = layout(&manifest);
    let mut peer = manifest.clone();
    peer.abi_version = 5;
    peer.context += 10;
    peer.device += 1;
    for buffer in &mut peer.buffers {
        buffer.address += 0x10000;
    }
    peer.graphs[0].handle += 0x10000;
    assert_eq!(expected, layout(&peer));
    peer.buffers[3].dtype = Dtype::Float32;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.graphs[0].token_capacity += 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.graphs[0].prefill_capacity = 64;
    assert_ne!(expected, layout(&peer));
    peer = manifest;
    peer.graphs[0].input_bindings[2].host_offset += 8;
    assert_ne!(expected, layout(&peer));
}

#[test]
fn rank_layout_includes_resolved_completion_capacity() {
    let manifest = crate::execution::tests::manifest();
    let mut peer = manifest.clone();
    peer.completion_token_capacity = 33;
    assert_ne!(layout(&manifest), layout(&peer));
}

#[test]
fn rank_layout_requires_equal_cache_capacities_and_snapshot_exclusions() {
    let mut manifest = super::super::tests::manifest();
    let mut cache = flare_selene::testing::geometry(64, &[("kv", 4096)], 65536);
    cache.snapshot_pages = vec![100..120, 115..135];
    manifest.cache = Some(cache);
    let expected = layout(&manifest);
    let mut peer = manifest.clone();
    let cache = peer.cache.as_mut().unwrap();
    cache.row.address += 0x10000;
    cache.planes[0].address += 0x20000;
    assert_eq!(expected, layout(&peer));
    peer.cache.as_mut().unwrap().snapshot_pages[0].start += 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.cache.as_mut().unwrap().page_capacity -= 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest.clone();
    peer.cache.as_mut().unwrap().row_capacity -= 1;
    assert_ne!(expected, layout(&peer));
    peer = manifest;
    peer.cache = None;
    assert_ne!(expected, layout(&peer));
}

#[test]
fn fragmented_completion_remains_pending_until_all_bytes_arrive() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut channel = Channel::new(receiver).unwrap();
    channel.nonblocking(true).unwrap();
    sender.write_all(&[COMPLETE, 3]).unwrap();
    assert!(channel.poll().unwrap().is_none());
    sender.write_all(&[0, 0, 0, 10]).unwrap();
    assert!(channel.poll().unwrap().is_none());
    sender.write_all(&[11, 12]).unwrap();
    assert_eq!(channel.poll().unwrap(), Some((COMPLETE, vec![10, 11, 12])));
    drop(sender);
    assert!(
        channel
            .poll()
            .unwrap_err()
            .to_string()
            .contains("disconnected")
    );
}

#[test]
fn packet_bounds_and_peer_errors_are_reported() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut channel = Channel::new(receiver).unwrap();
    sender.write_all(&[COMPLETE, 255, 255, 255, 255]).unwrap();
    assert!(
        channel
            .poll()
            .unwrap_err()
            .to_string()
            .contains("oversized")
    );
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut sender = Channel::new(sender).unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    sender.send(FAILED, b"rank fence differs").unwrap();
    assert!(
        receiver
            .receive(READY)
            .unwrap_err()
            .to_string()
            .contains("rank fence differs")
    );
}

#[cfg(unix)]
fn assert_blocking(channel: &Channel) {
    use std::os::fd::AsRawFd;
    let fd = match &channel.stream {
        RankStream::Unix(stream) => stream.as_raw_fd(),
        RankStream::Tcp(stream) => stream.as_raw_fd(),
    };
    // SAFETY: F_GETFL reads flags from the live socket without retaining its descriptor.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    assert!(flags >= 0);
    assert_eq!(flags & libc::O_NONBLOCK, 0);
}

#[test]
fn receive_preserves_control_tags_on_both_transports() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    for (sender, receiver) in [crate::execution::rank_stream::test_pair().unwrap(),
                               (client.into(), server.into())] {
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        for tag in [READY, COMMIT, PREPARE, DRAIN] {
            sender.send(tag, &[tag]).unwrap();
            assert_eq!(receiver.receive(tag).unwrap(), vec![tag]);
            #[cfg(unix)]
            assert_blocking(&receiver);
        }
    }
}

#[test]
fn nonblocking_poll_preserves_fragment_for_receive_and_idle_wait() {
    let (mut sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    sender.write_all(&[PREPARE, 3, 0]).unwrap();
    receiver.nonblocking(true).unwrap();
    assert!(receiver.poll().unwrap().is_none());
    receiver.nonblocking(false).unwrap();
    #[cfg(unix)]
    assert_blocking(&receiver);
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        sender.write_all(&[0, 0, 10, 11, 12]).unwrap();
        let mut sender = Channel::new(sender).unwrap();
        sender.send(DRAIN, &[]).unwrap();
    });
    assert_eq!(receiver.receive(PREPARE).unwrap(), vec![10, 11, 12]);
    assert_eq!(receiver.wait(true).unwrap(), (DRAIN, vec![]));
    #[cfg(unix)]
    assert_blocking(&receiver);
    writer.join().unwrap();
}

#[test]
fn receive_reports_wrong_tags_peer_errors_and_disconnects() {
    for tag in [COMMIT, FAILED] {
        let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        sender.send(tag, b"rank failed").unwrap();
        let error = receiver.receive(READY).unwrap_err().to_string();
        assert!(error.contains(if tag == FAILED { "rank failed" } else { "received 4" }), "{error}");
        #[cfg(unix)]
        assert_blocking(&receiver);
    }
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    drop(sender);
    assert!(receiver.receive(READY).unwrap_err().to_string().contains("disconnected"));
    #[cfg(unix)]
    assert_blocking(&receiver);
}

#[test]
fn upload_decoder_rejects_unbounded_or_truncated_fragments() {
    let manifest = super::super::tests::manifest();
    let mut bytes = Vec::new();
    number(&mut bytes, usize::MAX);
    let mut remaining = manifest.input_frame_bytes;
    assert!(uploads(&manifest, &mut Reader(&bytes), &mut remaining).is_err());
    bytes.clear();
    for number_value in [1, 3, 8, 8] {
        number(&mut bytes, number_value);
    }
    bytes.extend_from_slice(&7_i64.to_le_bytes());
    let mut remaining = manifest.input_frame_bytes;
    let decoded = uploads(&manifest, &mut Reader(&bytes), &mut remaining).unwrap();
    assert_eq!(decoded[0].buffer, "tokens");
    assert_eq!(decoded[0].offset, 8);
    assert_eq!(decoded[0].bytes, 7_i64.to_le_bytes());
    let mut remaining = manifest.input_frame_bytes;
    assert!(uploads(&manifest, &mut Reader(&bytes[..bytes.len() - 1]), &mut remaining).is_err());
}

#[test]
fn prepare_packets_preserve_large_and_partial_uploads_on_both_transports() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    let mut bytes = vec![0x5a; 80 * 1024 * 1024];
    bytes[..8].copy_from_slice(&1u64.to_le_bytes());
    let end = bytes.len();
    bytes[end - 8..].copy_from_slice(&2u64.to_le_bytes());
    let mut manifest = super::super::tests::manifest();
    let buffer = &mut manifest.buffers[3];
    buffer.nbytes = bytes.len() + 16;
    buffer.shape = vec![buffer.nbytes / 8];
    manifest.graphs[0].input_bindings[2].nbytes = buffer.nbytes;
    manifest.input_frame_bytes = 16 + buffer.nbytes;
    manifest.validate().unwrap();
    assert_eq!(u64::from_le_bytes(layout(&manifest)[..8].try_into().unwrap()), 6);
    for (sender, receiver) in [crate::execution::rank_stream::test_pair().unwrap(),
                               (client.into(), server.into())] {
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        for parts in [1, 2] {
            std::thread::scope(|scope| {
                let writer = scope.spawn(|| {
                    let uploads: Vec<_> = bytes.chunks(bytes.len() / parts).enumerate()
                        .map(|(index, bytes)| Upload {
                            buffer: "tokens", offset: 8 + index * 80 * 1024 * 1024 / parts, bytes,
                        }).collect();
                    let mut packets = 0;
                    wire::prepare(&manifest, Fence(7), 0, true, &uploads, |tag, data| {
                        assert!(data.len() <= 64 * 1024 * 1024);
                        assert_eq!(tag, if packets == 0 { PREPARE_MORE } else { PREPARE });
                        packets += 1;
                        sender.send(tag, data)
                    }).unwrap();
                    assert_eq!(packets, 2);
                    sender.send(DRAIN, &[]).unwrap();
                });
                let (tag, packet) = receiver.wait(false).unwrap();
                let (fence, graph, grammar, owned) =
                    receive_prepare(&mut receiver, &manifest, tag, packet).unwrap();
                assert_eq!((fence, graph, grammar), (Fence(7), 0, true));
                assert!(owned.len() > parts);
                let mut offset = 8;
                for upload in &owned {
                    assert_eq!(upload.buffer, "tokens");
                    assert_eq!(upload.offset, offset);
                    assert_eq!(upload.bytes.len() % 8, 0);
                    assert_eq!(upload.bytes, bytes[offset - 8..offset - 8 + upload.bytes.len()]);
                    offset += upload.bytes.len();
                }
                assert_eq!(offset, bytes.len() + 8);
                let borrowed: Vec<_> = owned.iter().map(|upload| Upload {
                    buffer: &upload.buffer, offset: upload.offset, bytes: &upload.bytes,
                }).collect();
                super::super::transport::validate_uploads(&manifest, graph, &borrowed, &mut Vec::new()).unwrap();
                assert!(receiver.receive(DRAIN).unwrap().is_empty());
                writer.join().unwrap();
            });
        }
    }
}

#[test]
fn small_prepare_preserves_one_packet_and_upload_extents() {
    let manifest = super::super::tests::manifest();
    let bytes = 7_i64.to_le_bytes();
    let input = [Upload { buffer: "tokens", offset: 8, bytes: &bytes }];
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut sender = Channel::new(sender).unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    wire::prepare(&manifest, Fence(3), 0, false, &input, |tag, data| {
        assert_eq!(tag, PREPARE);
        sender.send(tag, data)
    }).unwrap();
    let (tag, packet) = receiver.wait(false).unwrap();
    let (fence, graph, grammar, owned) = receive_prepare(&mut receiver, &manifest, tag, packet).unwrap();
    assert_eq!((fence, graph, grammar), (Fence(3), 0, false));
    assert_eq!(owned.len(), 1);
    assert_eq!(owned[0].offset, 8);
    assert_eq!(owned[0].bytes, bytes);
}

#[test]
fn prepare_continuations_reject_identity_changes_wrong_tags_and_frame_overflow() {
    let manifest = super::super::tests::manifest();
    let packet = |fence, graph, grammar, size: usize| {
        let mut bytes = Vec::new();
        for value in [fence, graph, grammar, 1, 3, 0, size] {
            number(&mut bytes, value);
        }
        bytes.resize(bytes.len() + size, 0);
        bytes
    };
    for (tag, next, expected) in [
        (PREPARE, packet(8, 0, 0, 8), "identity differs"),
        (PREPARE, packet(7, 1, 0, 8), "identity differs"),
        (PREPARE, packet(7, 0, 1, 8), "identity differs"),
        (PREPARE, packet(7, 0, 2, 8), "invalid grammar"),
        (COMMIT, vec![], "expected rank prepare"),
        (PREPARE, packet(7, 0, 0, 32), "input-frame extent"),
    ] {
        let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
        let mut sender = Channel::new(sender).unwrap();
        let mut receiver = Channel::new(receiver).unwrap();
        sender.send(tag, &next).unwrap();
        let error = receive_prepare(&mut receiver, &manifest, PREPARE_MORE, packet(7, 0, 0, 24))
            .err().unwrap().to_string();
        assert!(error.contains(expected), "{error}");
    }
    let (sender, receiver) = crate::execution::rank_stream::test_pair().unwrap();
    let mut receiver = Channel::new(receiver).unwrap();
    drop(sender);
    let error = receive_prepare(&mut receiver, &manifest, PREPARE_MORE, packet(7, 0, 0, 8))
        .err().unwrap().to_string();
    assert!(error.contains("disconnected"), "{error}");
    let mut empty = Vec::new();
    for value in [7, 0, 0, 0] { number(&mut empty, value); }
    let error = receive_prepare(&mut receiver, &manifest, PREPARE_MORE, empty).err().unwrap().to_string();
    assert!(error.contains("empty rank prepare"), "{error}");
}

fn completed(status: i64) -> FrameCompletion {
    let record = CompletionRecord {
        generation: 7,
        submission_sequence: 3,
        consumed_tokens: 65,
        generated_tokens: 0,
        token_id: -1,
        status,
        error_code: 0,
        reserved: 0,
        vector: Some(super::super::TokenVector { tokens: vec![-1; 16], ..Default::default() }),
    };
    FrameCompletion {
        logprobs: None,
        prompt_logprobs: None,
        proposal_confidence: None,
        speculative_survival: None,
        fence: Fence(3),
        graph_index: 2,
        timing: Default::default(),
        records: vec![record.clone(); 4].into(),
    }
}

#[test]
fn peers_agree_on_request_consumption_without_sampling() {
    let mut root = completed(2);
    root.records[0].generated_tokens = 2;
    root.records[0].token_id = 19;
    let mut peer = completed(1);
    agree(&root, &peer).unwrap();
    peer.records[0].consumed_tokens -= 1;
    let error = agree(&root, &peer).unwrap_err().to_string();
    assert!(error.contains("graph 2, fence 3, lane 0"), "{error}");
    assert!(error.contains("consumed_tokens: 65"), "{error}");
    assert!(error.contains("consumed_tokens: 64"), "{error}");
    peer = completed(1);
    peer.records[0].generation += 1;
    assert!(agree(&root, &peer).is_err());
    peer = completed(3);
    assert!(agree(&root, &peer).is_err());
    root = completed(3);
    agree(&root, &peer).unwrap();
}

#[test]
fn completion_wire_preserves_the_execution_abi() {
    let mut root = completed(1);
    root.timing.gpu_ns[0] = Some(1250);
    root.timing.wall_ns = 2345;
    root.timing.queued = true;
    root.timing.readiness = Some(super::super::ReadinessIntervals::decode(
        &[100u64, 110, 111, 200, 210, 215].into_iter().flat_map(u64::to_ne_bytes).collect::<Vec<_>>()
    ).unwrap());
    let bytes = completion(&root);
    let mut reader = Reader(&bytes);
    assert_eq!(reader.number().unwrap(), 3);
    assert_eq!(reader.number().unwrap(), 2);
    assert_eq!(timing(&mut reader).unwrap(), root.timing);
    assert_eq!(proposal_confidence(&mut reader, 4, 16).unwrap(), root.proposal_confidence);
    assert_eq!(speculative_survival(&mut reader, 4, 16).unwrap(), root.speculative_survival);
    let records = super::super::transport::decode_records(reader.0, Fence(3), 4, 32).unwrap();
    for (expected, actual) in root.records.iter().zip(records.iter()) {
        assert_eq!(expected.fields(), actual.fields());
    }
}

#[test]
fn speculative_survival_is_optional_and_validated_per_rank_without_numeric_consensus() {
    let mut root = completed(1);
    for record in root.records.iter_mut() {
        record.vector = Some(crate::execution::TokenVector {
            emitted_count: 1, offered_drafts: 3, accepted_drafts: 0, verification_width: 4,
            tokens: vec![-1; 16],
            ..Default::default()
        });
    }
    let mut values = vec![vec![-1.; 15]; 4];
    values[..4].fill(vec![0.75, 0.5, 0.25, -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]);
    root.speculative_survival = Some(values.clone());
    let mut peer = completed(1);
    peer.records = root.records.clone();
    assert!(agree(&root, &peer).is_err());
    peer.speculative_survival = Some(values.clone());
    peer.speculative_survival.as_mut().unwrap()[1][1] = 0.6;
    agree(&root, &peer).unwrap();
    assert_eq!(root.speculative_survival, Some(values.clone()));
    for (slot, depth, invalid) in [(0, 0, f32::NAN), (0, 0, 1.1), (0, 0, -0.1),
        (0, 1, 0.8), (0, 1, -1.), (0, 2, -1.), (0, 3, 0.1)] {
        peer.speculative_survival = Some(values.clone());
        peer.speculative_survival.as_mut().unwrap()[slot][depth] = invalid;
        assert!(agree(&root, &peer).is_err(), "slot {slot}, depth {depth}, value {invalid}");
    }
    let bytes = completion(&root);
    let mut reader = Reader(&bytes);
    reader.number().unwrap();
    reader.number().unwrap();
    timing(&mut reader).unwrap();
    proposal_confidence(&mut reader, 4, 16).unwrap();
    let offset = bytes.len() - reader.0.len();
    assert_eq!(speculative_survival(&mut reader, 4, 16).unwrap(), Some(values.clone()));
    for length in [0, 1, 7, 8, 9, 8 + 4 * 15 * 4 - 1] {
        assert!(speculative_survival(&mut Reader(&bytes[offset..offset + length]), 4, 16).is_err());
    }
}

#[test]
fn rank_timing_rejects_truncation_and_preserves_zero_duration() {
    for gpu in [None, Some(0), Some(731)] {
        let mut root = completed(1);
        root.timing.gpu_ns[0] = gpu;
        root.timing.wall_ns = 1234;
        let bytes = completion(&root);
        for length in 0..40 {
            assert!(timing(&mut Reader(&bytes[16..16 + length])).is_err());
        }
        assert_eq!(timing(&mut Reader(&bytes[16..])).unwrap(), root.timing);
    }
    let mut invalid = vec![];
    for value in [2, 0, 1234, 0] {
        number(&mut invalid, value);
    }
    assert!(timing(&mut Reader(&invalid)).is_err());
}

#[test]
fn rank_completion_requires_every_record() {
    let root = completed(1);
    let mut peer = completed(1);
    peer.records = vec![peer.records[0].clone(); 3].into();
    assert!(agree(&root, &peer).is_err());
}

#[test]
fn proposal_confidence_survives_rank_transport_without_numeric_consensus() {
    let mut root = completed(1);
    root.proposal_confidence = Some(vec![vec![0.5; 15]; 4]);
    let mut peer = completed(1);
    assert!(agree(&root, &peer).is_err());
    peer.proposal_confidence = root.proposal_confidence.clone();
    agree(&root, &peer).unwrap();
    peer.proposal_confidence.as_mut().unwrap()[3][2] = 0.6;
    agree(&root, &peer).unwrap();
    let bytes = completion(&root);
    let mut reader = Reader(&bytes);
    reader.number().unwrap();
    reader.number().unwrap();
    timing(&mut reader).unwrap();
    assert_eq!(proposal_confidence(&mut reader, 4, 16).unwrap(), root.proposal_confidence);
    assert_eq!(speculative_survival(&mut reader, 4, 16).unwrap(), root.speculative_survival);
    assert_eq!(reader.0.len(), 4 * 32 * 8);
}

#[test]
fn proposal_confidence_validates_each_rank_without_changing_root_values() {
    let mut root = completed(1);
    let mut peer = completed(1);
    root.proposal_confidence = Some(vec![vec![0.0; 15]; 4]);
    peer.proposal_confidence = root.proposal_confidence.clone();
    root.proposal_confidence.as_mut().unwrap()[1][2] = f32::from_bits(0x3f79148d);
    peer.proposal_confidence.as_mut().unwrap()[1][2] = f32::from_bits(0x3f79149e);
    agree(&root, &peer).unwrap();
    peer.proposal_confidence.as_mut().unwrap()[1][2] = 0.973;
    let authoritative = root.proposal_confidence.clone();
    agree(&root, &peer).unwrap();
    assert_eq!(root.proposal_confidence, authoritative);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.000001, 1.000001] {
        root.proposal_confidence = authoritative.clone();
        peer.proposal_confidence = authoritative.clone();
        root.proposal_confidence.as_mut().unwrap()[1][2] = invalid;
        assert!(agree(&root, &peer).is_err());
        root.proposal_confidence = authoritative.clone();
        peer.proposal_confidence.as_mut().unwrap()[1][2] = invalid;
        assert!(agree(&root, &peer).is_err());
    }
}

#[test]
fn rank_zero_confidence_is_authoritative_without_relaxing_request_agreement() {
    let mut root = completed(1);
    let mut peer = completed(1);
    root.proposal_confidence = Some(vec![vec![0.5; 15]; 4]);
    peer.proposal_confidence = root.proposal_confidence.clone();
    root.proposal_confidence.as_mut().unwrap()[2][2] = 0.51028365;
    peer.proposal_confidence.as_mut().unwrap()[2][2] = 0.5102996;
    let authoritative = root.proposal_confidence.clone();
    agree(&root, &peer).unwrap();
    assert_eq!(root.proposal_confidence, authoritative);
    peer.records[0].consumed_tokens += 1;
    assert!(agree(&root, &peer).is_err());
}

#[test]
fn speculative_process_schema_rejects_a_different_proposal_depth() {
    let mut root = crate::execution::tests::manifest();
    let graph = &mut root.graphs[0];
    graph.kind = crate::execution::GraphKind::Speculative;
    graph.token_capacity = 6;
    graph.packed_token_capacity = 8;
    graph.draft_tokens = 1;
    let mut peer = root.clone();
    assert_eq!(layout(&root), layout(&peer));
    peer.graphs[0].draft_tokens = 3;
    assert_ne!(layout(&root), layout(&peer), "different proposal contracts must fail before any rank launch");
}

#[test]
fn logprob_uploads_budget_count_and_prefix_wire_bytes_separately() {
    let manifest = super::super::tests::with_logprobs(crate::execution::MAX_LOGPROB_TOKEN_IDS);
    let measure = |uploads: &[Upload<'_>]| {
        let mut size = 0;
        wire::prepare(&manifest, Fence(1), 0, false, uploads, |_, bytes| {
            size += bytes.len();
            Ok(())
        })
        .unwrap();
        size
    };
    let empty = measure(&[]);
    let counts = [0u8; 16];
    assert_eq!(
        measure(&[Upload {
            buffer: "logprob_token_id_counts",
            offset: 0,
            bytes: &counts
        }]) - empty,
        16 + 24
    );
    for count in [1, 20, 21, 127, 128] {
        let ids: Vec<_> = (0..count as i32).flat_map(i32::to_ne_bytes).collect();
        let additional = measure(&[
            Upload {
                buffer: "logprob_token_id_counts",
                offset: 0,
                bytes: &counts,
            },
            Upload {
                buffer: "logprob_token_ids",
                offset: 0,
                bytes: &ids,
            },
        ]) - empty;
        assert_eq!(additional, 16 + 24 + 4 * count + 24);
        eprintln!(
            "C=4 N={count}: H2D={} remote_peer={additional}",
            16 + 4 * count
        );
    }
}

#[test]
fn two_frame_window_retains_preparing_and_unconsumed_completions() {
    let mut manifest = crate::execution::tests::manifest();
    manifest.graphs[0].kind = crate::execution::GraphKind::Decode;
    let frame = |sequence| Pending {
        fence: Fence(sequence), graph: 0, grammar: false, uploads: Vec::new(),
        dispatched: Instant::now(), committed: false, queued: sequence > 1,
        ready: vec![false; 3], completions: (0..4).map(|_| None).collect(),
    };
    let mut pending = VecDeque::new();
    assert!(window_available(&pending, false, &manifest));
    pending.push_back(frame(1));
    assert!(!window_available(&pending, false, &manifest));
    for rank in 0..3 { ready_frame(&mut pending, rank, Fence(1), 0).unwrap(); }
    assert!(pending[0].ready.iter().all(|ready| *ready));
    pending[0].committed = true;
    assert!(window_available(&pending, false, &manifest));
    assert!(!window_available(&pending, true, &manifest));
    pending.push_back(frame(2));
    for rank in 0..3 { ready_frame(&mut pending, rank, Fence(2), 0).unwrap(); }
    pending[1].committed = true;
    assert!(!window_available(&pending, false, &manifest));
    for rank in 0..3 {
        assert!(completion_frame(&mut pending, rank, Fence(2), 0).is_err());
        let mut value = completed(1);
        value.graph_index = 0;
        value.fence = Fence(1);
        for record in value.records.iter_mut() { record.submission_sequence = 1; }
        completion_frame(&mut pending, rank, Fence(1), 0).unwrap().completions[rank + 1] = Some(value);
    }
    pending[0].completions[0] = Some(completed(1));
    assert!(!window_available(&pending, false, &manifest));
    pending.pop_front();
    assert!(window_available(&pending, false, &manifest));
    assert!(ready_frame(&mut pending, 0, Fence(2), 0).is_err());
    assert!(ready_frame(&mut pending, 0, Fence(3), 0).is_err());
    assert!(completion_frame(&mut pending, 0, Fence(1), 0).is_err());
    assert!(completion_frame(&mut pending, 0, Fence(2), 1).is_err());
    completion_frame(&mut pending, 0, Fence(2), 0).unwrap();
}

#[test]
fn rank_commit_identity_rejects_truncation_and_trailing_bytes() {
    let bytes = identity(Fence(7), 3);
    assert_eq!(read_identity(&bytes).unwrap(), (Fence(7), 3));
    assert!(read_identity(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(read_identity(&trailing).is_err());
}

#[test]
fn startup_tuning_rank_agreement_includes_producers_bindings_and_buffer_schemas() {
    let value = super::super::tests::tuning_manifest();
    let expected = layout(&value);
    let mut moved = value.clone();
    moved.tuning_graphs[0].handle += 32;
    assert_eq!(layout(&moved), expected);
    for mutation in 0..5 {
        let mut peer = value.clone();
        match mutation {
            0 => peer.tuning_graphs[0].name.push('x'),
            1 => peer.tuning_graphs[0].outputs[0].push('x'),
            2 => peer.tuning_graphs[0].status_buffer.push('x'),
            3 => peer.graphs[0].tuned_bindings[0].tuning_graph.push('x'),
            _ => peer.buffers.last_mut().unwrap().dtype = super::super::Dtype::Float32,
        }
        assert_ne!(layout(&peer), expected, "mutation {mutation}");
    }
}

#[test]
fn socket_completion_capacity_is_checked_before_rank_startup() {
    let mut manifest = crate::execution::tests::manifest();
    manifest.max_active = 64;
    manifest.completion_token_capacity = 65;
    validate_completion_packet(&manifest).unwrap();
    manifest.completion_token_capacity = 131072;
    assert!(validate_completion_packet(&manifest).unwrap_err().to_string().contains("packet capacity"));
    manifest.completion_token_capacity = 65536;
    validate_completion_packet(&manifest).unwrap();
    manifest.graphs[0].confidence_width = 1;
    manifest.buffers.push(crate::execution::BufferDescriptor {
        name: "completion_speculative_survival".into(), address: 1, nbytes: 1,
        dtype: crate::execution::Dtype::Float32, shape: vec![2, 64, 65535], strides: vec![64 * 65535, 65535, 1],
    });
    assert!(validate_completion_packet(&manifest).is_err());
}

#[test]
fn socket_parallel_execution_rejects_persistent_workers() {
    let mut manifest = crate::execution::tests::manifest();
    manifest.persistent_workers.push(crate::execution::PersistentWorkerDescriptor {
        worker_sm_count: 8,
        graph: "worker_loop".into(),
        handle: 0x99000,
        stop_flag_allocation: "worker_ctrl".into(),
        stop_flag_offset: 0,
        stop_flag_address: 0x2000_0000_0000,
        metrics_allocation: None,
        metrics_offset: 0,
        metrics_address: 0,
        metrics_count: 0,
        control: None,
    });
    let (root_stream, peer_stream) = crate::execution::rank_stream::test_pair().unwrap();
    let root_err = unsafe { TensorParallelExecutor::new(manifest.clone(), vec![root_stream]) }
        .err()
        .expect("TensorParallelExecutor::new must reject persistent workers");
    assert!(
        root_err.to_string().contains("single-rank"),
        "unexpected error: {root_err}"
    );
    let shutdown = unsafe { serve_rank(manifest, peer_stream, 1, 2) };
    assert!(shutdown.drained);
    let peer_err = shutdown
        .error
        .expect("serve_rank must reject persistent workers");
    assert!(
        peer_err.to_string().contains("single-rank"),
        "unexpected error: {peer_err}"
    );
}

#[cfg(unix)]
fn assert_nonblocking(channel: &Channel) {
    use std::os::fd::AsRawFd;
    let fd = match &channel.stream {
        RankStream::Unix(stream) => stream.as_raw_fd(),
        RankStream::Tcp(stream) => stream.as_raw_fd(),
    };
    // SAFETY: F_GETFL reads flags from the live socket without retaining its descriptor.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    assert!(flags >= 0);
    assert_ne!(flags & libc::O_NONBLOCK, 0);
}

#[test]
fn channel_caches_nonblocking_mode_and_restores_blocking_for_send_and_wait() {
    let (left, right) = crate::execution::rank_stream::test_pair().unwrap();
    let mut left = Channel::new(left).unwrap();
    let mut right = Channel::new(right).unwrap();
    assert!(!left.is_nonblocking());
    assert!(!right.is_nonblocking());
    #[cfg(unix)]
    {
        assert_blocking(&left);
        assert_blocking(&right);
    }

    right.nonblocking(true).unwrap();
    assert!(right.is_nonblocking());
    #[cfg(unix)]
    assert_nonblocking(&right);
    // Repeated polls in nonblocking mode leave the socket nonblocking and return None.
    right.nonblocking(true).unwrap();
    assert!(right.poll().unwrap().is_none());
    assert!(right.poll().unwrap().is_none());
    assert!(right.is_nonblocking());
    #[cfg(unix)]
    assert_nonblocking(&right);

    // Sending on a nonblocking channel restores blocking mode automatically.
    left.nonblocking(true).unwrap();
    assert!(left.is_nonblocking());
    left.send(READY, &[1, 2, 3, 4]).unwrap();
    assert!(!left.is_nonblocking());
    #[cfg(unix)]
    assert_blocking(&left);

    // Waiting on a nonblocking channel restores blocking mode and receives the packet.
    assert_eq!(right.wait(false).unwrap(), (READY, vec![1, 2, 3, 4]));
    assert!(!right.is_nonblocking());
    #[cfg(unix)]
    assert_blocking(&right);
}

#[test]
fn publish_completed_front_preserves_pending_frame_on_agreement_or_timing_rejection() {
    let mut manifest = crate::execution::tests::manifest();
    manifest.graphs[0].kind = crate::execution::GraphKind::Decode;

    let make_completion = |fence: u64, status: i64, gpu_ns: u64| {
        let mut value = completed(status);
        value.fence = Fence(fence);
        value.graph_index = 0;
        value.timing.gpu_ns[0] = Some(gpu_ns);
        value
    };
    let make_pending = |fence: u64, queued: bool, root: FrameCompletion, peer: FrameCompletion| {
        Pending {
            fence: Fence(fence),
            graph: 0,
            grammar: false,
            uploads: Vec::new(),
            dispatched: Instant::now(),
            committed: true,
            queued,
            ready: vec![true],
            completions: vec![Some(root), Some(peer)],
        }
    };

    // Agreement rejection preserves the failing front frame and its stored completions.
    let root = make_completion(1, 2, 100);
    let mut bad_peer = make_completion(1, 1, 200);
    bad_peer.records[0].consumed_tokens -= 1;
    let mut pending = VecDeque::from([
        make_pending(1, false, root, bad_peer),
        Pending {
            fence: Fence(2),
            graph: 0,
            grammar: false,
            uploads: Vec::new(),
            dispatched: Instant::now(),
            committed: false,
            queued: true,
            ready: vec![false],
            completions: vec![None, None],
        },
    ]);
    let error = publish_completed_front(&mut pending, &manifest, |f| f.queued, |f| f.dispatched)
        .unwrap_err()
        .to_string();
    assert!(error.contains("rank 1:"), "{error}");
    assert_eq!(pending.len(), 2);
    assert_eq!(pending[0].fence, Fence(1));
    assert!(pending[0].completions[0].is_some());
    assert!(pending[0].completions[1].is_some());
    assert_eq!(pending[0].completions[0].as_ref().unwrap().timing.gpu_ns[1], None);

    // Speculative agreement rejection preserves the failing front frame as well.
    let mut spec_manifest = manifest.clone();
    spec_manifest.graphs[0].kind = crate::execution::GraphKind::Speculative;
    let mut spec_root = make_completion(1, 1, 100);
    spec_root.records[0].token_id = 42;
    let mut spec_peer = make_completion(1, 1, 200);
    spec_peer.records[0].token_id = 43;
    pending[0].completions = vec![Some(spec_root), Some(spec_peer)];
    let error = publish_completed_front(
        &mut pending,
        &spec_manifest,
        |f| f.queued,
        |f| f.dispatched,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("rank 1:"), "{error}");
    assert_eq!(pending.len(), 2);
    assert_eq!(pending[0].fence, Fence(1));
    assert!(pending[0].completions[0].is_some());
    assert!(pending[0].completions[1].is_some());

    // Timing merge rejection preserves the failing front frame without partial mutation.
    let mut timed_root = make_completion(1, 1, 100);
    timed_root.timing.readiness = Some(
        super::super::ReadinessIntervals::decode(
            &[100u64, 110, 111]
                .into_iter()
                .flat_map(u64::to_ne_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    );
    let timed_peer = make_completion(1, 1, 250);
    pending[0].completions = vec![Some(timed_root), Some(timed_peer)];
    let error = publish_completed_front(&mut pending, &manifest, |f| f.queued, |f| f.dispatched)
        .unwrap_err()
        .to_string();
    assert!(error.contains("readiness timestamp bindings differ"), "{error}");
    assert_eq!(pending.len(), 2);
    assert_eq!(pending[0].fence, Fence(1));
    assert_eq!(pending[0].completions[0].as_ref().unwrap().timing.gpu_ns[1], None);

    // Valid agreement and timing merge pops only the front frame and merges peer timing.
    let good_root = make_completion(1, 2, 100);
    let good_peer = make_completion(1, 1, 250);
    pending[0].completions = vec![Some(good_root), Some(good_peer)];
    let published = publish_completed_front(&mut pending, &manifest, |f| f.queued, |f| f.dispatched)
        .unwrap()
        .expect("validated front frame should publish");
    assert_eq!(published.fence, Fence(1));
    assert_eq!(published.timing.gpu_ns[0], Some(100));
    assert_eq!(published.timing.gpu_ns[1], Some(250));
    assert!(!published.timing.queued);
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].fence, Fence(2));
}

#[test]
fn socket_partial_commit_dispatch_stops_at_failed_peer_and_preserves_prior_commits() {
    for failed_peer in 0..=3 {
        let mut peers = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..3 {
            let (coord, rank) = crate::execution::rank_stream::test_pair().unwrap();
            peers.push(Peer {
                channel: Channel::new(coord).unwrap(),
            });
            receivers.push(Some(Channel::new(rank).unwrap()));
        }
        if failed_peer < 3 {
            drop(receivers[failed_peer].take());
        }
        let result = dispatch_commit_peers(&mut peers, Fence(7), 1);
        if failed_peer < 3 {
            assert!(
                result.is_err(),
                "partial commit dispatch must fail when peer {failed_peer} is disconnected"
            );
        } else {
            result.unwrap();
        }
        for (index, receiver) in receivers.iter_mut().enumerate() {
            let Some(channel) = receiver.as_mut() else {
                continue;
            };
            channel.nonblocking(true).unwrap();
            if index < failed_peer {
                let (tag, bytes) = channel.wait(false).unwrap();
                assert_eq!(tag, COMMIT);
                assert_eq!(read_identity(&bytes).unwrap(), (Fence(7), 1));
                channel.nonblocking(true).unwrap();
                assert!(channel.poll().unwrap().is_none());
            } else {
                assert!(
                    channel.poll().unwrap().is_none(),
                    "peer {index} must not receive COMMIT after failed peer {failed_peer}"
                );
            }
        }
    }
}

#[test]
fn socket_startup_drain_withholds_release_unless_every_peer_and_local_drain() {
    // Case 1: Peer 0 replies DRAINED while peer 1 disconnects -> neither peer receives RELEASE.
    {
        let (c0, r0) = crate::execution::rank_stream::test_pair().unwrap();
        let (c1, r1) = crate::execution::rank_stream::test_pair().unwrap();
        let mut peers = vec![
            Peer {
                channel: Channel::new(c0).unwrap(),
            },
            Peer {
                channel: Channel::new(c1).unwrap(),
            },
        ];
        let mut rank0 = Channel::new(r0).unwrap();
        let mut rank1 = Channel::new(r1).unwrap();
        let peer0_thread = std::thread::spawn(move || {
            assert!(rank0.receive(DRAIN).unwrap().is_empty());
            rank0.send(STARTED, &[]).unwrap();
            rank0.send(DRAINED, &[]).unwrap();
            // Coordinator must close without sending RELEASE because peer 1 failed to drain.
            let err = rank0.receive(RELEASE).unwrap_err().to_string();
            assert!(err.contains("disconnected"), "{err}");
        });
        let peer1_thread = std::thread::spawn(move || {
            assert!(rank1.receive(DRAIN).unwrap().is_empty());
            drop(rank1);
        });
        let peers_drained = drain_startup_peers(&mut peers);
        assert!(!peers_drained);
        release_and_shutdown_peers(&mut peers, peers_drained);
        peer0_thread.join().unwrap();
        peer1_thread.join().unwrap();
    }

    // Case 2: All peers drain, but local fails to drain -> RELEASE is withheld.
    {
        let (c0, r0) = crate::execution::rank_stream::test_pair().unwrap();
        let mut peers = vec![Peer {
            channel: Channel::new(c0).unwrap(),
        }];
        let mut rank0 = Channel::new(r0).unwrap();
        let peer0_thread = std::thread::spawn(move || {
            assert!(rank0.receive(DRAIN).unwrap().is_empty());
            rank0.send(DRAINED, &[]).unwrap();
            let err = rank0.receive(RELEASE).unwrap_err().to_string();
            assert!(err.contains("disconnected"), "{err}");
        });
        let peers_drained = drain_startup_peers(&mut peers);
        assert!(peers_drained);
        release_and_shutdown_peers(&mut peers, false);
        peer0_thread.join().unwrap();
    }

    // Case 3: Every peer and local drain -> every peer receives RELEASE.
    {
        let (c0, r0) = crate::execution::rank_stream::test_pair().unwrap();
        let (c1, r1) = crate::execution::rank_stream::test_pair().unwrap();
        let mut peers = vec![
            Peer {
                channel: Channel::new(c0).unwrap(),
            },
            Peer {
                channel: Channel::new(c1).unwrap(),
            },
        ];
        let mut rank0 = Channel::new(r0).unwrap();
        let mut rank1 = Channel::new(r1).unwrap();
        let t0 = std::thread::spawn(move || {
            assert!(rank0.receive(DRAIN).unwrap().is_empty());
            rank0.send(DRAINED, &[]).unwrap();
            assert!(rank0.receive(RELEASE).unwrap().is_empty());
        });
        let t1 = std::thread::spawn(move || {
            assert!(rank1.receive(DRAIN).unwrap().is_empty());
            rank1.send(DRAINED, &[]).unwrap();
            assert!(rank1.receive(RELEASE).unwrap().is_empty());
        });
        let peers_drained = drain_startup_peers(&mut peers);
        assert!(peers_drained);
        release_and_shutdown_peers(&mut peers, true);
        t0.join().unwrap();
        t1.join().unwrap();
    }
}

#[test]
fn socket_drain_peers_rejects_nonempty_drained_payload() {
    let (c0, r0) = crate::execution::rank_stream::test_pair().unwrap();
    let mut peers = vec![Peer {
        channel: Channel::new(c0).unwrap(),
    }];
    let mut rank0 = Channel::new(r0).unwrap();
    let t0 = std::thread::spawn(move || {
        assert!(rank0.receive(DRAIN).unwrap().is_empty());
        rank0.send(DRAINED, &[1]).unwrap();
    });
    let err = drain_peers(&mut peers).unwrap_err().to_string();
    assert!(err.contains("unexpected rank drain payload"), "{err}");
    t0.join().unwrap();
}

struct FakePeerExecutor {
    manifest: ExecutionManifest,
    next_fence: u64,
    in_flight: VecDeque<(Fence, usize)>,
    ready_to_complete: VecDeque<FrameCompletion>,
    drain_ok: bool,
    closed: bool,
    auto_complete: bool,
}

impl FakePeerExecutor {
    fn new(manifest: ExecutionManifest, auto_complete: bool) -> Self {
        Self {
            manifest,
            next_fence: 1,
            in_flight: VecDeque::new(),
            ready_to_complete: VecDeque::new(),
            drain_ok: true,
            closed: false,
            auto_complete,
        }
    }
}

impl PeerExecutor for FakePeerExecutor {
    fn manifest(&self) -> &ExecutionManifest {
        &self.manifest
    }
    fn outstanding(&self) -> usize {
        self.in_flight.len()
    }
    fn validate_submission(
        &mut self,
        graph: usize,
        _uploads: &[Upload<'_>],
        _grammar: bool,
    ) -> Result<Option<Fence>> {
        if graph >= self.manifest.graphs.len() {
            return Ok(None);
        }
        Ok(Some(Fence(self.next_fence)))
    }
    fn submit(
        &mut self,
        graph: usize,
        _uploads: &[Upload<'_>],
        _grammar: bool,
    ) -> Result<Option<Fence>> {
        let fence = Fence(self.next_fence);
        self.next_fence += 1;
        self.in_flight.push_back((fence, graph));
        if self.auto_complete {
            let mut done = completed(1);
            done.fence = fence;
            done.graph_index = graph;
            for record in done.records.iter_mut() {
                record.submission_sequence = fence.0 as i64;
            }
            self.ready_to_complete.push_back(done);
        }
        Ok(Some(fence))
    }
    fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        if let Some(done) = self.ready_to_complete.pop_front() {
            self.in_flight.pop_front();
            return Ok(Some(done));
        }
        Ok(None)
    }
    fn park(&self) -> bool {
        false
    }
    fn close(&mut self) -> Shutdown {
        self.closed = true;
        Shutdown {
            drained: self.drain_ok,
            error: (!self.drain_ok).then(|| invalid("synthetic drain failure")),
        }
    }
}

#[test]
fn socket_peer_control_loop_enforces_prepare_commit_drain_release_invariants() {
    let manifest = super::super::tests::manifest();

    // 1. Valid lifecycle: PREPARE -> READY -> COMMIT -> COMPLETE -> DRAIN -> DRAINED -> RELEASE.
    {
        let (coord_s, peer_s) = crate::execution::rank_stream::test_pair().unwrap();
        let mut coord = Channel::new(coord_s).unwrap();
        let mut peer_ch = Channel::new(peer_s).unwrap();
        let mut fake = FakePeerExecutor::new(manifest.clone(), true);
        let peer_thread = std::thread::spawn(move || {
            let result = run_peer_control_loop(&mut peer_ch, &mut fake);
            (result, fake.closed)
        });
        wire::prepare(&manifest, Fence(1), 0, false, &[], |tag, bytes| {
            coord.send(tag, bytes)
        })
        .unwrap();
        let ready = coord.receive(READY).unwrap();
        assert_eq!(read_identity(&ready).unwrap(), (Fence(1), 0));
        coord.send(COMMIT, &identity(Fence(1), 0)).unwrap();
        let comp = coord.receive(COMPLETE).unwrap();
        let mut reader = Reader(&comp);
        assert_eq!(reader.number().unwrap(), 1);
        assert_eq!(reader.number().unwrap(), 0);
        coord.send(DRAIN, &[]).unwrap();
        assert!(coord.receive(DRAINED).unwrap().is_empty());
        coord.send(RELEASE, &[]).unwrap();
        let (result, closed) = peer_thread.join().unwrap();
        result.unwrap();
        assert!(closed);
    }

    // 2. Rejects duplicate PREPARE, mismatched COMMIT, COMMIT without PREPARE, DRAIN while prepared,
    //    DRAIN while executing, unexpected command tag, and missing RELEASE after DRAINED.
    fn assert_peer_loop_fails(
        manifest: &ExecutionManifest,
        auto_complete: bool,
        expected_err: &str,
        drive: impl FnOnce(&mut Channel, &ExecutionManifest),
    ) {
        let (coord_s, peer_s) = crate::execution::rank_stream::test_pair().unwrap();
        let mut coord = Channel::new(coord_s).unwrap();
        let mut peer_ch = Channel::new(peer_s).unwrap();
        let mut fake = FakePeerExecutor::new(manifest.clone(), auto_complete);
        let peer_thread = std::thread::spawn(move || run_peer_control_loop(&mut peer_ch, &mut fake));
        drive(&mut coord, manifest);
        let err = peer_thread.join().unwrap().unwrap_err().to_string();
        assert!(err.contains(expected_err), "expected '{expected_err}', got '{err}'");
    }

    // COMMIT without PREPARE
    assert_peer_loop_fails(&manifest, true, "commit arrived without preparation", |coord, _| {
        coord.send(COMMIT, &identity(Fence(1), 0)).unwrap();
    });

    // Duplicate PREPARE while a frame is already prepared
    assert_peer_loop_fails(&manifest, true, "already has a prepared frame", |coord, m| {
        wire::prepare(m, Fence(1), 0, false, &[], |t, b| coord.send(t, b)).unwrap();
        assert_eq!(read_identity(&coord.receive(READY).unwrap()).unwrap(), (Fence(1), 0));
        wire::prepare(m, Fence(2), 0, false, &[], |t, b| coord.send(t, b)).unwrap();
    });

    // COMMIT with mismatched fence identity
    assert_peer_loop_fails(&manifest, true, "commit identity differs", |coord, m| {
        wire::prepare(m, Fence(1), 0, false, &[], |t, b| coord.send(t, b)).unwrap();
        assert_eq!(read_identity(&coord.receive(READY).unwrap()).unwrap(), (Fence(1), 0));
        coord.send(COMMIT, &identity(Fence(2), 0)).unwrap();
    });

    // DRAIN while a frame is prepared (uncommitted)
    assert_peer_loop_fails(&manifest, true, "drain arrived with outstanding frames", |coord, m| {
        wire::prepare(m, Fence(1), 0, false, &[], |t, b| coord.send(t, b)).unwrap();
        assert_eq!(read_identity(&coord.receive(READY).unwrap()).unwrap(), (Fence(1), 0));
        coord.send(DRAIN, &[]).unwrap();
    });

    // DRAIN while a committed frame is still executing
    assert_peer_loop_fails(&manifest, false, "drain arrived with outstanding frames", |coord, m| {
        wire::prepare(m, Fence(1), 0, false, &[], |t, b| coord.send(t, b)).unwrap();
        assert_eq!(read_identity(&coord.receive(READY).unwrap()).unwrap(), (Fence(1), 0));
        coord.send(COMMIT, &identity(Fence(1), 0)).unwrap();
        coord.send(DRAIN, &[]).unwrap();
    });

    // Unexpected command tag
    assert_peer_loop_fails(&manifest, true, "unexpected rank command", |coord, _| {
        coord.send(READY, &[]).unwrap();
    });

    // Missing RELEASE after DRAINED (coordinator drops socket without releasing)
    assert_peer_loop_fails(&manifest, true, "disconnected", |coord, _| {
        coord.send(DRAIN, &[]).unwrap();
        assert!(coord.receive(DRAINED).unwrap().is_empty());
        let _ = coord.stream.shutdown(std::net::Shutdown::Both);
    });
}
