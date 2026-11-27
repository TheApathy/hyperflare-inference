/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::{pool::RankRecords, record::HEADER_BYTES, testing::geometry};
use std::{path::Path, time::SystemTime};

fn key(tokens: u64) -> PrefixKey {
    PrefixKey {
        consumed_tokens: tokens,
        digest: [tokens as u8; 16],
    }
}

fn write(
    directory: &Path,
    geometry: &[CacheGeometry],
    tokens: u64,
    parent: Option<PrefixKey>,
    values: &[u8],
) -> Header {
    let mut header = Header::new(
        geometry,
        [1; 32],
        [2; 16],
        key(tokens),
        parent,
        [0; 16],
        values.len() as u32,
        SystemTime::now(),
    )
    .unwrap();
    header.row_checksum = record::checksum(&vec![tokens as u8; 4096]);
    let mut bytes = vec![0; header.length_with_row() as usize];
    bytes[..HEADER_BYTES].copy_from_slice(&header.encode().unwrap());
    for (page, &value) in values.iter().enumerate() {
        let at = header.page_segment as usize + page * 4096;
        bytes[at..at + 4096].fill(value);
        let sum = record::checksum(&bytes[at..at + 4096]);
        let at = header.checksum_segment() as usize + page * 8;
        bytes[at..at + 8].copy_from_slice(&sum.to_le_bytes());
    }
    bytes[header.row_segment as usize..].fill(tokens as u8);
    std::fs::write(directory.join(record::file_name(&header.key)), bytes).unwrap();
    header
}

#[test]
fn rank_records_read_slices_rows_and_chain_spills_with_persisted_sums() {
    let directory =
        std::env::temp_dir().join(format!("selene-pool-records-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
    let root = write(&directory, &geometry, 128, None, &[11, 22]);
    let tail = write(&directory, &geometry, 256, Some(root.key), &[33, 44]);
    let lock = crate::testing::directory_lock(&directory).unwrap();
    let mut records = DeferredRecords(Arc::new(Mutex::new(Some(Records {
        directory: directory.clone(),
        identity: [1; 32],
        fingerprint: [2; 16],
        layout: Layout::new(&geometry).unwrap(),
        geometry,

        _lock: lock,
    }))));
    let mut storage = vec![0; 4 * 4096];
    let offset = storage.as_ptr().align_offset(4096);
    let dst = &mut storage[offset..offset + 3 * 4096];
    let sums = records
        .read(root.key, 1, 1, true, &mut dst[..8192])
        .unwrap();
    assert_eq!(&dst[..4096], vec![22; 4096]);
    assert_eq!(&dst[4096..8192], vec![128; 4096]);
    assert_eq!(sums.pages, vec![record::checksum(&vec![22; 4096])]);
    assert_eq!(sums.row, Some(root.row_checksum));
    let sums = records.fill(tail.key, 1, 2, true, dst).unwrap();
    assert_eq!(&dst[..4096], vec![22; 4096]);
    assert_eq!(&dst[4096..8192], vec![33; 4096]);
    assert_eq!(&dst[8192..], vec![0; 4096]);
    assert_eq!(sums.pages.len(), 2);
    assert_eq!(sums.row, Some(tail.row_checksum));
    assert!(
        records
            .read(root.key, 2, 1, false, &mut dst[..4096])
            .is_err()
    );
    assert!(records.read(root.key, 0, 1, false, &mut dst[..12]).is_err());
    // A damaged slice still serves the persisted checksum for the manager to verify.
    let path = directory.join(record::file_name(&root.key));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[root.page_segment as usize] ^= 1;
    std::fs::write(path, bytes).unwrap();
    let sums = records
        .read(root.key, 0, 1, false, &mut dst[..4096])
        .unwrap();
    assert_eq!(sums.pages, vec![record::checksum(&vec![11; 4096])]);
    assert_ne!(sums.pages[0], record::checksum(&dst[..4096]));
    drop(records);
    std::fs::remove_dir_all(directory).unwrap();
}

fn write_packed_v4(
    directory: &Path,
    geometry: &[CacheGeometry],
    tokens: u64,
    parent: Option<PrefixKey>,
    first_page: u64,
    page_packets: &[&[u8]],
    row_packet: Option<&[u8]>,
) -> Header {
    use crate::shared::{
        packed,
        wire::{
            PackedOffer, PackedUnit,
            packed_unit::{Kind, Representation},
        },
    };
    let mut units = Vec::with_capacity(page_packets.len() + usize::from(row_packet.is_some()));
    let mut payload = Vec::new();
    for (i, pkt) in page_packets.iter().enumerate() {
        units.push(PackedUnit {
            kind: Kind::Page as i32,
            logical_index: first_page + i as u64,
            stored_rank: 0,
            representation: Representation::Packet as i32,
            profile: vec![0x5a; 32],
            original_length: 4096,
            stored_length: pkt.len() as u64,
            replicas: Vec::new(),
            stored_crc: packed::crc32(pkt),
            original_crc: 1 + i as u32,
            raw_xxh3: 0,
        });
        payload.extend_from_slice(pkt);
    }
    if let Some(pkt) = row_packet {
        units.push(PackedUnit {
            kind: Kind::Row as i32,
            logical_index: 0,
            stored_rank: 0,
            representation: Representation::Packet as i32,
            profile: vec![0x5a; 32],
            original_length: 4096,
            stored_length: pkt.len() as u64,
            replicas: Vec::new(),
            stored_crc: packed::crc32(pkt),
            original_crc: 99,
            raw_xxh3: 0,
        });
        payload.extend_from_slice(pkt);
    }
    let header = Header::new(
        geometry,
        [1; 32],
        [2; 16],
        key(tokens),
        parent,
        [0; 16],
        page_packets.len() as u32,
        SystemTime::now(),
    )
    .unwrap()
    .with_packed(PackedOffer { first_page, units })
    .unwrap();
    let mut bytes = header.packed_metadata().unwrap();
    bytes.extend_from_slice(&payload);
    std::fs::write(directory.join(record::file_name(&header.key)), bytes).unwrap();
    header
}

#[test]
fn rank_records_read_and_fill_packed_v4_and_literal_fallback() {
    let directory =
        std::env::temp_dir().join(format!("selene-pool-records-v4-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];

    let root = write_packed_v4(
        &directory,
        &geometry,
        128,
        None,
        0,
        &[b"root-p0", b"root-page-1"],
        Some(b"root-row"),
    );
    let child = write_packed_v4(
        &directory,
        &geometry,
        256,
        Some(root.key),
        2,
        &[b"child-p2", b"child-p3-extra"],
        Some(b"child-row-bytes"),
    );
    let raw_fallback = write(&directory, &geometry, 192, None, &[0x41, 0x42, 0x43]);

    let lock = crate::testing::directory_lock(&directory).unwrap();
    let mut records = DeferredRecords(Arc::new(Mutex::new(Some(Records {
        directory: directory.clone(),
        identity: [1; 32],
        fingerprint: [2; 16],
        layout: Layout::new(&geometry).unwrap(),
        geometry,
        _lock: lock,
    }))));

    // Raw read must reject a v4 packed record.
    let mut raw_buf = vec![0u8; 4096];
    assert!(records.read(root.key, 0, 1, false, &mut raw_buf).is_err());

    // Packed single-record slices: page range, page+row, and row-only.
    let mut dst = vec![0u8; 64];
    let n = records.read_packed(root.key, 1, 1, true, &mut dst).unwrap();
    assert_eq!(&dst[..n], b"root-page-1root-row");
    let n = records.read_packed(root.key, 2, 0, true, &mut dst).unwrap();
    assert_eq!(&dst[..n], b"root-row");
    // Out-of-bounds and undersized destination are rejected.
    assert!(
        records
            .read_packed(root.key, 2, 1, false, &mut dst)
            .is_err()
    );
    assert!(records.read_packed(root.key, 0, 1, true, &mut dst).is_err());
    let mut tiny = [0u8; 3];
    assert!(
        records
            .read_packed(root.key, 0, 1, false, &mut tiny)
            .is_err()
    );

    // Chain-spanning packed put fill (pages 1..3 + leaf row).
    let expected = b"root-page-1child-p2child-p3-extrachild-row-bytes";
    let mut spill = vec![0u8; expected.len()];
    let n = records
        .fill_packed(child.key, 1, 3, true, &mut spill)
        .unwrap();
    assert_eq!(n, expected.len());
    assert_eq!(&spill[..], expected);

    // Raw v3 fallback record serves literal bytes through read_packed and fill_packed.
    let mut lit = vec![0u8; 2 * 4096];
    let n = records
        .read_packed(raw_fallback.key, 1, 1, true, &mut lit)
        .unwrap();
    assert_eq!(n, 2 * 4096);
    assert_eq!(&lit[..4096], vec![0x42; 4096]);
    assert_eq!(&lit[4096..], vec![192; 4096]);
    let mut lit_fill = vec![0u8; 2 * 4096];
    let n = records
        .fill_packed(raw_fallback.key, 0, 2, false, &mut lit_fill)
        .unwrap();
    assert_eq!(n, 2 * 4096);
    assert_eq!(&lit_fill[..4096], vec![0x41; 4096]);
    assert_eq!(&lit_fill[4096..], vec![0x42; 4096]);

    drop(records);
    std::fs::remove_dir_all(directory).unwrap();
}
