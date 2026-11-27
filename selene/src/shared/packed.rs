/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Physical layout of v4 records. The packet payload is opaque to Selene.

use super::{
    record::RankSizes,
    wire::packed_unit::{Kind, Representation},
    wire::{CodecCaps, PackedOffer, PackedUnit, RankPlanes},
};
use crate::{Result, geometry::invalid};
use prost::Message;

/// Header version byte for v4 packed records (raw records use `record::VERSION = 3`).
pub const REVISION: u32 = 4;
pub const MAX_DIRECTORY: usize = 2 * 1024 * 1024;
pub const MAX_UNITS: usize = 16384;
pub const MAX_RANKS: usize = 64;

/// Whether a record tail with `pages`, optional `row`, and `ranks` fits within `MAX_UNITS`.
pub(crate) fn fits_units(pages: usize, row: bool, ranks: usize) -> bool {
    pages.saturating_add(usize::from(row)).saturating_mul(ranks) <= MAX_UNITS
}

pub fn raw_reader() -> CodecCaps {
    CodecCaps {
        envelope_revision: REVISION,
        receive_profiles: Vec::new(),
    }
}

pub fn canonical_caps(caps: Option<&CodecCaps>) -> Result<Option<CodecCaps>> {
    let Some(caps) = caps else { return Ok(None) };
    if caps.receive_profiles.len() > 64 || caps.receive_profiles.iter().any(|p| p.len() != 32) {
        return Err(invalid("invalid codec profile set"));
    }
    let mut profiles = caps.receive_profiles.clone();
    profiles.sort_unstable();
    profiles.dedup();
    Ok(Some(CodecCaps {
        envelope_revision: caps.envelope_revision,
        receive_profiles: profiles,
    }))
}

pub fn permits<'a>(
    caps: Option<&CodecCaps>,
    profiles: impl IntoIterator<Item = &'a Vec<u8>>,
) -> bool {
    caps.is_some_and(|c| {
        c.envelope_revision == REVISION
            && !c.receive_profiles.is_empty()
            && profiles.into_iter().all(|p| c.receive_profiles.contains(p))
    })
}

pub fn permits_offer(caps: Option<&CodecCaps>, offer: &PackedOffer) -> bool {
    permits(
        caps,
        offer
            .units
            .iter()
            .filter(|u| u.representation != Representation::Literal as i32)
            .map(|u| &u.profile),
    )
}

pub fn permits_holding(caps: Option<&CodecCaps>, packed: bool, profiles: &[Vec<u8>]) -> bool {
    (!packed && profiles.is_empty()) || permits(caps, profiles)
}

pub fn crc32(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

pub fn crc32_chunks(chunks: &[&[u8]]) -> u32 {
    let mut hash = crc32fast::Hasher::new();
    for chunk in chunks {
        hash.update(chunk);
    }
    hash.finalize()
}

/// Directory on disk, protected as one length-delimited blob.
#[derive(Clone, PartialEq, Message)]
pub(crate) struct Directory {
    #[prost(message, repeated, tag = "1")]
    pub ranks: Vec<RankPlanes>,
    #[prost(message, optional, tag = "2")]
    pub offer: Option<PackedOffer>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub(crate) offer: PackedOffer,
    // Offset of each unit, and one end sentinel. Relative to the payload.
    offsets: Vec<u64>,
    // Index of the first unit of each page, then the first row unit.
    pages: Vec<usize>,
}

// PackedOffer holds only integer, byte, and nested integer fields (no floats).
impl Eq for Layout {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub first: u32,
    pub pages: u32,
    pub row: bool,
    pub offset: u64,
    pub length: u64,
}

impl Layout {
    /// Windows contain whole pages. The row is one independent grant, even above the target size.
    pub fn windows(&self, skip: u32, row: bool, max: u64) -> Vec<Span> {
        let mut windows = Vec::new();
        let mut i = skip as usize;
        while i < self.page_count() {
            let mut end = i + 1;
            let offset = self.page_offset(i);
            while end < self.page_count()
                && end - i < 64
                && self.page_offset(end + 1) - offset <= max
            {
                end += 1
            }
            windows.push(Span {
                first: i as u32,
                pages: (end - i) as u32,
                row: false,
                offset,
                length: self.page_offset(end) - offset,
            });
            i = end;
        }
        if row && self.row_bytes() != 0 {
            windows.push(Span {
                first: self.page_count() as u32,
                pages: 0,
                row: true,
                offset: self.page_offset(self.page_count()),
                length: self.row_bytes(),
            });
        }
        windows
    }

    pub fn new(offer: PackedOffer, ranks: &[RankSizes], first: u64, tail: u32) -> Result<Self> {
        if ranks.is_empty()
            || ranks.len() > MAX_RANKS
            || offer.first_page != first
            || offer.units.is_empty()
            || offer.units.len() > MAX_UNITS
            || tail as usize > offer.units.len()
            || offer.encoded_len() > MAX_DIRECTORY
        {
            return Err(invalid(
                "packed directory exceeds bounds or has the wrong first page",
            ));
        }
        // Geometry from disk is untrusted. Check the padding before using the raw size accessors.
        use crate::geometry::padded_bytes;
        for rank in ranks {
            if rank.planes.is_empty()
                || rank.planes.contains(&0)
                || rank.row_bytes == 0
                || padded_bytes(rank.row_bytes).is_none()
                || rank
                    .planes
                    .iter()
                    .try_fold(0u64, |n, &p| n.checked_add(padded_bytes(p)?))
                    .is_none()
            {
                return Err(invalid("packed geometry exceeds address space"));
            }
        }
        let end = first
            .checked_add(tail.into())
            .ok_or_else(|| invalid("packed page overflow"))?;
        let mut offsets = vec![0u64];
        let mut pages = Vec::new();
        let mut i = 0;
        for (kind, index) in (first..end)
            .map(|p| (Kind::Page, p))
            .chain(std::iter::once((Kind::Row, 0)))
        {
            pages.push(i);
            if kind == Kind::Row && i == offer.units.len() {
                break;
            }
            let mut seen = 0u64;
            let mut previous = None;
            while let Some(unit) = offer
                .units
                .get(i)
                .filter(|u| u.kind == kind as i32 && u.logical_index == index)
            {
                if previous.is_some_and(|p| p >= unit.stored_rank)
                    || unit.stored_length == 0
                    || unit.stored_length > unit.original_length
                    || unit.replicas.len() >= ranks.len()
                {
                    return Err(invalid("invalid packed unit order or size"));
                }
                previous = Some(unit.stored_rank);
                match Representation::try_from(unit.representation) {
                    Ok(Representation::Literal)
                        if unit.profile.is_empty()
                            && unit.original_length == unit.stored_length
                            && unit.stored_crc == 0
                            && unit.original_crc == 0 => {}
                    Ok(Representation::Packet)
                        if unit.profile.len() == 32 && unit.raw_xxh3 == 0 => {}
                    _ => return Err(invalid("unknown unit representation or checksum domain")),
                }
                let mut last = None;
                for rank in std::iter::once(&unit.stored_rank)
                    .chain(&unit.replicas)
                    .copied()
                {
                    let geometry = ranks
                        .get(rank as usize)
                        .ok_or_else(|| invalid("packed unit rank"))?;
                    let length = if kind == Kind::Page {
                        geometry.page_record_bytes()
                    } else {
                        geometry.row_record_bytes()
                    };
                    if last.is_some_and(|r| r >= rank)
                        || seen & (1u64 << rank) != 0
                        || length != unit.original_length
                    {
                        return Err(invalid("packed unit geometry or replica"));
                    }
                    last = Some(rank);
                    seen |= 1u64 << rank;
                }
                let next = offsets
                    .last()
                    .unwrap()
                    .checked_add(unit.stored_length)
                    .ok_or_else(|| invalid("packed length overflow"))?;
                offsets.push(next);
                i += 1;
            }
            if seen.count_ones() as usize != ranks.len() {
                return Err(invalid("incomplete packed rank coverage"));
            }
        }
        if i != offer.units.len() {
            return Err(invalid("unexpected trailing packed units"));
        }
        Ok(Self {
            offer,
            offsets,
            pages,
        })
    }

    pub fn offer(&self) -> &PackedOffer {
        &self.offer
    }
    pub fn physical_bytes(&self) -> u64 {
        *self.offsets.last().unwrap()
    }
    pub fn page_offset(&self, local: usize) -> u64 {
        self.offsets[self.pages[local]]
    }
    pub fn page_count(&self) -> usize {
        self.pages.len() - 1
    }
    pub fn row_bytes(&self) -> u64 {
        self.physical_bytes() - self.page_offset(self.page_count())
    }
    pub fn units(&self, first: usize, pages: usize, row: bool) -> Option<&[PackedUnit]> {
        let tail = self.page_count();
        let first = if pages == 0 && row { tail } else { first };
        let end = first.checked_add(pages)?;
        if end > tail || (row && end != tail) {
            return None;
        }
        let stop = if row {
            self.offer.units.len()
        } else {
            self.pages[end]
        };
        self.offer.units.get(self.pages[first]..stop)
    }
    pub fn required_profiles(&self, row: bool) -> Vec<Vec<u8>> {
        let mut profiles: Vec<_> = self
            .units(0, self.page_count(), row)
            .unwrap()
            .iter()
            .filter(|u| u.representation == Representation::Packet as i32)
            .map(|u| u.profile.clone())
            .collect();
        profiles.sort_unstable();
        profiles.dedup();
        profiles
    }
}

/// Authenticate the presented grant against the pinned selection before consuming it.
#[cfg(any(test, feature = "test-support"))]
pub fn verify_window(
    expected: &[PackedUnit],
    manifest: &super::wire::PackedWindow,
    bytes: &[u8],
) -> Result<()> {
    if expected != manifest.units {
        return Err(invalid("packed window differs from the pinned selection"));
    }
    verify_units(expected, bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checksum {
    Stored,
    Original,
}

impl std::fmt::Display for Checksum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stored => "stored codec checksum mismatch",
            Self::Original => "original codec checksum mismatch",
        })
    }
}

impl std::error::Error for Checksum {}

// Check before handing the bytes to a decoder, including on shared-memory reads.
pub fn verify_units(units: &[PackedUnit], bytes: &[u8]) -> Result<()> {
    let mut rest = bytes;
    for unit in units {
        let n = usize::try_from(unit.stored_length).map_err(|_| invalid("packed unit length"))?;
        let (data, tail) = rest
            .split_at_checked(n)
            .ok_or_else(|| invalid("short packed window"))?;
        let valid = match Representation::try_from(unit.representation) {
            Ok(Representation::Literal) => super::record::checksum(data) == unit.raw_xxh3,
            Ok(Representation::Packet) => crc32(data) == unit.stored_crc,
            _ => false,
        };
        if !valid {
            return Err(crate::Error::carrying(Checksum::Stored));
        }
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(invalid("extra packed window bytes"));
    }
    Ok(())
}

/// Select the rank's units while retaining their offsets in the shared extent
/// (including replicated units for read/restore paths).
pub fn leg_units(
    window: &super::wire::PackedWindow,
    rank: Option<u32>,
) -> impl Iterator<Item = (u64, &PackedUnit)> {
    window
        .units
        .iter()
        .scan(0u64, |offset, u| {
            let at = *offset;
            *offset = at.checked_add(u.stored_length)?;
            Some((at, u))
        })
        .filter(move |(_, u)| rank.is_none_or(|r| r == u.stored_rank || u.replicas.contains(&r)))
}

/// Select only units whose canonical `stored_rank` equals `rank` (for exclusive
/// write/put paths so replicated units are never written concurrently by two ranks).
pub fn stored_rank_units(
    window: &super::wire::PackedWindow,
    rank: u32,
) -> impl Iterator<Item = (u64, &PackedUnit)> {
    window
        .units
        .iter()
        .scan(0u64, |offset, u| {
            let at = *offset;
            *offset = at.checked_add(u.stored_length)?;
            Some((at, u))
        })
        .filter(move |(_, u)| u.stored_rank == rank)
}

/// Stored bytes belonging to `rank` inside a world-wide packed window (read/restore view).
pub fn rank_window_bytes(window: &super::wire::PackedWindow, rank: u32) -> u64 {
    leg_units(window, Some(rank))
        .map(|(_, u)| u.stored_length)
        .sum()
}

/// Stored bytes canonically owned by `rank` (`stored_rank == rank`) for exclusive writes.
pub fn stored_rank_window_bytes(window: &super::wire::PackedWindow, rank: u32) -> u64 {
    stored_rank_units(window, rank)
        .map(|(_, u)| u.stored_length)
        .sum()
}

/// Scatter `rank`'s contiguous packed slice `src` into its unit offsets inside
/// a shared world-wide packed window `dst`.
pub fn scatter_rank_units(
    window: &super::wire::PackedWindow,
    rank: u32,
    dst: &mut [u8],
    src: &[u8],
) -> Result<()> {
    let mut cursor = 0usize;
    for (at, unit) in stored_rank_units(window, rank) {
        let len = usize::try_from(unit.stored_length).map_err(|_| invalid("packed unit length"))?;
        let at = usize::try_from(at).map_err(|_| invalid("packed unit offset"))?;
        let cursor_end = cursor
            .checked_add(len)
            .ok_or_else(|| invalid("packed rank slice offset overflow"))?;
        let at_end = at
            .checked_add(len)
            .ok_or_else(|| invalid("packed rank unit offset overflow"))?;
        let from = src
            .get(cursor..cursor_end)
            .ok_or_else(|| invalid("packed rank slice is shorter than its manifest"))?;
        let to = dst
            .get_mut(at..at_end)
            .ok_or_else(|| invalid("packed rank unit lies outside the shared window"))?;
        to.copy_from_slice(from);
        cursor = cursor_end;
    }
    if cursor != src.len() {
        return Err(invalid("packed rank slice has trailing bytes"));
    }
    Ok(())
}

/// Build a single-rank all-literal `PackedOffer` from a raw v3 header and its
/// persisted page checksums so mixed compressed/literal multi-host ranks can
/// form a single world-wide v4 offer.
pub fn literal_offer_from_raw(
    header: &super::record::Header,
    page_sums: &[u64],
    row: bool,
) -> Result<PackedOffer> {
    if header.packed.is_some() || header.ranks.len() != 1 {
        return Err(invalid("expected a single-rank raw header"));
    }
    if page_sums.len() != header.tail_pages as usize {
        return Err(invalid(
            "raw header checksum count disagrees with tail pages",
        ));
    }
    let first_page = header
        .parent
        .map_or(0, |p| p.consumed_tokens / header.page_tokens);
    let rank = &header.ranks[0];
    let page_bytes = rank.page_record_bytes();
    let row_bytes = rank.row_record_bytes();
    let mut units = Vec::with_capacity(page_sums.len() + usize::from(row));
    for (i, &sum) in page_sums.iter().enumerate() {
        units.push(PackedUnit {
            kind: Kind::Page as i32,
            logical_index: first_page + i as u64,
            stored_rank: 0,
            representation: Representation::Literal as i32,
            profile: Vec::new(),
            original_length: page_bytes,
            stored_length: page_bytes,
            replicas: Vec::new(),
            stored_crc: 0,
            original_crc: 0,
            raw_xxh3: sum,
        });
    }
    if row {
        units.push(PackedUnit {
            kind: Kind::Row as i32,
            logical_index: 0,
            stored_rank: 0,
            representation: Representation::Literal as i32,
            profile: Vec::new(),
            original_length: row_bytes,
            stored_length: row_bytes,
            replicas: Vec::new(),
            stored_crc: 0,
            original_crc: 0,
            raw_xxh3: header.row_checksum,
        });
    }
    Ok(PackedOffer { first_page, units })
}

/// Merge `world` single-rank offers (each describing its own rank as `stored_rank == 0`)
/// into a canonical world-wide [`Layout`] covering ranks `0..world`.
pub fn merge_rank_offers(
    offers: &[PackedOffer],
    ranks: &[RankSizes],
    first_page: u64,
    tail_pages: u32,
    row: bool,
) -> Result<Layout> {
    let world = ranks.len();
    if world == 0 || offers.len() != world {
        return Err(invalid("packed rank offer count disagrees with world size"));
    }
    let end_page = first_page
        .checked_add(u64::from(tail_pages))
        .ok_or_else(|| invalid("packed page range overflow"))?;
    let mut page_units: Vec<Vec<Option<&PackedUnit>>> =
        vec![vec![None; world]; tail_pages as usize];
    let mut row_units: Vec<Option<&PackedUnit>> = vec![None; world];
    for (rank, offer) in offers.iter().enumerate() {
        if offer.first_page != first_page {
            return Err(invalid("rank packed offer has a different first page"));
        }
        for unit in &offer.units {
            if unit.stored_rank != 0 || !unit.replicas.is_empty() {
                return Err(invalid(
                    "single-rank packed offer must store rank zero without replicas",
                ));
            }
            match Kind::try_from(unit.kind) {
                Ok(Kind::Page) => {
                    if unit.logical_index < first_page || unit.logical_index >= end_page {
                        return Err(invalid("rank packed offer page outside record range"));
                    }
                    let slot = &mut page_units[(unit.logical_index - first_page) as usize][rank];
                    if slot.replace(unit).is_some() {
                        return Err(invalid("duplicate page unit in rank packed offer"));
                    }
                }
                Ok(Kind::Row) => {
                    if unit.logical_index != 0 {
                        return Err(invalid("invalid row logical index in rank packed offer"));
                    }
                    if row && row_units[rank].replace(unit).is_some() {
                        return Err(invalid("duplicate row unit in rank packed offer"));
                    }
                }
                Err(_) => return Err(invalid("unknown packed unit kind")),
            }
        }
    }
    let mut units = Vec::with_capacity(
        (tail_pages as usize)
            .saturating_add(usize::from(row))
            .saturating_mul(world),
    );
    for page in page_units {
        for (rank, unit) in page.into_iter().enumerate() {
            let mut unit = unit
                .ok_or_else(|| invalid("rank packed offer missing a page unit"))?
                .clone();
            unit.stored_rank = rank as u32;
            units.push(unit);
        }
    }
    if row {
        for (rank, unit) in row_units.into_iter().enumerate() {
            let mut unit = unit
                .ok_or_else(|| invalid("rank packed offer missing a row unit"))?
                .clone();
            unit.stored_rank = rank as u32;
            units.push(unit);
        }
    }
    Layout::new(
        PackedOffer { first_page, units },
        ranks,
        first_page,
        tail_pages,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::wire::PackedWindow;

    fn unit(rank: u32, data: &[u8], replicas: Vec<u32>) -> PackedUnit {
        PackedUnit {
            stored_rank: rank,
            replicas,
            representation: 1,
            profile: vec![4; 32],
            original_length: 4096,
            stored_length: data.len() as u64,
            stored_crc: crc32(data),
            original_crc: 7,
            ..Default::default()
        }
    }

    #[test]
    fn replica_legs_read_the_owner_bytes() {
        let a = unit(0, b"aaa", vec![2]);
        let b = unit(1, b"bb", vec![]);
        let window = PackedWindow { units: vec![a, b] };
        let ranks = vec![
            RankSizes {
                planes: vec![4096],
                row_bytes: 1
            };
            3
        ];
        let offer = PackedOffer {
            first_page: 0,
            units: window.units.clone(),
        };
        Layout::new(offer, &ranks, 0, 1).unwrap();
        for (rank, expected) in [
            (None, vec![(0, 3), (3, 2)]),
            (Some(1), vec![(3, 2)]),
            (Some(2), vec![(0, 3)]),
            (Some(0), vec![(0, 3)]),
        ] {
            let spans: Vec<_> = leg_units(&window, rank)
                .map(|(at, u)| (at, u.stored_length))
                .collect();
            assert_eq!(spans, expected);
        }
    }

    #[test]
    fn packed_receive_checks_selection_length_and_stored_crc() {
        let expected = vec![unit(0, b"abc", vec![])];
        let window = PackedWindow {
            units: expected.clone(),
        };
        let mut mutated = window.clone();
        mutated.units[0].original_crc ^= 1;
        assert!(verify_window(&expected, &mutated, b"abc").is_err());
        assert!(verify_window(&expected, &window, b"abd").is_err());
        assert!(verify_window(&expected, &window, b"ab").is_err());
        assert!(verify_window(&expected, &window, b"abcd").is_err());
        verify_window(&expected, &window, b"abc").unwrap();
    }

    #[test]
    fn a_raw_only_receiver_or_missing_ack_cannot_receive_a_packet() {
        let literal = PackedOffer {
            units: vec![PackedUnit::default()],
            ..Default::default()
        };
        assert!(!permits_offer(Some(&raw_reader()), &literal));
        let need = vec![vec![9; 32]];
        assert!(!permits(None, &need));
        assert!(!permits(Some(&raw_reader()), &need));
        let mut caps = CodecCaps {
            envelope_revision: 4,
            receive_profiles: vec![vec![9; 32], vec![5; 32], vec![9; 32]],
        };
        let negotiated_caps = canonical_caps(Some(&caps)).unwrap().unwrap();
        assert_eq!(
            negotiated_caps.receive_profiles,
            vec![vec![5; 32], vec![9; 32]]
        );
        assert!(permits(Some(&negotiated_caps), &need));
        assert!(permits_offer(Some(&negotiated_caps), &literal));
        caps.receive_profiles[1].pop();
        assert!(canonical_caps(Some(&caps)).is_err());
    }

    #[test]
    fn fits_units_enforces_max_units_boundary() {
        // TP4: 4,095 pages + 1 row = 4,096 * 4 = 16,384 (fits); 4,096 pages + 1 row = 16,388 (exceeds).
        assert!(fits_units(4095, true, 4));
        assert!(!fits_units(4096, true, 4));
        assert!(fits_units(4096, false, 4));
        assert!(!fits_units(4097, false, 4));
        assert!(!fits_units(usize::MAX, true, 4));
    }

    #[test]
    fn merge_rank_offers_combines_compressed_and_literal_ranks_and_scatters_windows() {
        let rank_geom = crate::testing::geometry(64, &[("kv", 4096)], 4096);
        let mut raw_header = crate::record::Header::new(
            std::slice::from_ref(&rank_geom),
            [1; 32],
            [2; 16],
            crate::PrefixKey {
                consumed_tokens: 128,
                digest: [9; 16],
            },
            None,
            [0; 16],
            2,
            std::time::SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        let r1_p0 = vec![0x11u8; 4096];
        let r1_p1 = vec![0x22u8; 4096];
        let r1_row = vec![0x33u8; 4096];
        raw_header.row_checksum = crate::record::checksum(&r1_row);
        let r1_offer = literal_offer_from_raw(
            &raw_header,
            &[
                crate::record::checksum(&r1_p0),
                crate::record::checksum(&r1_p1),
            ],
            true,
        )
        .unwrap();

        let r0_p0 = b"r0-page0-pkt";
        let r0_p1 = b"r0-p1";
        let r0_row = b"r0-row-pkt";
        let mut u_p0 = unit(0, r0_p0, vec![]);
        u_p0.logical_index = 0;
        let mut u_p1 = unit(0, r0_p1, vec![]);
        u_p1.logical_index = 1;
        let mut u_row = unit(0, r0_row, vec![]);
        u_row.kind = Kind::Row as i32;
        u_row.logical_index = 0;
        let r0_offer = PackedOffer {
            first_page: 0,
            units: vec![u_p0, u_p1, u_row],
        };

        let world_ranks = vec![raw_header.ranks[0].clone(); 2];
        let merged = merge_rank_offers(
            &[r0_offer.clone(), r1_offer.clone()],
            &world_ranks,
            0,
            2,
            true,
        )
        .unwrap();
        assert_eq!(merged.page_count(), 2);
        assert_eq!(merged.row_bytes(), r0_row.len() as u64 + 4096);
        let window = PackedWindow {
            units: merged.units(0, 2, true).unwrap().to_vec(),
        };
        assert_eq!(
            rank_window_bytes(&window, 0),
            (r0_p0.len() + r0_p1.len() + r0_row.len()) as u64
        );
        assert_eq!(rank_window_bytes(&window, 1), 3 * 4096);

        let mut shared = vec![0u8; merged.physical_bytes() as usize];
        let r0_slice = [r0_p0.as_slice(), r0_p1.as_slice(), r0_row.as_slice()].concat();
        let r1_slice = [r1_p0.as_slice(), r1_p1.as_slice(), r1_row.as_slice()].concat();
        scatter_rank_units(&window, 0, &mut shared, &r0_slice).unwrap();
        scatter_rank_units(&window, 1, &mut shared, &r1_slice).unwrap();
        verify_units(&window.units, &shared).unwrap();

        // Merging with row=false drops row units cleanly.
        let no_row = merge_rank_offers(
            &[r0_offer.clone(), r1_offer.clone()],
            &world_ranks,
            0,
            2,
            false,
        )
        .unwrap();
        assert_eq!(no_row.row_bytes(), 0);

        // Nonzero stored_rank in a single-rank offer is rejected.
        let mut bad = r0_offer;
        bad.units[0].stored_rank = 1;
        assert!(merge_rank_offers(&[bad, r1_offer], &world_ranks, 0, 2, true).is_err());
    }
}
