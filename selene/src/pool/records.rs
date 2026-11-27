/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{Checksums, RankRecords};
use crate::{
    CacheGeometry, Layout, PrefixKey, Result,
    geometry::invalid,
    record::{self, Header},
};
use std::{
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::{Arc, Mutex},
};

/// Filled after cache allocation and scanning, before the peer can serve a pool command.
#[derive(Clone, Default)]
pub struct DeferredRecords(Arc<Mutex<Option<Records>>>);

struct Records {
    directory: PathBuf,
    identity: [u8; 32],
    fingerprint: [u8; 16],
    geometry: Vec<CacheGeometry>,
    layout: Layout,
    _lock: record::DirectoryLock,
}

impl DeferredRecords {
    pub fn bind(
        &self,
        directory: PathBuf,
        identity: [u8; 32],
        fingerprint: [u8; 16],
        geometry: Vec<CacheGeometry>,
        lock: record::DirectoryLock,
    ) -> Result<()> {
        *self.0.lock().unwrap() = Some(Records {
            directory,
            identity,
            fingerprint,
            layout: Layout::new(&geometry)?,
            geometry,
            _lock: lock,
        });
        Ok(())
    }
}

impl Records {
    fn header(&self, key: PrefixKey) -> Result<(std::fs::File, Header)> {
        let path = self.directory.join(record::file_name(&key));
        let mut file =
            std::fs::File::open(path).map_err(|e| record::io_failure("open a rank record", e))?;
        let metadata = Header::read_metadata(&mut file)
            .map_err(|e| record::io_failure("read a rank record header", e))?;
        let header = Header::decode(&metadata)?;
        if header.key != key
            || header.identity != self.identity
            || header.fingerprint != self.fingerprint
            || !header.matches(&self.geometry)
        {
            return Err(invalid(
                "a pool rank record has a different identity or layout",
            ));
        }
        Ok((file, header))
    }

    fn chain(&self, key: PrefixKey) -> Result<Vec<Header>> {
        let mut chain = Vec::new();
        let mut cursor = key;
        loop {
            let (_, header) = self.header(cursor)?;
            let parent = header.parent;
            chain.push(header);
            let Some(parent) = parent else { break };
            if parent.consumed_tokens >= cursor.consumed_tokens {
                return Err(invalid("a pool rank record chain does not decrease"));
            }
            cursor = parent;
        }
        chain.reverse();
        Ok(chain)
    }

    fn read(
        &self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        let (mut file, header) = self.header(key)?;
        if header.packed.is_some() {
            return Err(invalid("cannot raw-read a v4 packed pool rank record"));
        }
        let page_bytes = self.layout.page_record_bytes as usize;
        let length = pages as usize * page_bytes
            + if row {
                self.layout.row_record_bytes as usize
            } else {
                0
            };
        if first
            .checked_add(pages)
            .is_none_or(|end| end > header.tail_pages)
            || dst.len() != length
        {
            return Err(invalid(
                "a pool rank record slice exceeds its record or destination",
            ));
        }
        let mut stored = vec![0; pages as usize * 8];
        file.seek(SeekFrom::Start(
            header.checksum_segment() + u64::from(first) * 8,
        ))
        .and_then(|_| file.read_exact(&mut stored))
        .map_err(|e| record::io_failure("read rank record checksums", e))?;
        let sums = Checksums {
            pages: stored
                .chunks_exact(8)
                .map(|sum| u64::from_le_bytes(sum.try_into().unwrap()))
                .collect(),
            row: row.then_some(header.row_checksum),
        };
        let page_length = pages as usize * page_bytes;
        file.seek(SeekFrom::Start(
            header.page_segment + u64::from(first) * page_bytes as u64,
        ))
        .and_then(|_| file.read_exact(&mut dst[..page_length]))
        .map_err(|e| record::io_failure("read rank record pages", e))?;
        if row {
            file.seek(SeekFrom::Start(header.row_segment))
                .and_then(|_| file.read_exact(&mut dst[page_length..]))
                .map_err(|e| record::io_failure("read a rank record row", e))?;
        }
        Ok(sums)
    }

    fn read_packed(
        &self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<usize> {
        let (mut file, header) = self.header(key)?;
        if let Some(layout) = &header.packed {
            let first = first as usize;
            let count = pages as usize;
            let page_count = layout.page_count();
            if first.checked_add(count).is_none_or(|end| end > page_count)
                || (row && (first + count != page_count || layout.row_bytes() == 0))
            {
                return Err(invalid(
                    "a packed pool rank record slice exceeds its v4 directory",
                ));
            }
            let start_rel = if count == 0 && row {
                layout.page_offset(page_count)
            } else {
                layout.page_offset(first)
            };
            let end_rel = if row {
                layout.physical_bytes()
            } else {
                layout.page_offset(first + count)
            };
            let len = end_rel
                .checked_sub(start_rel)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| invalid("packed pool rank record slice length overflow"))?;
            let out = dst.get_mut(..len).ok_or_else(|| {
                invalid("a packed pool rank record slice exceeds its destination")
            })?;
            if len > 0 {
                file.seek(SeekFrom::Start(header.page_segment + start_rel))
                    .and_then(|_| file.read_exact(out))
                    .map_err(|e| record::io_failure("read packed rank record slice", e))?;
            }
            return Ok(len);
        }
        // Raw v3 fallback when this rank fell back to all-literal storage while
        // another rank of the same world-wide v4 record compressed as packets.
        let page_bytes = self.layout.page_record_bytes as usize;
        let row_bytes = self.layout.row_record_bytes as usize;
        let page_length = pages as usize * page_bytes;
        let length = page_length + if row { row_bytes } else { 0 };
        if first
            .checked_add(pages)
            .is_none_or(|end| end > header.tail_pages)
        {
            return Err(invalid("a pool rank record slice exceeds its record"));
        }
        let out = dst
            .get_mut(..length)
            .ok_or_else(|| invalid("a packed pool rank record slice exceeds its destination"))?;
        if page_length > 0 {
            file.seek(SeekFrom::Start(
                header.page_segment + u64::from(first) * page_bytes as u64,
            ))
            .and_then(|_| file.read_exact(&mut out[..page_length]))
            .map_err(|e| record::io_failure("read literal rank record pages", e))?;
        }
        if row {
            file.seek(SeekFrom::Start(header.row_segment))
                .and_then(|_| file.read_exact(&mut out[page_length..]))
                .map_err(|e| record::io_failure("read literal rank record row", e))?;
        }
        Ok(length)
    }

    fn fill(
        &self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        let chain = self.chain(key)?;
        let end = u64::from(first) + u64::from(pages);
        let (mut base, mut offset) = (0u64, 0usize);
        let mut sums = Checksums {
            pages: Vec::new(),
            row: None,
        };
        for header in &chain {
            let from = u64::from(first).max(base);
            let to = end.min(base + u64::from(header.tail_pages));
            let count = to.saturating_sub(from) as u32;
            let with_row = row && header.key == key;
            if count > 0 || with_row {
                let local_first = if count > 0 {
                    (from - base) as u32
                } else {
                    header.tail_pages
                };
                let bytes = count as usize * self.layout.page_record_bytes as usize
                    + if with_row {
                        self.layout.row_record_bytes as usize
                    } else {
                        0
                    };
                let chunk = dst
                    .get_mut(offset..offset + bytes)
                    .ok_or_else(|| invalid("a pool rank spill destination is too small"))?;
                let read = self.read(header.key, local_first, count, with_row, chunk)?;
                sums.pages.extend(read.pages);
                if with_row {
                    sums.row = read.row;
                }
                offset += bytes;
            }
            base += u64::from(header.tail_pages);
        }
        if end > base || offset != dst.len() {
            return Err(invalid(
                "a pool rank spill exceeds its chain or destination",
            ));
        }
        Ok(sums)
    }

    fn fill_packed(
        &self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<usize> {
        let chain = self.chain(key)?;
        let end = u64::from(first) + u64::from(pages);
        let (mut base, mut offset) = (0u64, 0usize);
        for header in &chain {
            let from = u64::from(first).max(base);
            let to = end.min(base + u64::from(header.tail_pages));
            let count = to.saturating_sub(from) as u32;
            let with_row = row && header.key == key;
            if count > 0 || with_row {
                let local_first = if count > 0 {
                    (from - base) as u32
                } else {
                    header.tail_pages
                };
                let written =
                    self.read_packed(header.key, local_first, count, with_row, &mut dst[offset..])?;
                offset += written;
            }
            base += u64::from(header.tail_pages);
        }
        if end > base || offset != dst.len() {
            return Err(invalid(
                "a packed pool rank spill exceeds its chain or destination",
            ));
        }
        Ok(offset)
    }
}

impl RankRecords for DeferredRecords {
    fn read(
        &mut self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        // Rank zero holds every shard until the agent's read drains.
        self.0
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| invalid("pool rank records are not initialized"))?
            .read(key, first, pages, row, dst)
    }

    fn fill(
        &mut self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<Checksums> {
        self.0
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| invalid("pool rank records are not initialized"))?
            .fill(key, first, pages, row, dst)
    }

    fn read_packed(
        &mut self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<usize> {
        self.0
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| invalid("pool rank records are not initialized"))?
            .read_packed(key, first, pages, row, dst)
    }

    fn fill_packed(
        &mut self,
        key: PrefixKey,
        first: u32,
        pages: u32,
        row: bool,
        dst: &mut [u8],
    ) -> Result<usize> {
        self.0
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| invalid("pool rank records are not initialized"))?
            .fill_packed(key, first, pages, row, dst)
    }
}

#[cfg(test)]
mod tests;
