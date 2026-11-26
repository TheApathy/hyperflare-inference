/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::{
    Result, invalid,
    record::{self, Descriptor},
};
use flare_eos::direct_update_file as open;
use flare_selene::shared::record::Header;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const POOL_MARKER: &str = ".selene-pool";
const POOL_MARKER_TMP: &str = ".selene-pool.tmp";

pub use flare_selene::shared::record::DirectoryLock as RootLock;

pub struct DiskExtent {
    pub path: PathBuf,
    retired: AtomicBool,
}
impl DiskExtent {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            retired: AtomicBool::new(false),
        }
    }
    pub fn reader(&self) -> Result<Arc<File>> {
        open(&self.path, false).map(Arc::new)
    }
    pub fn keep(&self) {
        self.retired.store(false, Ordering::Release);
    }
    pub fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }
}
impl Drop for DiskExtent {
    fn drop(&mut self) {
        if self.retired.load(Ordering::Acquire) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
pub use flare_selene::shared::disk::{Faults, IoGate};
pub struct Recovered {
    pub descriptor: Descriptor,
    pub records: Vec<(Header, bool, Arc<DiskExtent>)>,
}

fn bind_or_verify_pool(root: &Path, pool_name: &str) -> Result<()> {
    if pool_name.is_empty() || pool_name.bytes().any(|b| matches!(b, b'\0' | b'\n' | b'\r')) {
        return Err(invalid("invalid pool name for durable record root"));
    }
    let marker = root.join(POOL_MARKER);
    match fs::symlink_metadata(&marker) {
        Ok(meta) => {
            if !meta.file_type().is_file() {
                return Err(invalid(format!(
                    "pool ownership marker {} is not a regular file",
                    marker.display()
                )));
            }
            let content = fs::read_to_string(&marker)?;
            let recorded = content.strip_suffix('\n').unwrap_or(&content);
            if recorded.is_empty() || recorded.bytes().any(|b| matches!(b, b'\0' | b'\n' | b'\r'))
            {
                return Err(invalid(format!(
                    "corrupt pool ownership marker {}",
                    marker.display()
                )));
            }
            if recorded != pool_name {
                return Err(invalid(format!(
                    "durable root {} belongs to pool {recorded:?}, not {pool_name:?}",
                    root.display()
                )));
            }
            Ok(())
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            let tmp = root.join(POOL_MARKER_TMP);
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            writeln!(file, "{pool_name}")?;
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp, &marker)?;
            if let Ok(dir) = File::open(root) {
                let _ = dir.sync_all();
            }
            Ok(())
        }
        Err(err) => Err(err),
    }
}

pub fn recover(root: &Path, pool_name: &str) -> Result<Vec<Recovered>> {
    use flare_selene::shared::record::{DirectoryLock, directory_name, scan};
    bind_or_verify_pool(root, pool_name)?;
    let mut namespaces = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let directory = entry.path();
        let lock = DirectoryLock::acquire(&directory).map_err(|e| invalid(e.to_string()))?;
        let mut header = None;
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() || entry.path().extension().is_none_or(|e| e != "ckpt")
            {
                continue;
            }
            let block = match Header::read_metadata(File::open(entry.path())?) {
                Ok(block) => block,
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::UnexpectedEof | io::ErrorKind::InvalidData
                    ) =>
                {
                    continue;
                }
                Err(e) => return Err(e),
            };
            if let Ok(value) = Header::decode(&block) {
                if entry.file_name().to_str().and_then(record::parse_file_name) == Some(value.key)
                    && entry.path().parent().unwrap().file_name().unwrap()
                        == directory_name(&value.identity, &value.fingerprint).as_str()
                {
                    header = Some(value);
                    break;
                }
            }
        }
        let identity = header.as_ref().map_or([0; 32], |h| h.identity);
        let fingerprint = header.as_ref().map_or([0; 16], |h| h.fingerprint);
        let records = scan(
            &lock,
            &identity,
            &fingerprint,
            header.as_ref().map_or(0, |h| h.page_tokens),
            header.as_ref().map_or(&[], |h| h.ranks.as_slice()),
        )
        .map_err(|e| invalid(e.to_string()))?;
        if let Some(header) = header {
            namespaces.push(Recovered {
                descriptor: Descriptor {
                    identity,
                    fingerprint,
                    page_tokens: header.page_tokens,
                    ranks: header.ranks,
                    pool_name: pool_name.to_owned(),
                },
                records: records
                    .into_iter()
                    .map(|r| (r.header, r.row, Arc::new(DiskExtent::new(r.path))))
                    .collect(),
            });
        }
    }
    Ok(namespaces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Key;
    use flare_selene::shared::record::{RankSizes, checksum, directory_name};
    use std::sync::atomic::AtomicU64;

    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_root(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "selene-disk-{}-{}-{label}",
            std::process::id(),
            TEST_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_sample_record(root: &Path, pool_name: &str) -> (Descriptor, Key, PathBuf) {
        let mut rank = RankSizes {
            planes: vec![4096],
            row_bytes: 4096,
        };
        flare_selene::pool::pad_pool_rank(&mut rank.planes, &mut rank.row_bytes);
        let descriptor = Descriptor {
            identity: [0x11; 32],
            fingerprint: [0x22; 16],
            page_tokens: 64,
            ranks: vec![rank],
            pool_name: pool_name.to_owned(),
        };
        let key = Key {
            consumed_tokens: 64,
            digest: [0x33; 16],
        };
        let mut header = record::new_header(&descriptor, key, None, [0x44; 16], 1).unwrap();
        let mut bytes = vec![0x55u8; header.length_with_row() as usize];
        let page_bytes = header.page_record_bytes() as usize;
        let page_hash = checksum(&bytes[header.page_segment as usize..header.row_segment as usize]);
        header.row_checksum = checksum(&bytes[header.row_segment as usize..]);
        assert_eq!(
            header.row_segment as usize - header.page_segment as usize,
            page_bytes
        );
        bytes[..4096].copy_from_slice(&header.encode().unwrap());
        bytes[4096..4104].copy_from_slice(&page_hash.to_le_bytes());

        let ns_dir = root.join(directory_name(&descriptor.identity, &descriptor.fingerprint));
        fs::create_dir_all(&ns_dir).unwrap();
        let path = ns_dir.join(record::file_name(&key));
        fs::write(&path, &bytes).unwrap();
        (descriptor, key, path)
    }

    #[test]
    fn recover_binds_pool_name_and_rejects_different_pool_without_touching_records() {
        let root = temp_root("bind-and-reject");
        // Initial open on an empty root binds it to "pool-a".
        assert!(recover(&root, "pool-a").unwrap().is_empty());

        let (descriptor, key, ckpt_path) = write_sample_record(&root, "pool-a");

        // Starting a manager with a different --name on the same --root must fail and leave
        // the existing pool's durable files intact.
        let err = recover(&root, "pool-b").err().expect("different pool name must fail");
        let msg = err.to_string();
        assert!(msg.contains("pool-a") && msg.contains("pool-b"), "{msg}");
        assert!(ckpt_path.exists());

        // Restarting with the owning --name recovers the records under "pool-a".
        let recovered = recover(&root, "pool-a").unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].descriptor, descriptor);
        assert_eq!(recovered[0].records.len(), 1);
        assert_eq!(recovered[0].records[0].0.key, key);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn recover_binds_legacy_unbound_root_and_rejects_invalid_names_or_markers() {
        let root = temp_root("legacy-and-invalid");
        let (_, key, _) = write_sample_record(&root, "pool-legacy");

        // Reject empty pool names before creating any marker.
        assert!(recover(&root, "").is_err());
        assert!(!root.join(POOL_MARKER).exists());

        // First recovery on an unbound pre-existing root binds it to the starting pool.
        let recovered = recover(&root, "pool-legacy").unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].records[0].0.key, key);
        assert!(recover(&root, "other-pool").is_err());

        // Corrupt (empty) marker fails recovery cleanly.
        fs::write(root.join(POOL_MARKER), b"\n").unwrap();
        assert!(recover(&root, "pool-legacy").is_err());

        let _ = fs::remove_dir_all(&root);
    }
}
