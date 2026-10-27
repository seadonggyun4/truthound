"""Explicit state-pinned reconciliation of retained native filesystem batches."""

import hashlib
import json
import os
import re
import shutil
from contextlib import ExitStack

from truthound.stores.base import StoreReadError, StoreWriteError
from truthound.stores.concurrency.atomic import AtomicFileWriter, atomic_write
from truthound.stores.concurrency.index import ConcurrentIndex
from truthound.stores.concurrency.locks import NoOpLockStrategy
from truthound.stores.results import ValidationResult


def authorize(store):
    if store._io_guard is not None:
        store._io_guard()


def file_hash(path):
    if path.is_symlink():
        raise StoreReadError("Recovery paths cannot be symbolic links")
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(store, identity):
    authorize(store)
    if not isinstance(identity, str) or not re.fullmatch(
        r"\.batch-recovery-[A-Za-z0-9_-]+", identity
    ):
        raise StoreReadError("Invalid recovery identity")
    root = store.config.get_full_path().resolve()
    directory = root / identity
    manifest_path = directory / "manifest.json"
    if directory.is_symlink() or manifest_path.is_symlink():
        raise StoreReadError("Recovery paths cannot be symbolic links")
    try:
        with manifest_path.open("rb") as stream:
            content = stream.read(16 * 1024**2 + 1)
        if len(content) > 16 * 1024**2:
            raise ValueError("Oversized manifest")
        manifest = json.loads(content)
        if manifest["format_version"] != 1 or manifest["state"] not in {
            "preparing",
            "prepared",
            "index_attempted",
            "committed",
            "rolled_back",
            "kept_current",
        }:
            raise ValueError("Invalid manifest version/state")
        files = manifest["files"]
        if not isinstance(files, list) or len(files) > 100_000:
            raise ValueError("Invalid manifest files")
        names, backups, current = set(), set(), {}
        for row in files:
            name, backup, expected = row["name"], row["backup"], row["sha256"]
            if (
                not isinstance(name, str)
                or name.startswith(".")
                or "/" in name
                or "\\" in name
                or name in names
            ):
                raise ValueError("Invalid recovery filename")
            # Native data files, never index/WAL/owner or arbitrary backend files.
            extension = store.config.file_extension + (
                ".gz" if store.config.use_compression else ""
            )
            if not name.endswith(extension) or name in {
                "_index.json",
                "_index.json.wal",
                "_batch-owner",
                "_batch-owner.lock",
            }:
                raise ValueError("Invalid recovery data filename")
            names.add(name)
            authorize(store)
            current[name] = file_hash(root / name)
            if backup is not None:
                if (
                    not isinstance(backup, str)
                    or not re.fullmatch(r"[0-9]+", backup)
                    or backup in backups
                ):
                    raise ValueError("Invalid recovery backup")
                backups.add(backup)
                actual = file_hash(directory / backup)
                if manifest["state"] != "preparing" and (
                    not isinstance(expected, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", expected)
                    or actual != expected
                ):
                    raise ValueError("Invalid recovery backup checksum")
            elif expected is not None:
                raise ValueError("Unexpected backup checksum")
        fingerprint = hashlib.sha256(
            json.dumps(
                {"manifest": hashlib.sha256(content).hexdigest(), "current": current},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return directory, manifest, fingerprint
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise StoreReadError("Recovery evidence is invalid or unreadable") from error


def list_recovery(store, limit, offset):
    authorize(store)
    if limit < 1 or limit > 1000 or offset < 0:
        raise ValueError("Invalid recovery page")
    root = store.config.get_full_path()
    values = sorted((*root.glob(".batch-recovery-*"), *root.glob(".batch-resolved-*")))
    result = []
    for path in values[offset : offset + limit]:
        if path.name.startswith(".batch-resolved-"):
            result.append(resolved_descriptor(store, path.name))
        else:
            _, manifest, fingerprint = inspect(store, path.name)
            result.append(
                {
                    "recovery_id": path.name,
                    "state": manifest["state"],
                    "file_count": len(manifest["files"]),
                    "fingerprint": fingerprint,
                }
            )

    return result


def reconcile(store, identity, resolution, expected_fingerprint):
    authorize(store)
    if resolution not in {"rollback", "keep_current"}:
        raise ValueError("Invalid recovery resolution")
    # Prepare native locking without passing through the quarantine gate.
    with store._initialization_lock:
        store._prepare_storage()
        if isinstance(store._lock_manager._strategy, NoOpLockStrategy):
            raise StoreReadError("Batch recovery requires native process locking")
        owner_path = store.config.get_full_path() / "_batch-owner"
        with store._lock_manager.write_lock(owner_path), ExitStack() as locks:
            directory, manifest, _ = inspect(store, identity)
            root = store.config.get_full_path()
            for row in sorted(manifest["files"], key=lambda row: row["name"]):
                locks.enter_context(store._lock_manager.write_lock(root / row["name"]))
            directory, manifest, fingerprint = inspect(store, identity)
            if fingerprint != expected_fingerprint:
                raise StoreWriteError("Recovery state changed; inspect again")
            # Validate every selected DTO before the first restoration write.
            records = {}
            preparing = manifest["state"] == "preparing"
            for row in manifest["files"]:
                authorize(store)
                path = root / row["name"]
                source = (
                    directory / row["backup"]
                    if resolution == "rollback" and not preparing and row["backup"] is not None
                    else path
                )
                absent = resolution == "rollback" and not preparing and row["backup"] is None
                item = (
                    None
                    if absent or not source.exists()
                    else ValidationResult.from_dict(store._deserialize(source.read_bytes()))
                )
                if item is not None and store._get_file_path(item.id).name != row["name"]:
                    raise StoreReadError("Recovery record identity mismatch")
                records[row["name"]] = item
            fresh = ConcurrentIndex(
                root,
                lock_manager=store._lock_manager,
                wal_enabled=store.config.concurrency.enable_wal,
            )
            try:
                authorize(store)
                fresh.initialize()
                txn = fresh.begin_transaction()
                for row in manifest["files"]:
                    authorize(store)
                    path = root / row["name"]
                    item = records[row["name"]]
                    if resolution == "rollback" and not preparing:
                        if row["backup"] is None:
                            path.unlink(missing_ok=True)
                        else:
                            with (
                                (directory / row["backup"]).open("rb") as original,
                                AtomicFileWriter(path, lock_manager=store._lock_manager) as output,
                            ):
                                shutil.copyfileobj(original, output)
                                if not output.commit().success:
                                    raise StoreWriteError("Recovery file restoration failed")
                    if item is None:
                        extension = store.config.file_extension + (
                            ".gz" if store.config.use_compression else ""
                        )
                        txn.remove(row["name"][: -len(extension)] if extension else row["name"])
                    else:
                        txn.upsert(item.id, store._index_metadata(item, path))
                authorize(store)
                txn.commit()
                completed = finish(
                    store,
                    directory,
                    manifest,
                    "rolled_back" if resolution == "rollback" else "kept_current",
                )
                shutil.rmtree(completed)
                store._index = fresh
                store._initialized = True
                store._batch_uncertain = any(root.glob(".batch-recovery-*"))
                return {
                    "resolved": True,
                    "resolution": resolution,
                    "file_count": len(manifest["files"]),
                    "quarantined": store._batch_uncertain,
                }
            except BaseException:
                store._batch_uncertain = True
                raise


def sync_directory(root):
    if os.name != "nt":
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def finish(store, directory, manifest, state):
    """Publish completed data/index state before deleting any recovery evidence."""
    authorize(store)
    result = atomic_write(
        directory / "manifest.json",
        json.dumps({**manifest, "state": state}),
        lock_manager=store._lock_manager,
    )
    if not result.success:
        raise StoreWriteError("Recovery completion receipt could not be persisted")
    destination = directory.with_name(
        directory.name.replace(".batch-recovery-", ".batch-resolved-", 1)
    )
    if destination.exists():
        raise StoreWriteError("Recovery completion identity conflict")
    os.rename(directory, destination)
    sync_directory(store.config.get_full_path())
    return destination


def resolved_descriptor(store, identity):
    authorize(store)
    if not isinstance(identity, str) or not re.fullmatch(
        r"\.batch-resolved-[A-Za-z0-9_-]+", identity
    ):
        raise StoreReadError("Invalid completed recovery identity")
    directory = store.config.get_full_path() / identity
    if directory.is_symlink() or not directory.is_dir():
        raise StoreReadError("Invalid completed recovery directory")
    entries = []
    for path in sorted(directory.iterdir()):
        authorize(store)
        if path.is_symlink() or not path.is_file():
            raise StoreReadError("Invalid completed recovery evidence")
        entries.append((path.name, file_hash(path)))
    digest = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
    return {
        "recovery_id": identity,
        "state": "cleanup_pending",
        "file_count": len(entries),
        "fingerprint": digest,
    }


def cleanup(store, identity, expected_fingerprint):
    authorize(store)
    with store._initialization_lock:
        store._prepare_storage()
        if isinstance(store._lock_manager._strategy, NoOpLockStrategy):
            raise StoreReadError("Batch recovery requires native process locking")
        with store._lock_manager.write_lock(store.config.get_full_path() / "_batch-owner"):
            descriptor = resolved_descriptor(store, identity)
            if descriptor["fingerprint"] != expected_fingerprint:
                raise StoreWriteError("Recovery state changed; inspect again")
            authorize(store)
            shutil.rmtree(store.config.get_full_path() / identity)
            sync_directory(store.config.get_full_path())
            store._batch_uncertain = any(store.config.get_full_path().glob(".batch-recovery-*"))
            return {"cleaned": True, "quarantined": store._batch_uncertain}
