#!/usr/bin/env python3
"""Mirror every file of a pinned raw HiPHI release locally and to private S3.

Examples, from the hiphi checkout:
    source scripts/project_env.sh
    env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "$HIPHI_PYTHON" \
        scripts/mirror_raw_release.py download
    "$HIPHI_PYTHON" scripts/mirror_raw_release.py upload
    "$HIPHI_PYTHON" scripts/mirror_raw_release.py verify
    "$HIPHI_PYTHON" scripts/mirror_raw_release.py copy \
        --source-prefix antzhan/datasets/HiPHI/b50ee4bcff519365503fec6146f8aca93f10eb49

The approved saved Hugging Face login is used for downloading. Never pass tokens
in arguments. Original release archives remain compressed and byte-identical;
no source files, caches, or credentials outside the explicit catalog are uploaded.
Existing differing files/objects cause an error instead of being overwritten.
Progress, integrity records, and logs belong under /data, not in the checkout.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import boto3
import tyro
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from huggingface_hub import HfApi, hf_hub_download

REPO = "noitomrobotics/HiPHI"
REVISION = "b50ee4bcff519365503fec6146f8aca93f10eb49"
GIB = 2**30
PART_BYTES = 64 * 2**20
MULTIPART_THRESHOLD = 128 * 2**20


@dataclass
class Args:
    command: Annotated[
        Literal["download", "upload", "copy", "verify"], tyro.conf.Positional
    ]
    raw_root: Path = Path("/data/sky/projects/hiphi/datasets/HiPHI")
    run_root: Path = Path("/data/sky/projects/hiphi/runs/full_release_b50ee4b")
    revision: str = REVISION
    bucket: str = "far-research-internal"
    prefix: str = "antzhan/datasets/HiPHI/2026-09-09"
    source_prefix: str | None = None
    profile: str = "node-identity"
    region: str = "us-west-2"
    workers: int = 4


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def storage_check(needed=0, verbose=True):
    root, data = shutil.disk_usage("/"), shutil.disk_usage("/data")
    inodes = os.statvfs("/")
    if verbose:
        subprocess.run(["df", "-hT", "/", "/data"], check=True)
        subprocess.run(["df", "-i", "/", "/data"], check=True)
    if root.free < 350 * GIB or inodes.f_favail / inodes.f_files < 0.1:
        raise RuntimeError("Root filesystem safety threshold reached")
    if data.free < needed + 20 * GIB:
        raise RuntimeError("Insufficient /data space including safety margin")
    return dict(
        root_free_bytes=root.free,
        data_free_bytes=data.free,
        root_free_inode_fraction=inodes.f_favail / inodes.f_files,
    )


def monitor(args, finished):
    while not finished.wait(60):
        try:
            health = storage_check(verbose=False)
            atomic_json(
                args.run_root / f"{args.command}_heartbeat.json",
                dict(pid=os.getpid(), unix_time=time.time(), **health),
            )
            print(
                f"Heartbeat: {args.command}; storage healthy; PID {os.getpid()}",
                flush=True,
            )
        except Exception as error:
            print(
                f"Stopping this job only: {type(error).__name__}: {error}", flush=True
            )
            os._exit(75)


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Unsafe catalog path")
    destination = root.joinpath(*p.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Catalog path escapes raw root")
    return destination


def file_integrity(path, expected):
    """Verify HF blob identity and compute the S3 SHA256 checksum independently."""
    stat = path.stat()
    if stat.st_size != expected["bytes"]:
        raise ValueError(f"File size differs: {expected['path']}")
    whole = hashlib.sha256()
    git = hashlib.sha1(f"blob {stat.st_size}\0".encode())
    parts = []
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(PART_BYTES), b""):
            whole.update(block)
            git.update(block)
            parts.append(hashlib.sha256(block).digest())
    digest = whole.hexdigest()
    if expected["lfs_sha256"]:
        valid = digest == expected["lfs_sha256"]
    else:
        valid = git.hexdigest() == expected["git_blob_sha1"]
    if not valid:
        raise ValueError(f"Hugging Face content hash differs: {expected['path']}")
    multipart = stat.st_size >= MULTIPART_THRESHOLD
    checksum = (
        base64.b64encode(hashlib.sha256(b"".join(parts)).digest()).decode()
        + f"-{len(parts)}"
        if multipart
        else base64.b64encode(whole.digest()).decode()
    )
    return dict(
        **expected,
        sha256=digest,
        verified_git_blob_sha1=git.hexdigest(),
        s3_checksum_sha256=checksum,
        s3_checksum_type="COMPOSITE" if multipart else "FULL_OBJECT",
        s3_part_bytes=PART_BYTES if multipart else None,
        mtime_ns=stat.st_mtime_ns,
    )


def load_catalog(args):
    catalog = json.loads((args.run_root / "catalog.json").read_text())
    if catalog["repo_id"] != REPO or catalog["revision"] != args.revision:
        raise ValueError("Catalog release differs from command arguments")
    return catalog


def download(args):
    info = HfApi().dataset_info(REPO, revision=args.revision, files_metadata=True)
    files = [
        dict(
            path=f.rfilename,
            bytes=f.size,
            git_blob_sha1=f.blob_id,
            lfs_sha256=f.lfs.sha256 if f.lfs else None,
        )
        for f in info.siblings
    ]
    if info.sha != args.revision or any(f["bytes"] is None for f in files):
        raise ValueError("Expected a pinned complete release catalog")
    catalog = dict(
        repo_id=REPO,
        revision=info.sha,
        files=sorted(files, key=lambda f: f["path"]),
        file_count=len(files),
        total_bytes=sum(f["bytes"] for f in files),
    )
    catalog_path = args.run_root / "catalog.json"
    if catalog_path.exists() and json.loads(catalog_path.read_text()) != catalog:
        raise FileExistsError("Existing catalog differs; use another run root")
    atomic_json(catalog_path, catalog)
    # Includes all downloads and ample room for network/cache scratch. Extraction
    # is a separate, explicit operation and needs an additional ~846 GB.
    storage_check(catalog["total_bytes"] + 20 * GIB)
    verified = []

    def fetch(entry):
        destination = safe_path(args.raw_root, entry["path"])
        if destination.exists():
            record = file_integrity(destination, entry)
            print(f"Verified existing: {entry['path']}", flush=True)
            return record
        if entry["bytes"] > 5 * GIB:
            storage_check()
        print(f"Downloading: {entry['path']} ({entry['bytes']} bytes)", flush=True)
        for attempt in range(3):
            try:
                path = hf_hub_download(
                    REPO,
                    entry["path"],
                    repo_type="dataset",
                    revision=args.revision,
                    local_dir=args.raw_root,
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                print(f"Retry {attempt + 1}: {entry['path']}", flush=True)
                time.sleep(2 ** (attempt + 1))
        record = file_integrity(Path(path), entry)
        if entry["bytes"] > 5 * GIB:
            storage_check()
        print(f"Downloaded and hash-verified: {entry['path']}", flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, entry) for entry in catalog["files"]]
        for future in concurrent.futures.as_completed(futures):
            verified.append(future.result())
            atomic_json(
                args.run_root / "download_progress.json",
                dict(
                    verified_files=len(verified),
                    verified_bytes=sum(r["bytes"] for r in verified),
                    total_files=catalog["file_count"],
                    total_bytes=catalog["total_bytes"],
                    updated_unix=time.time(),
                    records=sorted(verified, key=lambda r: r["path"]),
                ),
            )
    report = dict(
        repo_id=REPO,
        revision=args.revision,
        local_root=str(args.raw_root),
        file_count=len(verified),
        total_bytes=sum(r["bytes"] for r in verified),
        files=sorted(verified, key=lambda r: r["path"]),
        status="all_local_files_hash_verified",
    )
    atomic_json(args.run_root / "release_manifest.json", report)
    print(
        f"DOWNLOAD COMPLETE: {len(verified)} files, {report['total_bytes']} bytes",
        flush=True,
    )


def s3_client(args):
    return boto3.Session(profile_name=args.profile).client(
        "s3",
        region_name=args.region,
        config=Config(
            connect_timeout=15,
            read_timeout=90,
            max_pool_connections=32,
            retries={"mode": "standard", "max_attempts": 5},
        ),
    )


def verify_remote(client, args, record):
    key = f"{args.prefix.rstrip('/')}/{record['path']}"
    response = client.head_object(Bucket=args.bucket, Key=key, ChecksumMode="ENABLED")
    if response["ContentLength"] != record["bytes"]:
        raise ValueError(f"S3 size differs: {key}")
    if response.get("ChecksumSHA256") != record["s3_checksum_sha256"]:
        raise ValueError(f"S3 stored SHA256 checksum differs: {key}")
    return dict(
        path=record["path"],
        bytes=response["ContentLength"],
        key=key,
        checksum_sha256=response["ChecksumSHA256"],
        checksum_type=response.get("ChecksumType"),
        sha256=record["sha256"],
        etag=response["ETag"],
        version_id=response.get("VersionId"),
    )


def upload_or_verify(args):
    catalog = load_catalog(args)
    manifest = json.loads((args.run_root / "release_manifest.json").read_text())
    expected = {f["path"]: f for f in catalog["files"]}
    if {f["path"] for f in manifest["files"]} != set(expected):
        raise ValueError("Local manifest does not cover the complete source catalog")
    client = s3_client(args)
    config = TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD,
        multipart_chunksize=PART_BYTES,
        max_concurrency=4,
        preferred_transfer_client="classic",
    )
    completed = []

    def transfer(old_record):
        path = safe_path(args.raw_root, old_record["path"])
        record = file_integrity(path, expected[old_record["path"]])
        key = f"{args.prefix.rstrip('/')}/{record['path']}"
        try:
            result = verify_remote(client, args, record)
            print(f"Verified S3: {record['path']}", flush=True)
            return result
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
            if args.command not in ("upload", "copy"):
                raise
        print(f"Transferring ({args.command}): {record['path']}", flush=True)
        content_type = (
            mimetypes.guess_type(record["path"])[0] or "application/octet-stream"
        )
        extra = dict(
            ContentType=content_type,
            ChecksumAlgorithm="SHA256",
            Metadata={"sha256": record["sha256"], "hf-revision": args.revision},
        )
        if args.command == "copy":
            source = verify_remote(
                client, replace(args, prefix=args.source_prefix), record
            )
            client.copy(
                {"Bucket": args.bucket, "Key": source["key"]},
                args.bucket,
                key,
                ExtraArgs={
                    **extra,
                    "CopySourceIfMatch": source["etag"],
                    "MetadataDirective": "REPLACE",
                },
                Config=config,
            )
        else:
            client.upload_file(
                str(path), args.bucket, key, ExtraArgs=extra, Config=config
            )
        result = verify_remote(client, args, record)
        print(f"Transferred and S3-checksum-verified: {record['path']}", flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed(
            [pool.submit(transfer, r) for r in manifest["files"]]
        ):
            completed.append(future.result())
            atomic_json(
                args.run_root / f"{args.command}_progress.json",
                dict(
                    verified_files=len(completed),
                    verified_bytes=sum(r["bytes"] for r in completed),
                    updated_unix=time.time(),
                    records=sorted(completed, key=lambda r: r["path"]),
                ),
            )
    listing = {}
    prefix = args.prefix.rstrip("/") + "/"
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=args.bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(prefix) :]
            if not relative.startswith("_mirror/"):
                listing[relative] = obj["Size"]
    if listing != {name: f["bytes"] for name, f in expected.items()}:
        raise ValueError("S3 inventory differs from the full source catalog")
    report = dict(
        status="complete_source_catalog_verified_locally_and_on_s3",
        repo_id=REPO,
        revision=args.revision,
        s3_uri=f"s3://{args.bucket}/{prefix}",
        file_count=len(completed),
        total_bytes=sum(r["bytes"] for r in completed),
        files=sorted(completed, key=lambda r: r["path"]),
        verified_unix=time.time(),
        verification=(
            "HF LFS SHA256 or Git blob SHA1 locally; "
            "independent S3 SHA256 checksum and exact inventory"
        ),
    )
    atomic_json(args.run_root / "s3_verification.json", report)
    if args.command in ("upload", "copy"):
        for name in ("catalog.json", "release_manifest.json", "s3_verification.json"):
            body = (args.run_root / name).read_bytes()
            key = prefix + "_mirror/" + name
            checksum = base64.b64encode(hashlib.sha256(body).digest()).decode()
            client.put_object(
                Bucket=args.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                ChecksumSHA256=checksum,
            )
            remote = client.head_object(
                Bucket=args.bucket, Key=key, ChecksumMode="ENABLED"
            )
            if remote.get("ChecksumSHA256") != checksum:
                raise ValueError("Mirror report checksum differs")
    print(
        f"{args.command.upper()} COMPLETE: {len(completed)} source files, "
        f"{report['total_bytes']} bytes",
        flush=True,
    )


def main(args):
    for path in (args.raw_root, args.run_root):
        if not path.resolve().is_relative_to(Path("/data")):
            raise ValueError("Dataset and run output paths must be under /data")
        path.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.workers > 8:
        raise ValueError("Use between 1 and 8 file workers")
    if (
        not args.prefix
        or args.prefix.strip("/") != args.prefix
        or ".." in PurePosixPath(args.prefix).parts
    ):
        raise ValueError("Use a normalized nonempty S3 prefix")
    if args.command == "copy" and (
        not args.source_prefix
        or args.source_prefix.strip("/") != args.source_prefix
        or ".." in PurePosixPath(args.source_prefix).parts
        or args.source_prefix == args.prefix
    ):
        raise ValueError(
            "Copy requires a normalized source prefix different from the destination"
        )
    print(
        f"Project: hiphi; raw data: {args.raw_root}; run records: {args.run_root}",
        flush=True,
    )
    print(
        f"S3: s3://{args.bucket}/{args.prefix}/; process PID: {os.getpid()}", flush=True
    )
    atomic_json(
        args.run_root / f"{args.command}_identity.json",
        dict(pid=os.getpid(), started_unix=time.time(), revision=args.revision),
    )
    storage_check()
    finished = threading.Event()
    watcher = threading.Thread(target=monitor, args=(args, finished), daemon=True)
    watcher.start()
    try:
        if args.command == "download":
            download(args)
        else:
            upload_or_verify(args)
        storage_check()
    finally:
        finished.set()


if __name__ == "__main__":
    main(tyro.cli(Args))
