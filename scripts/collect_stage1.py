#!/usr/bin/env python3
"""Plan and collect a small, reproducible HiPHI subset for humanoid retargeting.

Source project_env.sh first. Credentials are resolved by huggingface_hub; never
put a token in this script, its arguments, or a collection manifest.

Examples (run from /home/sky/sky_workdir/heavyhumanoid/hiphi):
    source scripts/project_env.sh
    env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "$HIPHI_PYTHON" scripts/collect_stage1.py plan
    env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "$HIPHI_PYTHON" scripts/collect_stage1.py collect
    "$HIPHI_PYTHON" scripts/collect_stage1.py validate

The env -u flags let this node use its approved saved Hugging Face login.
For a separate pilot, pass --subset-name stage1_v2 --per-group 4 to plan,
then use the same --subset-name for collect and validate.
"""
from __future__ import annotations

import collections
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
from typing import Annotated, Literal

from huggingface_hub import HfApi, hf_hub_download
import tyro
import zstandard

REPO = "noitomrobotics/HiPHI"
GROUPS = ("reaching", "lifting", "carrying", "lowering_placing", "pushing", "walking", "posture")
GIB = 2**30
METADATA = ("metadata/frame_lu_index.csv", "metadata/hiphi_metadata.csv",
            "metadata/hiphi_actor_metadata.csv", "data/archive_manifest.csv",
            "data/motion_to_part.csv", "LICENSE.md")


@dataclass
class CollectionArgs:
    # Plan a selection, download/extract it, or revalidate its local files.
    command: Annotated[Literal["plan", "collect", "validate"], tyro.conf.Positional]
    # Name of the dataset and run directory under PROJECT_STORAGE_ROOT.
    subset_name: str = "stage1_v1"
    # Minimum number of clips covering each behavior; groups can overlap.
    per_group: int = 8
    # Maximum number of full compressed archives to download.
    max_archives: int = 3
    # Compressed archive budget in GiB.
    max_download_gib: float = 24.0
    # Deterministic tie-breaking seed for motion selection.
    seed: int = 20260905


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def store_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def storage_check(needed=0):
    root = shutil.disk_usage("/")
    data = shutil.disk_usage("/data")
    inodes = os.statvfs("/")
    print(f"Storage: root {root.free/GIB:.1f} GiB free; /data {data.free/GIB:.1f} GiB free", flush=True)
    if root.free < 350 * GIB or inodes.f_favail / inodes.f_files < 0.1:
        raise RuntimeError("Root filesystem storage safety threshold reached")
    if data.free < needed + 10 * GIB:
        raise RuntimeError("Insufficient /data space for operation plus 10 GiB margin")


def motion_groups(row):
    groups = []
    label = row["frame_lu"]
    hoi = row["is_hoi"].lower() == "true"
    if not hoi:
        if label == "Body_movement-reach":
            groups.append("reaching")
        if label == "Self_motion-walk":
            groups.append("walking")
        if label in {"Body_movement-bend", "Change_posture-bend", "Change_posture-squat_down"}:
            groups.append("posture")
    if hoi and row["object_categories"] == "box":
        if label == "Cause_motion-lift":
            groups.append("lifting")
        if label == "Bringing-carry":
            groups.append("carrying")
        if label == "Cause_motion-push":
            groups.append("pushing")
        # Candidate phase coverage from text, not verified frame-level labels.
        if label in {"Placing-put", "Placing-situate", "Bringing-carry", "Cause_motion-lift"}:
            if re.search(r"(lower|put|set|place).*\b(floor|ground)\b", row["text_annotation"].lower()):
                groups.append("lowering_placing")
    return groups


def plan_collection(args, raw, run):
    storage_check()
    api = HfApi()
    source_file = raw / "source_revision.json"
    revision = json.loads(source_file.read_text())["revision"] if source_file.exists() else None
    info = api.dataset_info(REPO, revision=revision, files_metadata=True)
    revision = info.sha
    files = {f.rfilename: f for f in info.siblings}
    for filename in METADATA:
        hf_hub_download(REPO, filename, repo_type="dataset", revision=revision, local_dir=raw)
    store_json(source_file, {"repo_id": REPO, "revision": revision})
    part_rows = {r["motion_id"]: r for r in read_csv(raw / "data/motion_to_part.csv")}
    archives = {r["archive_name"]: r for r in read_csv(raw / "data/archive_manifest.csv")}
    candidates = []
    by_part = collections.defaultdict(collections.Counter)
    for row in read_csv(raw / "metadata/hiphi_metadata.csv"):
        if row["mirrored"].lower() != "false":
            continue
        groups = motion_groups(row)
        if not groups:
            continue
        row = dict(row, groups=groups, archive=part_rows[row["motion_id"]]["archive_name"])
        candidates.append(row)
        by_part[row["archive"]].update(groups)
    chosen_parts = None
    # Prefer the fewest archives, then the smallest total compressed size.
    for count in range(1, args.max_archives + 1):
        feasible = []
        for parts in itertools.combinations(sorted(by_part), count):
            size = sum(int(archives[p]["compressed_bytes"]) for p in parts)
            if size > args.max_download_gib * GIB:
                continue
            if all(sum(by_part[p][g] for p in parts) >= args.per_group for g in GROUPS):
                feasible.append((size, parts))
        if feasible:
            _, chosen_parts = min(feasible)
            break
    if chosen_parts is None:
        raise RuntimeError("Coverage does not fit the archive/download limits; adjust the explicit plan limits")
    pool = [r for r in candidates if r["archive"] in chosen_parts]
    selected, selected_ids = [], set()
    coverage, actors = collections.Counter(), collections.Counter()
    while any(coverage[g] < args.per_group for g in GROUPS):
        unmet = [g for g in GROUPS if coverage[g] < args.per_group]
        group = min(unmet, key=lambda g: (sum(g in r["groups"] and r["motion_id"] not in selected_ids for r in pool), g))
        options = [r for r in pool if group in r["groups"] and r["motion_id"] not in selected_ids]
        row = min(options, key=lambda r: (
            actors[r["actor_id"]],
            -sum(coverage[g] < args.per_group for g in r["groups"]),
            hashlib.sha256((str(args.seed) + r["motion_id"]).encode()).hexdigest()))
        selected.append(row)
        selected_ids.add(row["motion_id"])
        coverage.update(row["groups"])
        actors.update([row["actor_id"]])
    archive_list = []
    for name in chosen_parts:
        remote = files["data/" + name]
        archive_list.append({"filename": "data/" + name, "compressed_bytes": remote.size,
                             "sha256": remote.lfs.sha256,
                             "full_uncompressed_bytes": int(archives[name]["uncompressed_bytes"])})
    # Pair sizes include the unused mirror, so this overestimates extracted bytes.
    extract_bound = sum(int(part_rows[r["motion_id"]]["uncompressed_bytes"]) for r in selected)
    download_bytes = sum(a["compressed_bytes"] for a in archive_list)
    manifest = {"repo_id": REPO, "revision": revision, "seed": args.seed,
                "per_group_target": args.per_group, "coverage": dict(coverage),
                "unique_clips": len(selected), "unique_actors": len(actors),
                "duration_seconds": sum(float(r["duration_sec"]) for r in selected),
                "download_bytes": download_bytes, "extraction_upper_bound_bytes": extract_bound,
                "peak_new_bytes_upper_bound": download_bytes + extract_bound + GIB,
                "archives": archive_list, "motions": sorted(selected, key=lambda r: r["motion_id"]),
                "selection_note": "Archive-budgeted pilot with actor diversity; not a balanced benchmark. Coverage groups overlap. Lowering/placing uses clip descriptions and needs visual review. No mirrored clips or train/test split yet."}
    existing = run / "plan.json"
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise RuntimeError("Existing plan differs; choose a new --subset-name to preserve it")
    store_json(existing, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"motions", "archives"}}, indent=2), flush=True)
    print("Archives:", [a["filename"] for a in archive_list], flush=True)
    storage_check(manifest["peak_new_bytes_upper_bound"])


def extract_selected(archive_path, destination, selected, max_bytes):
    prefixes = {("HiPHI", "data", r["frame"], r["lu"], r["motion_id"]) for r in selected}
    written = 0
    with archive_path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    if parts[:5] not in prefixes:
                        continue
                    if member.isdir():
                        continue
                    if not member.isfile() or ".." in parts or member.name.startswith("/"):
                        raise ValueError(f"Unsafe selected archive member: {member.name}")
                    written += member.size
                    if written > max_bytes:
                        raise RuntimeError("Selected extraction exceeded its metadata size bound")
                    path = destination.joinpath(*parts[1:])
                    if not path.resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Archive destination escapes subset")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if path.exists():
                        digest = hashlib.sha256()
                        for block in iter(lambda: source.read(1024**2), b""):
                            digest.update(block)
                        if path.stat().st_size != member.size or sha256(path) != digest.hexdigest():
                            raise FileExistsError(f"Existing extracted file differs: {path}")
                        continue
                    temp = path.with_name(path.name + ".partial")
                    with temp.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024**2)
                    if temp.stat().st_size != member.size:
                        raise RuntimeError(f"Truncated extraction: {member.name}")
                    temp.replace(path)
    print(f"Extracted/verified {written/GIB:.3f} GiB from {archive_path.name}", flush=True)


def copy_identical(source, destination):
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise FileExistsError(f"Refusing to replace differing file: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def collect(raw, run, subset):
    manifest = json.loads((run / "plan.json").read_text())
    storage_check(manifest["peak_new_bytes_upper_bound"])
    store_json(run / "collector_identity.json", {"pid": os.getpid(), "started_unix": time.time()})
    subset.mkdir(parents=True, exist_ok=True)
    for archive in manifest["archives"]:
        storage_check(archive["compressed_bytes"] + manifest["extraction_upper_bound_bytes"])
        print("Downloading:", archive["filename"], flush=True)
        path = Path(hf_hub_download(REPO, archive["filename"], repo_type="dataset",
                                   revision=manifest["revision"], local_dir=raw))
        if path.stat().st_size != archive["compressed_bytes"] or sha256(path) != archive["sha256"]:
            raise RuntimeError(f"Archive checksum/size mismatch: {path.name}")
        print("SHA-256 verified:", path.name, flush=True)
        selected = [r for r in manifest["motions"] if r["archive"] == path.name]
        extract_selected(path, subset, selected, manifest["extraction_upper_bound_bytes"])
        storage_check()
    meshes = set()
    for motion in manifest["motions"]:
        meta_path = subset / "data" / motion["frame"] / motion["lu"] / motion["motion_id"] / "metadata.json"
        metadata = json.loads(meta_path.read_text())
        for obj in metadata["objects"]:
            mesh = obj["mesh_path"]
            parts = PurePosixPath(mesh).parts
            if len(parts) != 2 or parts[0] != "object_meshes" or ".." in parts:
                raise ValueError(f"Unexpected mesh path: {mesh}")
            meshes.add(mesh)
    for mesh in sorted(meshes):
        path = hf_hub_download(REPO, mesh, repo_type="dataset", revision=manifest["revision"], local_dir=raw)
        copy_identical(Path(path), subset / mesh)
    for name in ("LICENSE.md", "metadata/hiphi_actor_metadata.csv"):
        copy_identical(raw / name, subset / name)
    # Viewer index lists selected clips only; source indices stay untouched in raw.
    output = subset / "metadata/hiphi_metadata.csv"
    fields = list(read_csv(raw / "metadata/hiphi_metadata.csv")[0])
    temp = output.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest["motions"])
    temp.replace(output)
    copy_identical(run / "plan.json", subset / "selection_manifest.json")
    validate(run, subset)
    storage_check()


def validate(run, subset):
    manifest = json.loads((run / "plan.json").read_text())
    records, meshes = [], set()
    for motion in manifest["motions"]:
        package = subset / "data" / motion["frame"] / motion["lu"] / motion["motion_id"]
        metadata = json.loads((package / "metadata.json").read_text())
        if metadata["motion_id"] != motion["motion_id"] or metadata["mirrored"]:
            raise ValueError(f"Package identity mismatch: {package}")
        if metadata["actor_id"] != motion["actor_id"] or bool(metadata["is_hoi"]) != (motion["is_hoi"] == "true"):
            raise ValueError(f"Metadata/index mismatch: {package}")
        channels, frames, frame_time, actual_frames = 0, None, None, 0
        with (package / "motion_actor.bvh").open() as stream:
            for line in stream:
                words = line.split()
                if not words:
                    continue
                if words[0] == "CHANNELS":
                    channels += int(words[1])
                elif words[0] == "Frames:":
                    frames = int(words[1])
                elif words[:2] == ["Frame", "Time:"]:
                    frame_time = float(words[2])
                elif frame_time is not None:
                    if len(words) != channels or not all(math.isfinite(float(v)) for v in words):
                        raise ValueError(f"Invalid BVH motion row: {package}")
                    actual_frames += 1
        if frames != actual_frames or frames != int(motion["frame_count"]) or frames != metadata["frame_count"]:
            raise ValueError(f"BVH frame count mismatch: {package}")
        if not frame_time or channels != 330:
            raise ValueError(f"Unexpected BVH timing/channel schema: {package}")
        if metadata["is_hoi"] and not metadata["objects"]:
            raise ValueError(f"HOI package has no objects: {package}")
        for obj in metadata["objects"]:
            track_path = (package / obj["trajectory_path"]).resolve()
            if not track_path.is_relative_to(package.resolve()):
                raise ValueError("Object trajectory path escapes package")
            tracks = read_csv(track_path)
            if len(tracks) != frames:
                raise ValueError(f"Object/BVH frame count mismatch: {package}")
            for index, row in enumerate(tracks):
                numbers = [float(row[k]) for k in ("time_sec", "px", "py", "pz", "qx", "qy", "qz", "qw")]
                if not all(math.isfinite(v) for v in numbers) or int(row["frame"]) != index:
                    raise ValueError(f"Invalid object trajectory: {track_path}")
                if abs(float(row["time_sec"]) - index * frame_time) > 2e-5:
                    raise ValueError(f"Object/BVH timestamp mismatch: {track_path}")
                if abs(sum(float(row[k])**2 for k in ("qx", "qy", "qz", "qw")) - 1) > 0.005:
                    raise ValueError(f"Non-unit object quaternion: {track_path}")
            mesh = subset / obj["mesh_path"]
            if not mesh.is_file() or mesh.stat().st_size == 0:
                raise ValueError(f"Missing object mesh: {mesh}")
            meshes.add(obj["mesh_path"])
        records.append({"motion_id": motion["motion_id"], "actor_id": motion["actor_id"],
                        "groups": motion["groups"], "frames": frames, "object_count": len(metadata["objects"]),
                        "files": [{"path": str(p.relative_to(subset)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                                  for p in sorted(package.rglob("*")) if p.is_file()]})
    report = {"status": "validated_raw_data", "revision": manifest["revision"],
              "unique_clips": len(records), "unique_actors": manifest["unique_actors"],
              "coverage": manifest["coverage"], "duration_minutes": manifest["duration_seconds"] / 60,
              "object_mesh_count": len(meshes), "hoi_clips": sum(r["object_count"] > 0 for r in records),
              "total_subset_bytes": sum(p.stat().st_size for p in subset.rglob("*") if p.is_file()),
              "limits": "Checks file integrity and synchronization only. Visual phase review, retargeting, and physical feasibility are not yet validated.",
              "meshes": [{"path": mesh, "sha256": sha256(subset / mesh)} for mesh in sorted(meshes)],
              "motions": records}
    store_json(run / "validation.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"motions", "meshes"}}, indent=2), flush=True)


def main():
    args = tyro.cli(CollectionArgs, description=__doc__)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.subset_name):
        raise SystemExit("subset-name must contain only letters, digits, underscores, and hyphens")
    if args.per_group < 1 or not 1 <= args.max_archives <= 4 or args.max_download_gib <= 0:
        raise SystemExit("Use a positive per-group/download budget and one to four archives")
    storage = Path(os.environ["PROJECT_STORAGE_ROOT"]).resolve()
    if not storage.is_relative_to(Path("/data/sky/projects")):
        raise SystemExit("PROJECT_STORAGE_ROOT must be under /data/sky/projects")
    raw, run = storage / "datasets/HiPHI", storage / "runs" / args.subset_name
    subset = storage / "datasets" / args.subset_name / "HiPHI"
    run.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        plan_collection(args, raw, run)
    elif args.command == "collect":
        collect(raw, run, subset)
    else:
        validate(run, subset)


if __name__ == "__main__":
    main()
