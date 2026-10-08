#!/usr/bin/env python3
"""Extract raw HiPHI archives or selected motion pairs after the full download.

Examples, from hiphi:
    source scripts/project_env.sh
    "$HIPHI_PYTHON" scripts/extract_raw_release.py --parts 1 \
        --output-root /data/sky/projects/hiphi/datasets/full_v1/HiPHI
    "$HIPHI_PYTHON" scripts/extract_raw_release.py --all-parts \
        --output-root /data/sky/projects/hiphi/datasets/full_v1/HiPHI
    "$HIPHI_PYTHON" scripts/extract_raw_release.py \
        --motion-ids Bringing-carry_0010 \
        --output-root /data/sky/projects/hiphi/datasets/inspect_v1/HiPHI

Motion selection includes the original and its mirror. Existing files must match
the archive bytes; differing files are never overwritten. Uses Python zstandard,
so no system zstd binary is required. Full extraction needs about 846 GB in
addition to the approximately 215 GB raw release. All outputs must be under /data.
"""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import tyro
from collect_stage1 import copy_identical, extract_selected, read_csv, sha256
from mirror_raw_release import atomic_json, monitor, storage_check


@dataclass
class Args:
    output_root: Path
    raw_root: Path = Path("/data/sky/projects/hiphi/datasets/HiPHI")
    manifest: Path = Path(
        "/data/sky/projects/hiphi/runs/full_release_b50ee4b/release_manifest.json"
    )
    parts: tuple[int, ...] = ()
    motion_ids: tuple[str, ...] = ()
    all_parts: bool = False


def main(args):
    if sum((bool(args.parts), bool(args.motion_ids), args.all_parts)) != 1:
        raise ValueError("Choose exactly one of --parts, --motion-ids, or --all-parts")
    output, raw = args.output_root.resolve(), args.raw_root.resolve()
    if not output.is_relative_to(Path("/data")) or output == raw:
        raise ValueError("Use a separate extracted dataset directory under /data")
    manifest = json.loads(args.manifest.read_text())
    inventory = {f["path"]: f for f in manifest["files"]}
    for relative in (
        "data/archive_manifest.csv",
        "data/motion_to_part.csv",
        "metadata/hiphi_metadata.csv",
    ):
        if sha256(raw / relative) != inventory[relative]["sha256"]:
            raise ValueError(f"Source index differs: {relative}")
    archives = read_csv(raw / "data/archive_manifest.csv")
    pairs = read_csv(raw / "data/motion_to_part.csv")
    motions = {r["motion_id"]: r for r in read_csv(raw / "metadata/hiphi_metadata.csv")}
    by_motion = {
        key: pair
        for pair in pairs
        for key in (pair["motion_id"], pair["mirrored_motion_id"])
    }
    if args.motion_ids:
        chosen_pairs = {by_motion[m]["motion_id"] for m in args.motion_ids}
        names = {by_motion[m]["archive_name"] for m in chosen_pairs}
    elif args.all_parts:
        chosen_pairs = None
        names = {a["archive_name"] for a in archives}
    else:
        available = {int(a["part_index"]): a["archive_name"] for a in archives}
        names = {available[p] for p in args.parts}
        chosen_pairs = None
    selected_pairs = [
        p
        for p in pairs
        if p["archive_name"] in names
        and (chosen_pairs is None or p["motion_id"] in chosen_pairs)
    ]
    bound = sum(int(p["uncompressed_bytes"]) for p in selected_pairs)
    storage_check(bound + 2**30)
    print(
        f"Project: hiphi; extraction destination: {output}; PID: {os.getpid()}",
        flush=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    finished = threading.Event()
    monitor_args = SimpleNamespace(command="extract", run_root=output / "_extraction")
    threading.Thread(target=monitor, args=(monitor_args, finished), daemon=True).start()
    try:
        for relative, record in inventory.items():
            if relative.startswith(
                ("object_meshes/", "metadata/", "docs/")
            ) or relative in ("LICENSE.md", "README.md"):
                if sha256(raw / relative) != record["sha256"]:
                    raise ValueError(f"Source companion file differs: {relative}")
                copy_identical(raw / relative, output / relative)
        completed = []
        for name in sorted(names):
            relative = "data/" + name
            if sha256(raw / relative) != inventory[relative]["sha256"]:
                raise ValueError(f"Source archive differs: {name}")
            selected = [p for p in selected_pairs if p["archive_name"] == name]
            rows = [
                motions[m]
                for p in selected
                for m in (p["motion_id"], p["mirrored_motion_id"])
            ]
            storage_check(sum(int(p["uncompressed_bytes"]) for p in selected))
            extract_selected(
                raw / relative,
                output,
                rows,
                sum(int(p["uncompressed_bytes"]) for p in selected),
            )
            for row in rows:
                package = output / "data" / row["frame"] / row["lu"] / row["motion_id"]
                if (
                    not (package / "motion_actor.bvh").is_file()
                    or not (package / "metadata.json").is_file()
                ):
                    raise ValueError(f"Missing extracted package: {row['motion_id']}")
            completed.extend(row["motion_id"] for row in rows)
            atomic_json(
                output / "_extraction/report.json",
                dict(
                    source_revision=manifest["revision"],
                    verified_packages=completed,
                    status="partial"
                    if len(completed) < 2 * len(selected_pairs)
                    else "complete",
                ),
            )
            storage_check()
        print(
            f"Extracted and verified {len(completed)} motion packages: {output}",
            flush=True,
        )
    finally:
        finished.set()


if __name__ == "__main__":
    main(tyro.cli(Args))
