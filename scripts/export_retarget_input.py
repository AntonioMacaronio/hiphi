#!/usr/bin/env python3
"""Export synchronized HiPHI positions and object geometry for robot retargeting.

Example, from the hiphi checkout (all outputs go to /data):
    source scripts/project_env.sh
    "$HIPHI_PYTHON" scripts/export_retarget_input.py --motion-id Bringing-carry_0010 \
        --start-frame 720 --stop-frame 1260 --stride 3 \
        --output /data/sky/projects/hiphi/runs/retarget_inputs/carry_0010.npz

stop-frame is exclusive. Striding samples the original timing, without slowing it.
This exports the raw geometry; it does not infer or correct hand contacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from build_frame_lu_bvh_previews import parse_bvh
from view_stage1_viser import load_motion


@dataclass
class Args:
    motion_id: str
    output: Path
    data_root: Path = Path("/data/sky/projects/hiphi/datasets/stage1_v1/HiPHI")
    start_frame: int = 0
    stop_frame: int | None = None
    stride: int = 3


def main(args: Args) -> None:
    if args.stride < 1 or args.start_frame < 0:
        raise ValueError("stride must be positive and start-frame nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    matches = list((args.data_root / "data").glob(f"*/*/{args.motion_id}/metadata.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one package for {args.motion_id}; found {len(matches)}")
    package = matches[0].parent
    motion = load_motion(package, args.data_root)
    if len(motion.objects) != 1:
        raise ValueError("This first adapter supports exactly one tracked object")
    nodes, count, _, _, _ = parse_bvh(package / "motion_actor.bvh")
    stop = count if args.stop_frame is None else args.stop_frame
    if not args.start_frame < stop <= count:
        raise ValueError(f"Frame interval must lie within [0, {count})")
    frames = np.arange(args.start_frame, stop, args.stride)
    if len(frames) < 2:
        raise ValueError("Select at least two frames")
    obj = motion.objects[0]
    metadata = dict(motion.metadata)
    files = [package / "motion_actor.bvh", matches[0]]
    files += [
        package / motion.metadata["objects"][0]["trajectory_path"],
        args.data_root / motion.metadata["objects"][0]["mesh_path"],
    ]
    metadata["source_sha256"] = {
        str(p.relative_to(args.data_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    metadata["coordinate_convention"] = "meters, right-handed Y-up; common initial-hips XZ recenter"
    metadata["quaternion_convention"] = "object-local-to-world WXYZ"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=1,
        metadata_json=json.dumps(metadata),
        joint_names=np.array([node.name for node in nodes]),
        edges=motion.edges,
        human_positions=motion.positions[frames],
        object_positions=obj["positions"][frames],
        object_quaternions_wxyz=obj["wxyz"][frames],
        object_vertices=obj["vertices"],
        object_faces=obj["faces"],
        source_frames=frames,
        source_timestamps=frames * motion.frame_time,
        fps=1 / (args.stride * motion.frame_time),
    )
    print(f"Saved {len(frames)} frames at {1 / (args.stride * motion.frame_time):.6f} Hz: {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
