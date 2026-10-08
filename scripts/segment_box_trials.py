#!/usr/bin/env python3
"""Propose reviewable box-lift trials from synchronized HiPHI motion.

Examples, from the hiphi checkout:
    source scripts/project_env.sh
    "$HIPHI_PYTHON" scripts/segment_box_trials.py \
        --motion-ids Bringing-carry_0010 Placing-put_0003 \
        --output /data/sky/projects/hiphi/runs/trials_v1/trials.json
    "$HIPHI_PYTHON" scripts/view_stage1_viser.py \
        --trial-manifest /data/sky/projects/hiphi/runs/trials_v1/trials.json

Frame ranges are zero-based, half-open [start_frame, stop_frame) in the original
BVH. This is a floor-level box-lift proposal tool, not a general action segmenter:
sliding, table placement, and body-only motions require other boundary criteria.
Distance to an object-local bounding box proposes hand separation, not contact.
All proposals need visual review; neither human nor robot success is asserted.
Source files are never changed and existing output manifests are not overwritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import tyro
from build_frame_lu_bvh_previews import parse_bvh
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation
from view_stage1_viser import load_motion


@dataclass
class Thresholds:
    lift_height_m: float = 0.10
    floor_height_tolerance_m: float = 0.06
    maximum_floor_penetration_m: float = 0.08
    stable_speed_m_s: float = 0.05
    stable_angular_speed_rad_s: float = 0.30
    hands_clear_distance_m: float = 0.12
    minimum_lift_s: float = 0.40
    minimum_rest_s: float = 0.30
    approach_padding_s: float = 1.25
    settle_padding_s: float = 0.50
    smoothing_s: float = 0.10


@dataclass
class Args:
    output: Path
    motion_ids: tuple[str, ...] = ("Bringing-carry_0010", "Placing-put_0003")
    data_root: Path = Path("/data/sky/projects/hiphi/datasets/stage1_v1/HiPHI")
    thresholds: Thresholds = field(default_factory=Thresholds)


def runs(mask, minimum_frames=1):
    """Half-open true intervals, including intervals touching either file boundary."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        if b - a >= minimum_frames
    ]


def source_features(motion, joint_names, cfg):
    """Measure native Y-up source motion using an oriented bounding-box proxy."""
    obj = motion.objects[0]
    p = obj["positions"]
    rotation = Rotation.from_quat(obj["wxyz"][:, [1, 2, 3, 0]])
    matrix = rotation.as_matrix()
    dt = motion.frame_time
    window = max(1, round(cfg.smoothing_s / dt)) | 1
    low, high = obj["vertices"].min(axis=0), obj["vertices"].max(axis=0)
    corners = np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    bottom = p[:, 1] + np.einsum("tij,vj->tvi", matrix, corners)[:, :, 1].min(axis=1)
    speed = np.linalg.norm(
        np.gradient(median_filter(p, size=(window, 1)), dt, axis=0), axis=1
    )
    angular = np.r_[0.0, (rotation[:-1].inv() * rotation[1:]).magnitude() / dt]
    angular = median_filter(angular, size=window, mode="nearest")
    hand_gap = []
    for side in ("Left", "Right"):
        ids = [
            i
            for i, name in enumerate(joint_names)
            if name.startswith(side + "Hand") and name != side + "Hand"
        ]
        if not ids:
            raise ValueError(f"Missing {side} finger samples")
        local = np.einsum("tji,thj->thi", matrix, motion.positions[:, ids] - p[:, None])
        gaps = np.linalg.norm(
            np.maximum(np.maximum(low - local, local - high), 0), axis=-1
        )
        nearest = np.sort(gaps, axis=1)[:, : min(4, len(ids))].mean(axis=1)
        hand_gap.append(median_filter(nearest, size=window, mode="nearest"))
    hand_gap = np.stack(hand_gap, axis=1)
    return dict(
        bottom_m=bottom,
        speed_m_s=speed,
        angular_speed_rad_s=angular,
        hand_gap_m=hand_gap,
        object_positions=p,
        object_wxyz=obj["wxyz"],
    )


def propose_intervals(features, fps, cfg):
    """Bound each lift with rest windows; retain ambiguous/partial trials explicitly."""
    count = len(features["bottom_m"])
    if count < 2 or fps <= 0:
        raise ValueError("Need at least two frames and a positive source rate")
    floor = (features["bottom_m"] <= cfg.floor_height_tolerance_m) & (
        features["bottom_m"] >= -cfg.maximum_floor_penetration_m
    )
    stable = (
        floor
        & (features["speed_m_s"] <= cfg.stable_speed_m_s)
        & (features["angular_speed_rad_s"] <= cfg.stable_angular_speed_rad_s)
    )
    clear = np.all(features["hand_gap_m"] > cfg.hands_clear_distance_m, axis=1)
    min_rest = max(1, int(np.ceil(cfg.minimum_rest_s * fps)))
    lifted = features["bottom_m"] > cfg.lift_height_m
    # Bridge short height-threshold flicker without inventing full lift events.
    for a, b in runs(~lifted):
        if a > 0 and b < count and (b - a) / fps <= cfg.smoothing_s:
            lifted[a:b] = True
    events = runs(lifted, max(1, int(np.ceil(cfg.minimum_lift_s * fps))))
    trials = []
    for index, (lift_start, lift_stop) in enumerate(events):
        left = events[index - 1][1] if index else 0
        right = events[index + 1][0] if index + 1 < len(events) else count
        before = [
            (a + left, b + left)
            for a, b in runs((stable & clear)[left:lift_start], min_rest)
        ]
        after = [
            (a + lift_stop, b + lift_stop)
            for a, b in runs((stable & clear)[lift_stop:right], min_rest)
        ]
        flags = []
        if before:
            a, b = before[-1]
            start = max(a, b - max(1, round(cfg.approach_padding_s * fps)))
        else:
            start = max(left, lift_start - max(1, round(cfg.approach_padding_s * fps)))
            flags.append("no_clear_released_start")
        if after:
            a, b = after[0]
            stop = min(b, a + max(min_rest, round(cfg.settle_padding_s * fps)))
        else:
            rests = [
                (a + lift_stop, b + lift_stop)
                for a, b in runs(stable[lift_stop:right], min_rest)
            ]
            if rests:
                a, b = rests[0]
                stop = min(b, a + max(min_rest, round(cfg.settle_padding_s * fps)))
                flags.append("release_unconfirmed")
            else:
                stop = min(right, lift_stop + max(1, round(cfg.settle_padding_s * fps)))
                flags.extend(["release_unconfirmed", "no_stable_supported_end"])
        if lift_start == 0:
            flags.append("starts_already_lifted")
        if lift_stop == count:
            flags.append("ends_still_lifted")
        trials.append(
            dict(
                start_frame=start,
                stop_frame=stop,
                lift_start_frame=lift_start,
                lift_stop_frame=lift_stop,
                start_released_candidate=bool(before),
                end_released_candidate=bool(after),
                complete_cycle_candidate=bool(before and after),
                flags=flags,
            )
        )
    # Long rests can supply both an end tail and the next approach prefix. Keep
    # trials disjoint without silently cutting the lift interval or overlap data.
    for previous, current in zip(trials, trials[1:]):
        if previous["stop_frame"] > current["start_frame"]:
            boundary = (previous["stop_frame"] + current["start_frame"]) // 2
            previous["stop_frame"] = current["start_frame"] = boundary
            previous["flags"].append("shared_rest_boundary")
            current["flags"].append("shared_rest_boundary")
    return trials


def validate_manifest(manifest, dataset_root):
    """Validate source identity, half-open bounds, outcomes, and disjoint trial ranges."""
    if (
        manifest.get("schema_version") != 1
        or manifest.get("frame_convention") != "zero_based_half_open"
    ):
        raise ValueError("Unsupported trial manifest schema/frame convention")
    seen, previous = set(), {}
    for trial in sorted(
        manifest["trials"], key=lambda t: (t["motion_id"], t["start_frame"])
    ):
        key = trial["trial_id"]
        if key in seen:
            raise ValueError(f"Duplicate trial: {key}")
        seen.add(key)
        source = manifest["sources"][trial["motion_id"]]
        a, b = trial["start_frame"], trial["stop_frame"]
        if (
            not isinstance(a, int)
            or not isinstance(b, int)
            or not 0 <= a < b <= source["frame_count"]
        ):
            raise ValueError(f"Invalid source bounds: {key}")
        if a < previous.get(trial["motion_id"], 0):
            raise ValueError(f"Overlapping trials: {key}")
        previous[trial["motion_id"]] = b
        if trial["human_outcome"] not in (
            "unknown",
            "success",
            "failure",
            "incomplete",
        ):
            raise ValueError(f"Invalid human outcome: {key}")
        if trial["review_status"] not in ("unreviewed", "reviewed"):
            raise ValueError(f"Invalid review status: {key}")
        if trial["human_outcome"] != "unknown" and trial["review_status"] != "reviewed":
            raise ValueError(f"Human outcome needs visual review: {key}")
        package = (dataset_root / source["package"]).resolve()
        if not package.is_relative_to(dataset_root.resolve()):
            raise ValueError("Source package escapes dataset root")
        metadata = json.loads((package / "metadata.json").read_text())
        if (
            metadata["motion_id"] != trial["motion_id"]
            or metadata["frame_count"] != source["frame_count"]
        ):
            raise ValueError(f"Source metadata mismatch: {key}")
        if trial["object_id"] not in {obj["object_id"] for obj in metadata["objects"]}:
            raise ValueError(f"Source object mismatch: {key}")


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.data_root.resolve()
    cfg = args.thresholds
    if any(value <= 0 for value in asdict(cfg).values()):
        raise ValueError("All thresholds and time windows must be positive")
    manifest = dict(
        schema_version=1,
        frame_convention="zero_based_half_open",
        method="floor_box_lift_proposals_v1",
        thresholds=asdict(cfg),
        source_data_root=str(root),
        sources={},
        trials=[],
        notes="Candidate source segments; no reviewed human success or robot execution success implied.",
    )
    for motion_id in args.motion_ids:
        packages = list((root / "data").glob(f"*/*/{motion_id}/metadata.json"))
        if len(packages) != 1:
            raise ValueError(f"Expected one source package for {motion_id}")
        package = packages[0].parent
        motion = load_motion(package, root)
        if (
            len(motion.objects) != 1
            or motion.metadata["objects"][0]["object_category"] != "box"
        ):
            raise ValueError("This pilot supports one tracked box per recording")
        nodes, count, _, _, _ = parse_bvh(package / "motion_actor.bvh")
        features = source_features(motion, [n.name for n in nodes], cfg)
        intervals = propose_intervals(features, 1 / motion.frame_time, cfg)
        object_info = motion.metadata["objects"][0]
        paths = [
            package / "motion_actor.bvh",
            package / "metadata.json",
            package / object_info["trajectory_path"],
            root / object_info["mesh_path"],
        ]
        manifest["sources"][motion_id] = dict(
            package=str(package.relative_to(root)),
            frame_count=count,
            frame_time_s=motion.frame_time,
            fps=1 / motion.frame_time,
            actor_id=motion.metadata["actor_id"],
            object_id=object_info["object_id"],
            text_annotation=motion.metadata["text_annotation"],
            sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            split_group=motion_id.removesuffix("__mirror"),
            lifted_event_count=len(intervals),
            segmentation_status="proposals_need_review"
            if intervals
            else "no_lift_events_detected",
        )
        for i, interval in enumerate(intervals, 1):
            a, b = interval["start_frame"], interval["stop_frame"]
            p = features["object_positions"][a:b]
            displacement = float(np.linalg.norm(p[-1, [0, 2]] - p[0, [0, 2]]))
            if displacement < 0.10:
                interval["flags"].append("returns_near_start_goal_alone_is_ambiguous")
            trial = dict(
                trial_id=f"{motion_id}:trial{i:03}",
                motion_id=motion_id,
                object_id=object_info["object_id"],
                **interval,
                start_time_s=a * motion.frame_time,
                stop_time_s=b * motion.frame_time,
                human_outcome="unknown",
                review_status="unreviewed",
                review_notes="",
                robot_execution=dict(status="not_run", success=None, rollout_uri=None),
                start_object_pose_world_y_up=dict(
                    position=p[0].tolist(), wxyz=motion.objects[0]["wxyz"][a].tolist()
                ),
                end_object_pose_world_y_up=dict(
                    position=p[-1].tolist(),
                    wxyz=motion.objects[0]["wxyz"][b - 1].tolist(),
                ),
                evidence=dict(
                    initial_final_horizontal_displacement_m=displacement,
                    maximum_horizontal_excursion_m=float(
                        np.linalg.norm(p[:, [0, 2]] - p[0, [0, 2]], axis=1).max()
                    ),
                    maximum_box_bottom_height_m=float(features["bottom_m"][a:b].max()),
                    start_end_hand_bbox_gaps_m=features["hand_gap_m"][
                        [a, b - 1]
                    ].tolist(),
                    end_box_speed_m_s=float(features["speed_m_s"][b - 1]),
                ),
            )
            manifest["trials"].append(trial)
        print(f"{motion_id}: {len(intervals)} proposed trials", flush=True)
    validate_manifest(manifest, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Saved proposals for visual review: {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
