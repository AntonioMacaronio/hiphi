#!/usr/bin/env python3
"""Minimal Viser playback of HiPHI BVH skeletons and synchronized objects.

Run after sourcing scripts/project_env.sh. Only the selected clip is loaded;
meshes are sent once per selection. Adjust display FPS with the viewer slider
(1–90 FPS, default 30); the separate Speed slider controls playback speed.

Examples (run from /home/sky/sky_workdir/heavyhumanoid/hiphi):
    source scripts/project_env.sh
    "$HIPHI_PYTHON" scripts/view_stage1_viser.py --port 8667
    "$HIPHI_PYTHON" scripts/view_stage1_viser.py --no-share
    "$HIPHI_PYTHON" scripts/view_stage1_viser.py --data-root /data/sky/projects/hiphi/datasets/stage1_v1/HiPHI

Open http://127.0.0.1:8667 (forward the port over SSH/your IDE for remote use).
A public Viser URL is requested by default; use --no-share for local access only.
Press Ctrl+C to stop.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import tyro
import viser

from build_frame_lu_bvh_previews import parse_bvh


@dataclass
class ViewerArgs:
    # Extracted HiPHI root containing data/ and object_meshes/.
    data_root: Path = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/hiphi")) / "datasets/stage1_v1/HiPHI"
    # Bind address; localhost works with SSH/IDE port forwarding.
    host: str = "127.0.0.1"
    # Requested HTTP port; Viser may choose another if occupied.
    port: int = 8667
    # Request a public Viser share URL; disable with --no-share.
    share: bool = True


@dataclass
class Motion:
    metadata: dict
    positions: np.ndarray
    edges: np.ndarray
    frame_time: float
    objects: list[dict]


def joint_positions(nodes, values):
    """Vectorized HiPHI FK in meters, preserving the release's Y-up frame.

    HiPHI per-joint translation channels REPLACE the hierarchy OFFSET.
    End sites have no channels and use OFFSET. Adding both doubles limb lengths.
    """
    frames = len(values)
    positions = np.empty((frames, len(nodes), 3), dtype=np.float32)
    rotations = np.empty((frames, len(nodes), 3, 3), dtype=np.float64)
    for index, node in enumerate(nodes):
        translation = np.tile(np.asarray(node.offset, dtype=float), (frames, 1))
        rotation = np.broadcast_to(np.eye(3), (frames, 3, 3))
        if node.channels:
            local = values[:, node.channel_index:node.channel_index + len(node.channels)]
            position_channels = [j for j, channel in enumerate(node.channels) if channel.endswith("position")]
            if position_channels:
                translation = np.zeros((frames, 3))
                for j in position_channels:
                    translation[:, "XYZ".index(node.channels[j][0])] = local[:, j]
            rotation_channels = [j for j, channel in enumerate(node.channels) if channel.endswith("rotation")]
            if rotation_channels:
                order = "".join(node.channels[j][0] for j in rotation_channels)
                rotation = Rotation.from_euler(order, local[:, rotation_channels], degrees=True).as_matrix()
        translation *= 0.01  # Human BVH and mesh vertices use cm; object tracks use m.
        if node.parent is None:
            positions[:, index] = translation
            rotations[:, index] = rotation
        else:
            parent_rotation = rotations[:, node.parent]
            positions[:, index] = positions[:, node.parent] + np.einsum("fij,fj->fi", parent_rotation, translation)
            rotations[:, index] = parent_rotation @ rotation
    return positions


def load_motion(package, dataset_root):
    metadata = json.loads((package / "metadata.json").read_text())
    nodes, frames, frame_time, channels, rows = parse_bvh(package / "motion_actor.bvh")
    values = np.asarray(list(rows), dtype=np.float64)
    if values.shape != (frames, channels) or not np.isfinite(values).all():
        raise ValueError("Invalid BVH frame/channel data")
    positions = joint_positions(nodes, values)
    # Recenter only the horizontal origin; apply exactly the same shift to objects.
    origin = positions[0, 0].copy()
    origin[1] = 0
    positions -= origin
    edges = np.asarray([(node.parent, i) for i, node in enumerate(nodes) if node.parent is not None])
    objects = []
    for obj in metadata["objects"]:
        with (package / obj["trajectory_path"]).open(newline="") as stream:
            tracks = list(csv.DictReader(stream))
        if len(tracks) != frames:
            raise ValueError("Object track and skeleton have different frame counts")
        translations = np.asarray([[float(r[k]) for k in ("px", "py", "pz")] for r in tracks]) - origin
        # Viser accepts WXYZ; the source trajectories store XYZW.
        quaternions = np.asarray([[float(r[k]) for k in ("qw", "qx", "qy", "qz")] for r in tracks])
        mesh = trimesh.load(dataset_root / obj["mesh_path"], force="mesh", process=False)
        objects.append({"id": obj["object_id"], "vertices": np.asarray(mesh.vertices, dtype=np.float32) * 0.01,
                        "faces": np.asarray(mesh.faces, dtype=np.uint32),
                        "positions": translations, "wxyz": quaternions})
    return Motion(metadata, positions, edges, frame_time, objects)


def discover(root):
    packages = sorted((root / "data").glob("*/*/*/metadata.json"))
    choices = {}
    for path in packages:
        metadata = json.loads(path.read_text())
        if not (path.parent / "motion_actor.bvh").is_file():
            continue
        label = metadata["motion_id"]
        if metadata["objects"]:
            label += " [" + ", ".join(sorted({o["object_category"] for o in metadata["objects"]})) + "]"
        choices[label] = path.parent
    return choices


def serve(args):
    root = args.data_root.expanduser().resolve()
    choices = discover(root)
    if not choices:
        raise SystemExit(f"No extracted HiPHI motion packages found under {root / 'data'}")
    server = viser.ViserServer(host=args.host, port=args.port, label="HiPHI · motion preview")
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/ground", plane="xz", width=20, height=20, cell_size=0.5,
                          cell_color=(210, 218, 225), section_color=(160, 174, 188))
    first = next((name for name in choices if name.startswith("Bringing-carry")), next(iter(choices)))
    selector = server.gui.add_dropdown("Motion", list(choices), initial_value=first)
    play = server.gui.add_checkbox("Play", initial_value=False)
    frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    speed = server.gui.add_slider("Speed", min=0.25, max=2.0, step=0.25, initial_value=1.0)
    fps = server.gui.add_slider("FPS", min=1, max=90, step=1, initial_value=30,
                                hint="Display update rate. Use Speed to change playback speed.")
    reset = server.gui.add_button("Reset camera")
    info = server.gui.add_markdown("Loading motion…")
    clock = server.gui.add_markdown("")
    center = np.array([0.0, 0.9, 0.0])

    def set_camera(client):
        client.camera.up_direction = (0.0, 1.0, 0.0)
        client.camera.look_at = center
        client.camera.position = center + np.array([3.0, 1.4, 3.0])

    server.on_client_connect(set_camera)

    @reset.on_click
    def _(_event):
        for client in server.get_clients().values():
            set_camera(client)

    if args.share:
        print("Share URL:", server.request_share_url(), flush=True)
    print(f"Ready: {len(choices)} clips from {root}. PID: {os.getpid()}", flush=True)
    print("Use the URL/port in Viser's banner; a busy requested port may be changed by Viser.", flush=True)
    current_name, motion, handles = None, None, []
    playhead, last_frame = 0.0, -1
    previous_time = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            elapsed = now - previous_time
            previous_time = now
            if selector.value != current_name:
                current_name = selector.value
                play.value = False
                info.content = "Loading motion…"
                try:
                    loaded = load_motion(choices[current_name], root)
                except Exception as error:
                    motion = None
                    for handle in handles:
                        handle.remove()
                    handles = []
                    info.content = f"Could not load motion: {error}"
                    time.sleep(0.1)
                    continue
                motion = loaded
                with server.atomic():
                    for handle in handles:
                        handle.remove()
                    bones = server.scene.add_line_segments("/human/bones", motion.positions[0][motion.edges],
                                                           colors=(60, 135, 225), line_width=4)
                    joints = server.scene.add_point_cloud("/human/joints", motion.positions[0],
                                                         colors=(35, 80, 145), point_size=0.012, point_shape="circle")
                    handles = [bones, joints]
                    object_handles = []
                    for obj in motion.objects:
                        handle = server.scene.add_mesh_simple("/objects/" + obj["id"], obj["vertices"], obj["faces"],
                                                               color=(225, 155, 70), side="double",
                                                               position=obj["positions"][0], wxyz=obj["wxyz"][0])
                        object_handles.append(handle)
                        handles.append(handle)
                    frame.value = 0
                    frame.max = len(motion.positions) - 1
                playhead, last_frame = 0.0, -1
                center = motion.positions[0].mean(axis=0)
                for client in server.get_clients().values():
                    set_camera(client)
                metadata = motion.metadata
                info.content = (f"**{metadata['motion_id']}** · {metadata['actor_id']}\n\n"
                                f"{metadata['text_annotation']}\n\n"
                                f"{len(motion.positions) * motion.frame_time:.1f} s · "
                                f"{1/motion.frame_time:.1f} Hz source · {len(motion.objects)} tracked objects")
                previous_time = time.monotonic()
                elapsed = 0
                print("Loaded:", metadata["motion_id"], flush=True)
            if motion is not None:
                if frame.value != last_frame:
                    playhead = float(frame.value)
                if play.value:
                    playhead = (playhead + elapsed * speed.value / motion.frame_time) % len(motion.positions)
                index = int(playhead)
                if index != last_frame:
                    with server.atomic():
                        bones.points = motion.positions[index][motion.edges]
                        joints.points = motion.positions[index]
                        for obj, handle in zip(motion.objects, object_handles):
                            handle.position = obj["positions"][index]
                            handle.wxyz = obj["wxyz"][index]
                        frame.value = index
                    last_frame = index
                    clock.content = f"{index * motion.frame_time:.2f} / {(len(motion.positions)-1) * motion.frame_time:.2f} s"
            time.sleep(max(0.001, 1 / fps.value - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    args = tyro.cli(ViewerArgs, description=__doc__)
    serve(args)
