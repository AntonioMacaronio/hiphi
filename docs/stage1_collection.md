# Stage 1: a small HiPHI pilot for humanoid manipulation

The `hiphi` fork owns dataset acquisition, raw-data inspection, and the Viser viewer.
`FAR-Holosoma` owns robot retargeting, simulation, and policy training. Keep the repositories
as siblings; the simulator consumes data paths rather than importing the entire HiPHI website.

For the complete release, see the [raw dataset and S3 guide](../../FAR-Holosoma/projectpivot-docs/hiphi-raw-dataset.md).
It documents the separate full archive mirror and selective/full extraction commands;
the 50-clip pilot below remains available at its original path.

```text
/home/sky/sky_workdir/heavyhumanoid/
├── hiphi/                              # AntonioMacaronio fork; source only
│   ├── scripts/project_env.sh
│   ├── scripts/collect_stage1.py
│   ├── scripts/view_stage1_viser.py
│   └── docs/stage1_collection.md
└── FAR-Holosoma/                        # retargeting, simulation, training

/data/sky/projects/hiphi/
├── datasets/HiPHI/                      # pinned source metadata, archives, meshes
├── datasets/stage1_v1/HiPHI/             # selected originals in upstream package layout
│   ├── data/<frame>/<lu>/<motion_id>/
│   ├── object_meshes/
│   ├── metadata/hiphi_metadata.csv       # selected clips only
│   └── selection_manifest.json
└── runs/stage1_v1/
    ├── plan.json
    ├── collection.log
    ├── collector_identity.json
    └── validation.json                  # written after successful validation

/data/sky/projects/FAR-Holosoma/
└── datasets/hiphi_stage1_g1/             # proposed future robot data; not created yet
```

The checkout is sparse: `docs`, `scripts`, `viewer`, and top-level source files are present.
The website's videos and meshes are not checked out. `origin` is the AntonioMacaronio fork;
`upstream` is `noitom-robotics/hiphi`. No commits or pushes are needed to collect data.

## Pilot selection

The frozen plan selects **50 original clips, 51.65 minutes, and 32 actors**, with at least
eight clips covering each of reaching, lifting, carrying, lowering/placing, pushing,
walking, and posture changes. Coverage groups overlap; six clips cover two groups.
Manipulation candidates use tracked boxes. Reaching, walking, and the selected bends/squats
are body-only motions. Lowering/placing coverage is inferred from clip descriptions
mentioning lowering, putting, setting, or placing on the ground/floor. Review those phases
visually before treating them as release or contact labels.

Collection completed on September 5, 2026. All 50 packages passed the raw-data integrity
and synchronization checks, including 26 object-interaction clips and five shared meshes.
The extracted pilot occupies **1.10 GiB**. `runs/stage1_v1/validation.json` contains the
per-file hashes and checks; it does not certify physical execution.

The selection minimizes archive count, then compressed size, and favors actor diversity
when choosing clips. It is a pilot for inspection and retargeting, not a representative
benchmark or a physically validated training set. Original/mirrored duplicates are excluded.
No train/test split has been declared; establish performer and source-clip grouping before
creating training windows, so related windows and mirrors cannot cross splits.

Source: `noitomrobotics/HiPHI`, revision `b50ee4bcff519365503fec6146f8aca93f10eb49`.
Archives 1, 6, and 26 total **19.76 GiB compressed**. Selected extraction is bounded by
**2.20 GiB** using the original/mirror pair sizes (a conservative bound for originals only).
The planner reserves another 1 GiB for meshes/metadata and requires 10 GiB additional free
space. Archives are retained for repeatable extraction; the collector does not delete them.

## Collect or resume

```bash
cd /home/sky/sky_workdir/heavyhumanoid/hiphi
source scripts/project_env.sh
df -hT / /data
df -i / /data

# This node's inherited HF_TOKEN belongs to FAR-Amazon. These child processes
# use the already approved AntonioMacaronio saved login instead.
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "$HIPHI_PYTHON" scripts/collect_stage1.py plan
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "$HIPHI_PYTHON" -u scripts/collect_stage1.py collect
```

Run one collector per subset. Check the recorded PID and `collection.log` before resuming
an interrupted run. Downloads use the pinned revision and Hugging Face's local cache;
archives are checked against their published SHA-256 hashes. Extraction only writes the
selected packages, rejects unsafe members, and verifies existing extracted files on resume.
All large data and temporary files remain on `/data`.

Validation checks BVH channel/frame counts and finite values, package identities,
object-track synchronization and quaternion norms, and required mesh presence. It records
file hashes. It does not establish correct contacts, robot feasibility, or successful placement.

```bash
"$HIPHI_PYTHON" scripts/collect_stage1.py validate
```

## Viser playback

```bash
cd /home/sky/sky_workdir/heavyhumanoid/hiphi
source scripts/project_env.sh
"$HIPHI_PYTHON" scripts/view_stage1_viser.py --port 8667
```

Open the public share URL printed at startup; these links expire. For local access,
use `--no-share` and open `http://127.0.0.1:8667`, forwarding the port with your IDE/SSH
if needed. Read Viser's banner for the actual port if another viewer already occupies it.
The viewer offers a motion selector, play/pause, a source-frame slider, playback speed,
an FPS slider (1–90, default 90), and camera reset. The blue skeleton and orange tracked
meshes share the same coordinate frame. Only one clip is loaded at a time. FPS changes
the display update rate immediately; Speed controls how fast the motion plays.
No GPU or package installation is needed in this node's `ei_world_model` environment.
Both new Python scripts use Tyro and include runnable examples in their top docstrings.

The viewer requests a public Viser share URL by default; use `--no-share` for local
access only. Other flags include `--data-root <extracted-HiPHI-root>` and `--host 0.0.0.0`.
Press Ctrl+C in the viewer's terminal to stop it; do not use broad Python process
termination commands.

These BVHs use six channels on every joint. Translation channels replace `OFFSET`;
end sites use their offsets. Skeleton/OBJ positions are centimeters, object tracks are
meters, the shared world is Y-up, and source object quaternions are XYZW. The viewer
handles these conversions and sends WXYZ quaternions to Viser. It displays human motion,
not a robot-retargeted simulation.

## Next step

Review the [first eight trial proposals](trial_segmentation.md): five carry trials
and three placing trials from the two recordings already retargeted in FAR-Holosoma.
The trial viewer loops original frame ranges and shows ambiguous boundaries. Human
outcomes remain unreviewed and robot success remains unset. Finalize the ranges
before retargeting a new robot reference set and testing it through CF-Track.
