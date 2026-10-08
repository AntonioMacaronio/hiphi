# HiPHI box-trial segmentation — September 9, 2026

A source recording can contain several attempts. For the first pick/carry/place
experiment, propose a trial that starts with the box supported and the hands clear,
includes approach and one lift/transport/lower event, and ends after release and a
stable supported pause. Preserve the original source frame IDs and timing.

There are three separate outcomes: human demonstration completion, retargeting
quality, and robot execution success. A completed human demonstration does not
establish either of the latter two. In particular, passing through the last motion
frame is not the robot success criterion.

## User selection review — September 12, 2026

The user visually reviewed the proposals and accepted **all five carry trials and
placing trial 002** for the MVP. Placing trials 001 and 003 are excluded: the user
reported unwanted motion in the recording that passes the geometric segmentation
criteria. No exact offending frames or revised boundaries were supplied.

- All eight decisions and the user's review statement:
  `/data/sky/projects/hiphi/runs/trials_review_2026-09-12/trials.json`.
- The six accepted candidates, suitable for opening in the existing trial browser:
  `/data/sky/projects/hiphi/runs/trials_review_2026-09-12/accepted_trials.json`.

These are new manifests derived from the immutable v1 proposals. Each trial has
`review_status: "reviewed"` and a separate `selection_status` of `accepted` or
`excluded`. This review covers visual motion suitability. `human_outcome` remains
`unknown`; robot execution remains `not_run`. Exclusion does not assert human task
failure. All source frame ranges, timing, hashes, geometric flags, and evidence are
preserved. Placing 002 retains its `release_unconfirmed` flag and still needs a
specific completion/release check before being treated as a complete task example.

To view only the accepted candidates, from the `hiphi` checkout:

```bash
source scripts/project_env.sh
"$HIPHI_PYTHON" scripts/view_stage1_viser.py \
  --trial-manifest /data/sky/projects/hiphi/runs/trials_review_2026-09-12/accepted_trials.json \
  --port 8671
```

The current viewer lists every trial in the supplied manifest; it does not filter
on `selection_status`. Use `accepted_trials.json` for the six-candidate selection,
or the new `trials.json` to retain all eight for comparison. Keep this selection
review separate from contact annotation and subsequent physical execution checks.

## First proposals

The manifest is `/data/sky/projects/hiphi/runs/trials_v1/trials.json`. Source data is
unchanged; the manifest references the raw BVHs, metadata, object tracks, and meshes
with hashes. All ranges use **zero-based original BVH frames, stop exclusive**.

| Source | Trial | Frames [start, stop) | Time, seconds | Geometric boundary assessment |
| --- | --- | --- | --- | --- |
| Bringing-carry_0010 | 001 | [790, 1471) | 8.78–16.34 | Released start/end candidates |
| Bringing-carry_0010 | 002 | [1717, 2329) | 19.08–25.88 | Released start/end candidates |
| Bringing-carry_0010 | 003 | [2579, 3142) | 28.66–34.91 | Released start/end candidates |
| Bringing-carry_0010 | 004 | [3418, 4098) | 37.98–45.53 | Released start/end candidates |
| Bringing-carry_0010 | 005 | [4383, 5039) | 48.70–55.99 | Released start/end candidates |
| Placing-put_0003 | 001 | [464, 1924) | 5.16–21.38 | Released start/end candidates; long pause before clear release |
| Placing-put_0003 | 002 | [2036, 2850) | 22.62–31.67 | Release unconfirmed |
| Placing-put_0003 | 003 | [3555, 5219) | 39.50–57.99 | Clear released start unconfirmed; delayed release |

Every original v1 proposal has `human_outcome: "unknown"`, `review_status: "unreviewed"`, and
`robot_execution: {"status": "not_run", "success": null, "rollout_uri": null}`.
`complete_cycle_candidate` is a geometric screening result, not a success label.
The carry trials displace the box by approximately 2.1–2.2 m. All three placing
trials return it within 10 cm of the start: a final-position goal alone could reward
doing nothing. Preserve their lift/release intent, or use them as motion-controller
data rather than treating them as nontrivial goal-only placement demonstrations.

## Generate and inspect

From the `hiphi` checkout:

```bash
source scripts/project_env.sh
"$HIPHI_PYTHON" scripts/segment_box_trials.py \
  --motion-ids Bringing-carry_0010 Placing-put_0003 \
  --output /data/sky/projects/hiphi/runs/trials_v2/trials.json

"$HIPHI_PYTHON" scripts/view_stage1_viser.py \
  --trial-manifest /data/sky/projects/hiphi/runs/trials_v1/trials.json --port 8671
```

Use a new output path for each proposal version. The viewer requests a share link
by default, displays at 90 FPS, and loops the selected trial's source range. Select
**Full recording** to inspect context. It shows boundary flags and the separate
human/robot outcome fields; it does not write annotations through the public link.

The proposal tool uses box height and linear/angular speed, plus finger separation
from an object-local bounding-box proxy. A lift must persist for 0.4 seconds; a
supported stable rest must persist for 0.3 seconds. It keeps up to 1.25 seconds of
approach before hands become close and 0.5 seconds of a released rest at the end.
All thresholds are explicit Tyro options and recorded in the manifest. The phase
anchors mark height-threshold crossings, not exact contact times. This first method
assumes floor-level box lifts and is not suitable for pushing, sliding, table
placement, or motions without tracked objects.

Inspect the full motion around both boundaries and the entire intervening attempt.
Keep ambiguous or incomplete attempts flagged. Reviewed human outcomes may be
`success`, `failure`, or `incomplete`; a non-unknown outcome requires
`review_status: "reviewed"`. Preserve proposal versions and record review notes.
Boundary revisions require refreshing derived times, endpoint poses, and evidence
from the source; the v1 files remain immutable proposals, not a reviewed training set.

After accepting a range, export from the original recording, for example:

```bash
"$HIPHI_PYTHON" scripts/export_retarget_input.py \
  --motion-id Bringing-carry_0010 --start-frame 790 --stop-frame 1471 --stride 3 \
  --output /data/sky/projects/hiphi/runs/trials_v1/retarget_inputs/carry_trial001.npz
```

The five earlier FAR-Holosoma cycle outputs used slightly different hand-selected
boundaries; they are retained separately. Finalize segmentation before generating a
replacement robot reference set. Keep all trials/windows from a source recording,
including its mirror, in the same dataset split; use actor grouping when evaluating
transfer to new people. Robot success will be recorded from subsequent CF-Track
rollouts with freely simulated objects and complete release/stable placement checks.
