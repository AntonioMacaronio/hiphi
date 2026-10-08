"""Check trial boundaries, partial attempts, and annotation validation.

Example, from hiphi:
    source scripts/project_env.sh
    PYTHONPATH=scripts "$HIPHI_PYTHON" -m pytest -q tests/test_box_trials.py
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from segment_box_trials import (
    Thresholds,
    propose_intervals,
    source_features,
    validate_manifest,
)


def features(count, lifts):
    bottom = np.zeros(count)
    for a, b in lifts:
        bottom[a:b] = 0.4
    return dict(
        bottom_m=bottom,
        speed_m_s=np.zeros(count),
        angular_speed_rad_s=np.zeros(count),
        hand_gap_m=np.full((count, 2), 0.3),
    )


def test_complete_trials_keep_the_lift_and_settled_tail_without_overlap():
    data = features(140, [(30, 50), (90, 110)])
    trials = propose_intervals(data, 10, Thresholds())
    assert len(trials) == 2
    for trial, (a, b) in zip(trials, [(30, 50), (90, 110)]):
        assert trial["start_frame"] < a < b < trial["stop_frame"]
        assert trial["complete_cycle_candidate"]
    assert trials[0]["stop_frame"] <= trials[1]["start_frame"]


def test_an_initially_held_box_is_not_a_complete_pickup():
    trial = propose_intervals(features(60, [(0, 25)]), 10, Thresholds())[0]
    assert trial["start_frame"] == 0
    assert not trial["complete_cycle_candidate"]
    assert "starts_already_lifted" in trial["flags"]


def test_an_unfinished_lift_is_retained_without_success():
    trial = propose_intervals(features(60, [(30, 60)]), 10, Thresholds())[0]
    assert trial["stop_frame"] == 60
    assert not trial["complete_cycle_candidate"]
    assert "ends_still_lifted" in trial["flags"]


def test_a_floor_pause_with_hands_near_is_not_a_release():
    data = features(140, [(30, 50), (90, 110)])
    data["hand_gap_m"][50:90] = 0.01
    trials = propose_intervals(data, 10, Thresholds())
    assert "release_unconfirmed" in trials[0]["flags"]
    assert "no_clear_released_start" in trials[1]["flags"]
    assert all(not t["complete_cycle_candidate"] for t in trials)


def test_a_short_shared_rest_still_gives_disjoint_trials():
    data = features(100, [(25, 45), (49, 70)])
    trials = propose_intervals(data, 10, Thresholds())
    assert trials[0]["stop_frame"] == trials[1]["start_frame"]
    assert 45 <= trials[0]["stop_frame"] <= 49


def test_no_lift_does_not_invent_a_trial():
    assert propose_intervals(features(100, []), 10, Thresholds()) == []


def test_hand_separation_is_invariant_to_common_object_rotation_and_translation():
    names = [side + f"HandFinger{j}" for side in ("Left", "Right") for j in range(4)]
    vertices = np.array(
        [[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)]
    )
    points = np.array([[-0.3, 0.0, 0.0]] * 4 + [[0.3, 0.0, 0.0]] * 4)
    outputs = []
    for rotation, offset in [
        (Rotation.identity(), np.zeros(3)),
        (Rotation.from_euler("y", 0.8), np.array([1.2, 0.0, -0.9])),
    ]:
        motion = SimpleNamespace(
            frame_time=0.01,
            positions=np.tile(rotation.apply(points) + offset, (20, 1, 1)),
            objects=[
                dict(
                    vertices=vertices,
                    positions=np.tile(offset, (20, 1)),
                    wxyz=np.tile(rotation.as_quat()[[3, 0, 1, 2]], (20, 1)),
                )
            ],
        )
        outputs.append(source_features(motion, names, Thresholds())["hand_gap_m"])
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)
    np.testing.assert_allclose(outputs[0], 0.2, atol=1e-12)


def test_success_label_requires_review_and_valid_source_bounds(tmp_path):
    package = tmp_path / "data/example"
    package.mkdir(parents=True)
    (package / "metadata.json").write_text(
        json.dumps(
            dict(
                motion_id="example",
                frame_count=100,
                objects=[dict(object_id="box")],
            )
        )
    )
    trial = dict(
        trial_id="example:trial001",
        motion_id="example",
        object_id="box",
        start_frame=0,
        stop_frame=100,
        human_outcome="success",
        review_status="unreviewed",
    )
    manifest = dict(
        schema_version=1,
        frame_convention="zero_based_half_open",
        trials=[trial],
        sources={"example": dict(package="data/example", frame_count=100)},
    )
    with pytest.raises(ValueError, match="needs visual review"):
        validate_manifest(manifest, tmp_path)
    trial["review_status"] = "reviewed"
    validate_manifest(manifest, tmp_path)
    trial["stop_frame"] = 101
    with pytest.raises(ValueError, match="Invalid source bounds"):
        validate_manifest(manifest, tmp_path)
