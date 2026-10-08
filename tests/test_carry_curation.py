"""Geometric eligibility and diversity checks for the rectangular carry pilot."""
from copy import deepcopy

import numpy as np
import trimesh

from curate_box_carries import cuboid_geometry, rejection_reasons, select_diverse, body_frame


def test_anatomical_forward_uses_up_cross_right():
    positions=np.array([[[-.1,1.,0.],[.1,1.,0.]]])
    heading=body_frame(positions,['LeftUpLeg','RightUpLeg'],0)
    np.testing.assert_allclose(heading[:,0],[0,0,-1])
    np.testing.assert_allclose(heading[:,1],[1,0,0])


def test_rotated_rectangular_prism_passes_and_cylinder_fails(tmp_path):
    box=trimesh.creation.box(extents=[30,40,60])
    box.apply_transform(trimesh.transformations.rotation_matrix(.43,[.2,.8,.4]))
    box.export(tmp_path/'box.obj')
    result=cuboid_geometry(tmp_path/'box.obj')
    assert result['is_rectangular_prism']
    np.testing.assert_allclose(sorted(result['dimensions_m']),[.3,.4,.6],atol=1e-7)
    trimesh.creation.cylinder(radius=20,height=60).export(tmp_path/'cylinder.obj')
    assert not cuboid_geometry(tmp_path/'cylinder.obj')['is_rectangular_prism']


def test_missing_box_face_fails(tmp_path):
    box=trimesh.creation.box(extents=[30,40,60])
    # An open cuboid has the same bounds and convex hull as a closed box.
    box.update_faces(np.arange(len(box.faces)) != 0)
    box.export(tmp_path/'open.obj')
    assert not cuboid_geometry(tmp_path/'open.obj')['is_rectangular_prism']


def measurements():
    return dict(longest_airborne_s=2.,start_stable_s=.5,end_stable_s=.5,
        hand_near_airborne_fraction=1.,pelvis_airborne_path_m=3.,airborne_path_m=3.,
        pickup_setdown_displacement_m=.2)


def test_return_to_start_carry_does_not_pass_endpoint_filter():
    trial=dict(start_frame=0,lift_start_frame=30,lift_stop_frame=100,stop_frame=150)
    geom=dict(is_rectangular_prism=True)
    m=measurements()
    assert 'transport_below_minimum' in rejection_reasons(m,geom,trial,'endpoint',1.)
    assert not rejection_reasons(m,geom,trial,'airborne-path',1.)
    m['pickup_setdown_displacement_m']=1.0
    assert not rejection_reasons(m,geom,trial,'endpoint',1.)
    m['end_stable_s']=0.
    assert 'no_stable_supported_end' in rejection_reasons(m,geom,trial,'endpoint',1.)


def test_diversity_balances_sizes_and_limits_repeated_source():
    candidates=[]
    for size in ('small','medium','large'):
        for source in range(3):
            for repeat in range(3):
                candidates.append(dict(trial_id=f'{size}_{source}_{repeat}',motion_id=f'{size}_{source}',
                    actor_id=f'actor_{source}',box_geometry=dict(size_group=size),
                    carry_metrics=dict(start_box_forward_right_m=[.3,repeat*.2],
                        start_box_relative_height_m=-.6,start_pelvis_height_m=1.,
                        carry_direction_body_deg=source*90,max_box_bottom_height_m=.5,
                        end_released_stable_s=.5)))
    selected=select_diverse(candidates,10,2)
    from collections import Counter
    assert len(selected)==10
    assert sorted(Counter(t['box_geometry']['size_group'] for t in selected).values())==[3,3,4]
    assert max(Counter(t['motion_id'] for t in selected).values())<=2
