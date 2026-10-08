#!/usr/bin/env python3
"""Filter geometric HiPHI trials and select a diverse rectangular-box carry pilot.

Source scripts/project_env.sh first. All generated data belongs under /data.
This measures source geometry, not forces or robot execution success.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation
import trimesh
import tyro

from build_frame_lu_bvh_previews import parse_bvh
from segment_box_trials import Thresholds, main as segment_main, Args as SegmentArgs
from segment_box_trials import source_features, runs, validate_manifest
from view_stage1_viser import load_motion


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def cuboid_geometry(path, tolerance_m=0.002):
    """Require six filled planar faces of an oriented rectangular prism.

    OBJ vertices are cm. This handles rotated local mesh axes and duplicated
    face vertices. Mesh category/bounding dimensions alone are insufficient.
    """
    mesh = trimesh.load(path, force='mesh', process=False)
    transform, extents_cm = trimesh.bounds.oriented_bounds(mesh)
    vertices = trimesh.transform_points(mesh.vertices, transform) * 0.01
    extents = extents_cm * 0.01
    triangles = vertices[mesh.faces]
    # Each triangle must lie on one of the six planes, not just have vertices
    # somewhere on the surface (which could hide a diagonal interior face).
    residual = np.concatenate([
        np.abs(triangles + extents / 2).max(axis=1),
        np.abs(triangles - extents / 2).max(axis=1)], axis=1)
    face_id = residual.argmin(axis=1)
    error = residual.min(axis=1)
    areas = trimesh.triangles.area(triangles)
    face_areas = np.bincount(face_id, weights=areas, minlength=6)
    expected = np.tile(np.prod(extents) / extents, 2)
    area_ratio = face_areas / expected
    hull_ratio = mesh.convex_hull.volume * 1e-6 / np.prod(extents)
    valid = bool(np.all(extents > 0.01) and error.max() <= tolerance_m
                 and np.all(np.abs(area_ratio - 1) <= 0.03)
                 and abs(hull_ratio - 1) <= 0.03)
    return dict(is_rectangular_prism=valid, dimensions_m=extents.tolist(),
                size_group='x'.join(f'{d:.3f}' for d in sorted(extents)),
                tolerance_m=tolerance_m, max_triangle_plane_error_m=float(error.max()),
                six_face_area_ratios=area_ratio.tolist(), hull_to_cuboid_volume_ratio=float(hull_ratio),
                mesh_path=str(Path(path).resolve()), mesh_sha256=sha256(path))


def longest(mask, dt):
    return max((b-a for a,b in runs(mask)), default=0) * dt


def body_frame(positions, names, frame):
    """Heading from anatomical right and world up; +forward = up cross right."""
    left = positions[frame, names.index('LeftUpLeg')]
    right = positions[frame, names.index('RightUpLeg')]
    lateral = right - left
    lateral[1] = 0
    lateral /= np.linalg.norm(lateral)
    forward = np.cross([0., 1., 0.], lateral)
    return np.stack([forward, lateral], axis=1)


def measure_trial(motion, names, features, trial, cfg):
    a,b=trial['start_frame'],trial['stop_frame']
    la,lb=trial['lift_start_frame'],trial['lift_stop_frame']
    dt=motion.frame_time
    obj=motion.objects[0]
    center_local=(obj['vertices'].min(0)+obj['vertices'].max(0))/2
    p=features['object_positions']+Rotation.from_quat(obj['wxyz'][:,[1,2,3,0]]).apply(center_local)
    # A 0.1 s median suppresses marker jitter before integrating path length.
    smooth=median_filter(p, size=(max(1,round(.1/dt))|1,1), mode='nearest')
    bottom=features['bottom_m']
    airborne=bottom > cfg.lift_height_m
    edges=airborne[la:lb-1] & airborne[la+1:lb]
    path=float(np.linalg.norm(np.diff(smooth[la:lb,[0,2]],axis=0),axis=1)[edges].sum())
    # Tighten the legacy proposal's -8/+6 cm floor band for final acceptance.
    # In particular, a stationary box held 5 cm above the floor is not set down.
    rest=np.abs(bottom)<=.03
    rest &= features['speed_m_s'] <= cfg.stable_speed_m_s
    rest &= features['angular_speed_rad_s'] <= cfg.stable_angular_speed_rad_s
    clear=np.all(features['hand_gap_m'] > cfg.hands_clear_distance_m,axis=1)
    near=np.min(features['hand_gap_m'],axis=1)<=cfg.hands_clear_distance_m
    min_rest=int(np.ceil(cfg.minimum_rest_s/dt))
    before=runs(rest[a:la],min_rest); after=runs(rest[lb:b],min_rest)
    pickup=(a+before[-1][0],a+before[-1][1]) if before else (a,a+1)
    setdown=(lb+after[0][0],lb+after[0][1]) if after else (b-1,b)
    pickup_center=np.median(p[slice(*pickup)],axis=0)
    setdown_center=np.median(p[slice(*setdown)],axis=0)
    heading=body_frame(motion.positions,names,a)
    pelvis=motion.positions[:,names.index('Hips')]
    offset=p[a]-pelvis[a]
    relative=offset@heading
    direction=(smooth[lb-1]-smooth[la])@heading
    # Height/pose at crop start and relative object position describe meaningful
    # starting states independently of arbitrary global recording translation.
    return dict(duration_s=(b-a)*dt, endpoint_displacement_m=float(np.linalg.norm(p[b-1,[0,2]]-p[a,[0,2]])),
        pickup_setdown_displacement_m=float(np.linalg.norm(setdown_center[[0,2]]-pickup_center[[0,2]])),
        pickup_supported_frames=list(pickup),setdown_supported_frames=list(setdown),
        pickup_center_world_y_up=pickup_center.tolist(),setdown_center_world_y_up=setdown_center.tolist(),
        airborne_path_m=path, airborne_net_displacement_m=float(np.linalg.norm(smooth[lb-1,[0,2]]-smooth[la,[0,2]])),
        longest_airborne_s=longest(airborne[la:lb],dt), max_box_bottom_height_m=float(bottom[a:b].max()),
        start_stable_s=longest(rest[a:la],dt), end_stable_s=longest(rest[lb:b],dt),
        start_released_stable_s=longest((rest&clear)[a:la],dt), end_released_stable_s=longest((rest&clear)[lb:b],dt),
        hand_near_airborne_fraction=float(near[la:lb][airborne[la:lb]].mean()),
        pelvis_airborne_path_m=float(np.linalg.norm(np.diff(pelvis[la:lb,[0,2]],axis=0),axis=1)[edges].sum()),
        start_box_forward_right_m=relative.tolist(), start_box_relative_height_m=float(offset[1]),
        start_pelvis_height_m=float(pelvis[a,1]), start_box_bottom_m=float(bottom[a]),
        carry_direction_body_deg=float(np.degrees(np.arctan2(direction[1],direction[0]))),
        start_box_position_world_y_up=p[a].tolist(), end_box_position_world_y_up=p[b-1].tolist())


def rejection_reasons(metrics, geometry, trial, mode, distance):
    reasons=[]
    if not geometry['is_rectangular_prism']: reasons.append('not_rectangular_prism')
    if metrics['longest_airborne_s'] < .4: reasons.append('no_sustained_lift')
    if metrics['start_stable_s'] < .3: reasons.append('no_stable_supported_start')
    if metrics['end_stable_s'] < .3: reasons.append('no_stable_supported_end')
    if metrics['hand_near_airborne_fraction'] < .8: reasons.append('hand_separation_during_transport')
    if metrics['pelvis_airborne_path_m'] < distance: reasons.append('insufficient_human_locomotion')
    field='airborne_path_m' if mode=='airborne-path' else 'pickup_setdown_displacement_m'
    if metrics[field] < distance: reasons.append('transport_below_minimum')
    if not trial['start_frame'] < trial['lift_start_frame'] < trial['lift_stop_frame'] < trial['stop_frame']:
        reasons.append('incomplete_lift_boundaries')
    return reasons


def diversity_vector(t):
    m=t['carry_metrics']; a=np.radians(m['carry_direction_body_deg'])
    return np.array([*np.asarray(m['start_box_forward_right_m'])/.5,
                     m['start_box_relative_height_m']/.3,m['start_pelvis_height_m']/.3,
                     np.cos(a),np.sin(a),m['max_box_bottom_height_m']/.5])


def select_diverse(candidates, count, max_per_source):
    """Balance dimensions first, then sources/actors, then starting-pose distance."""
    selected=[]; sizes=Counter(); sources=Counter(); actors=Counter()
    remaining=list(candidates)
    while remaining and len(selected)<count:
        eligible=[t for t in remaining if sources[t['motion_id']]<max_per_source]
        if not eligible: break
        def score(t):
            distance=min((float(np.linalg.norm(diversity_vector(t)-diversity_vector(s))) for s in selected),default=0.)
            m=t['carry_metrics']
            return (sizes[t['box_geometry']['size_group']], sources[t['motion_id']], actors[t['actor_id']],
                    -distance,-m['end_released_stable_s'],t['trial_id'])
        chosen=min(eligible,key=score)
        selected.append(chosen); remaining.remove(chosen)
        sizes[chosen['box_geometry']['size_group']]+=1
        sources[chosen['motion_id']]+=1;actors[chosen['actor_id']]+=1
    return selected


@dataclass
class Args:
    data_root: Path
    output: Path
    count: int = 10
    minimum_distance_m: float = 1.0
    distance_mode: str = 'endpoint'
    max_per_source: int = 2
    export_stride: int = 3


def main(args):
    if args.distance_mode not in ('airborne-path','endpoint'): raise ValueError('Unknown distance mode')
    if not args.output.resolve().is_relative_to(Path('/data')): raise ValueError('Output must be under /data')
    args.output.mkdir(parents=True,exist_ok=False)
    packages={}
    for path in sorted((args.data_root/'data').glob('*/*/*/metadata.json')):
        m=json.loads(path.read_text())
        if (not m['mirrored'] and m['frame']=='Bringing' and len(m['objects'])==1
                and m['objects'][0]['object_category']=='box'):
            packages[m['motion_id']]=path.parent
    segment_main(SegmentArgs(output=args.output/'proposals.json', motion_ids=tuple(packages), data_root=args.data_root))
    manifest=json.loads((args.output/'proposals.json').read_text())
    cfg=Thresholds(**manifest['thresholds']); all_trials=[]; shape_cache={}; feature_paths={}
    feature_dir=args.output/'features'; feature_dir.mkdir()
    for mid,package in packages.items():
        motion=load_motion(package,args.data_root)
        nodes,*_=parse_bvh(package/'motion_actor.bvh'); names=[n.name for n in nodes]
        obj=motion.metadata['objects'][0]; mesh=obj['mesh_path']
        if mesh not in shape_cache: shape_cache[mesh]=cuboid_geometry(args.data_root/mesh)
        geom=shape_cache[mesh]
        features=source_features(motion,names,cfg)
        # Preserve source measurements for reproducible review plots.
        feature_path=feature_dir/f'{mid}.npz'
        np.savez_compressed(feature_path,**features,frame_time=motion.frame_time)
        feature_paths[mid]=str(feature_path.resolve())
        for trial in [t for t in manifest['trials'] if t['motion_id']==mid]:
            metrics=measure_trial(motion,names,features,trial,cfg)
            reasons=rejection_reasons(metrics,geom,trial,args.distance_mode,args.minimum_distance_m)
            trial.update(actor_id=motion.metadata['actor_id'],box_geometry=geom,carry_metrics=metrics,
                         filter_rejections=reasons,selection_status='rejected' if reasons else 'eligible')
            all_trials.append(trial)
        print(mid,'eligible',sum(not t['filter_rejections'] for t in all_trials if t['motion_id']==mid),flush=True)
    passing=[t for t in all_trials if not t['filter_rejections']]
    selected=select_diverse(passing,args.count,args.max_per_source)
    selected_ids={t['trial_id'] for t in selected}
    export_dir=args.output/'retarget_inputs'; export_dir.mkdir()
    # The existing adapter writes the unmodified synchronized source geometry.
    from export_retarget_input import main as export_main, Args as ExportArgs
    for index,t in enumerate(selected,1):
        label=f"carry_{index:03d}"; path=export_dir/f'{label}.npz'
        export_main(ExportArgs(motion_id=t['motion_id'],output=path,data_root=args.data_root,
                             start_frame=t['start_frame'],stop_frame=t['stop_frame'],stride=args.export_stride))
        t.update(label=label,selection_status='selected',retarget_input=str(path.resolve()),
                 retarget_input_sha256=sha256(path),export_stride=args.export_stride)
    manifest.update(trials=all_trials,method='floor_box_lift_proposals_v1_with_rectangular_carry_filter',
        carry_filter=dict(distance_mode=args.distance_mode,minimum_distance_m=args.minimum_distance_m,
                          minimum_lift_s=.4,minimum_stable_support_s=.3,minimum_hand_near_fraction=.8,
                          support_floor_tolerance_m=.03,
                          minimum_human_airborne_path_m=args.minimum_distance_m,
                          release_required=False,source_frame_labels='Bringing/*; originals only'),
        selection=dict(requested_count=args.count,selected_count=len(selected),max_per_source=args.max_per_source,
                       policy='balance box dimensions, source recordings, actors, then farthest starting pose/carry direction'),
        feature_paths=feature_paths,code_sha256={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('segment_box_trials.py'),Path(__file__).with_name('export_retarget_input.py'))},
        notes='Geometric source candidates; human outcome remains unknown and robot execution not run. Raw data unchanged.')
    validate_manifest(manifest,args.data_root)
    save_json(args.output/'screened_trials.json',manifest)
    selected_manifest={**manifest,'trials':selected,'sources':{k:v for k,v in manifest['sources'].items() if k in {t['motion_id'] for t in selected}}}
    validate_manifest(selected_manifest,args.data_root)
    save_json(args.output/'selected_trials.json',selected_manifest)
    summary=dict(sources_screened=len(packages),proposed=len(all_trials),eligible=len(passing),selected=len(selected),
        source_recordings=len({t['motion_id'] for t in selected}),actors=len({t['actor_id'] for t in selected}),
        box_size_counts=dict(Counter(t['box_geometry']['size_group'] for t in selected)),
        rejection_counts=dict(Counter(r for t in all_trials for r in t['filter_rejections'])))
    save_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main(tyro.cli(Args))
