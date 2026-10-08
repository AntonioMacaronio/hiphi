#!/usr/bin/env python3
"""Independently check selected exports against raw object tracks and actual mesh vertices."""
import csv
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import median_filter
import trimesh
import tyro
from curate_box_carries import sha256, save_json


@dataclass
class Args:
    manifest: Path


def main(args):
    m=json.loads(args.manifest.read_text());root=Path(m['source_data_root'])
    assert len(m['trials'])==m['selection']['requested_count']
    assert m['carry_filter']['distance_mode']=='endpoint'
    assert len({t['box_geometry']['size_group'] for t in m['trials']})>=2
    reports=[]
    for mid,source in m['sources'].items():
        for relative,expected in source['sha256'].items():assert sha256(root/relative)==expected
    for t in m['trials']:
        source=m['sources'][t['motion_id']];package=root/source['package']
        meta=json.loads((package/'metadata.json').read_text());obj=meta['objects'][0]
        rows=list(csv.DictReader(open(package/obj['trajectory_path'])))
        p=np.array([[float(r[k]) for k in ('px','py','pz')] for r in rows])
        q=np.array([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in rows])
        assert all(int(r['frame'])==i for i,r in enumerate(rows))
        rot=Rotation.from_quat(q)
        mesh=trimesh.load(root/obj['mesh_path'],force='mesh',process=False)
        v=np.asarray(mesh.vertices)*.01
        center=p+rot.apply((v.min(0)+v.max(0))/2)
        # Actual mesh vertices; independently avoids the segmenter's AABB proxy.
        bottom=np.einsum('tij,vj->tvi',rot.as_matrix(),v)[:,:,1].min(1)+p[:,1]
        a,b=t['start_frame'],t['stop_frame'];la,lb=t['lift_start_frame'],t['lift_stop_frame']
        dt=source['frame_time_s'];metrics=t['carry_metrics']
        pickup=slice(*metrics['pickup_supported_frames']);setdown=slice(*metrics['setdown_supported_frames'])
        displacement=float(np.linalg.norm(np.median(center[setdown][:,[0,2]],0)-np.median(center[pickup][:,[0,2]],0)))
        assert displacement>=1., (t['trial_id'],displacement)
        assert abs(displacement-metrics['pickup_setdown_displacement_m'])<2e-6
        for window in (pickup,setdown):
            assert (window.stop-window.start)*dt>=.3
            assert np.all(np.abs(bottom[window])<=m['carry_filter']['support_floor_tolerance_m']+.0001)
        elevated=bottom[la:lb]>.1
        changes=np.diff(np.r_[0,elevated.astype(int),0])
        duration=float(max(np.flatnonzero(changes==-1)-np.flatnonzero(changes==1),default=0)*dt)
        assert duration>=.4
        assert sha256(t['retarget_input'])==t['retarget_input_sha256']
        with np.load(t['retarget_input'],allow_pickle=False) as d:
            frames=np.arange(a,b,t['export_stride'])
            np.testing.assert_array_equal(d['source_frames'],frames)
            np.testing.assert_allclose(d['source_timestamps'],frames*dt,atol=1e-9)
            np.testing.assert_allclose(d['object_positions']-d['object_positions'][0],p[frames]-p[frames[0]],atol=2e-6)
            np.testing.assert_allclose(d['object_quaternions_wxyz'][:,[1,2,3,0]],q[frames],atol=1e-9)
            assert len(d['human_positions'])==len(frames) and np.isfinite(d['human_positions']).all()
            assert abs(float(d['fps'])-1/(dt*t['export_stride']))<1e-8
        reports.append(dict(label=t['label'],trial_id=t['trial_id'],source_and_export_hashes='passed',
            synchronized_source_frames='passed',pickup_setdown_displacement_m=displacement,
            sustained_actual_mesh_lift_s=duration,actual_mesh_support_start_bottom_m=float(np.median(bottom[pickup])),
            actual_mesh_support_end_bottom_m=float(np.median(bottom[setdown])),export_frames=len(frames)))
    save_json(args.manifest.parent/'verification.json',dict(status='passed',count=len(reports),trials=reports,
        limits='Source geometric checks; no contact-force labels or robot execution validation.'))
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main(tyro.cli(Args))
