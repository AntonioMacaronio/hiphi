#!/usr/bin/env python3
"""Render source-pose/trajectory evidence for selected geometric carry trials."""
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import tyro

from curate_box_carries import body_frame


@dataclass
class Args:
    manifest: Path


def main(args):
    manifest=json.loads(args.manifest.read_text())
    output=args.manifest.parent/'review'
    output.mkdir(exist_ok=False)
    rows=[]
    plt.rcParams.update({'font.size':9})
    for t in manifest['trials']:
        with np.load(t['retarget_input'],allow_pickle=False) as d:
            frames=d['source_frames'];human=d['human_positions'];edges=d['edges']
            names=d['joint_names'].tolist(); p=d['object_positions']; q=d['object_quaternions_wxyz']
            vertices=d['object_vertices'];faces=d['object_faces']
        with np.load(manifest['feature_paths'][t['motion_id']]) as f:
            a,b=t['start_frame'],t['stop_frame'];dt=float(f['frame_time'])
            bottom=f['bottom_m'][a:b];positions=f['object_positions'][a:b]
        metrics=t['carry_metrics']
        phase_frames=[a,t['lift_start_frame'],a+int(np.argmax(bottom)),t['lift_stop_frame'],b-1]
        titles=['Start','Lift threshold','Peak lift','Lowering threshold','End']
        fig=plt.figure(figsize=(16,7),layout='constrained')
        grid=fig.add_gridspec(2,5,height_ratios=[1.5,1])
        for k,(frame,title) in enumerate(zip(phase_frames,titles)):
            i=int(np.argmin(abs(frames-frame)))
            box=Rotation.from_quat(q[i,[1,2,3,0]]).apply(vertices)+p[i]
            pts=human[i][:,[0,2,1]];box=box[:,[0,2,1]]
            ax=fig.add_subplot(grid[0,k],projection='3d')
            ax.add_collection3d(Line3DCollection(pts[edges],colors='#206f9c',linewidths=1.4))
            ax.add_collection3d(Poly3DCollection(box[faces],facecolors='#ed9a35',edgecolors='#9d6421',alpha=.65,linewidths=.5))
            extent=np.concatenate([pts,box])
            center=(extent.min(0)+extent.max(0))/2
            span=max(1.8,float(np.ptp(extent,axis=0).max())+.2)
            ax.set_xlim(center[0]-span/2,center[0]+span/2)
            ax.set_ylim(center[1]-span/2,center[1]+span/2)
            ax.set_zlim(-.1,max(1.9,float(extent[:,2].max())+.1))
            ax.set_box_aspect([1,1,1.15]);ax.view_init(16,-55)
            ax.set_title(f'{title}\nsource {int(frames[i])} | {(frames[i]-a)*dt:.2f} s')
            ax.set_xlabel('X');ax.set_ylabel('Z');ax.set_zlabel('Height (m)')
        ax=fig.add_subplot(grid[1,:3])
        time=np.arange(len(bottom))*dt
        ax.plot(time,bottom,label='Box bottom above floor',color='#bf6c13')
        ax.axhline(.1,color='#7c8d99',ls='--',label='Lift threshold (0.10 m)')
        tolerance=manifest['carry_filter']['support_floor_tolerance_m']
        ax.axhspan(-tolerance,tolerance,color='#688c59',alpha=.12,label='Floor support tolerance')
        for key,color in [('pickup_supported_frames','#317db5'),('setdown_supported_frames','#53964b')]:
            lo,hi=metrics[key];ax.axvspan((lo-a)*dt,(hi-a)*dt,color=color,alpha=.2)
        ax.set(xlabel='Time within segment (s)',ylabel='Height (m)');ax.legend(loc='upper right',fontsize=8)
        ax=fig.add_subplot(grid[1,3:])
        heading=body_frame(human,names,0)
        origin=human[0,names.index('Hips')]
        track=(positions-origin)@heading
        ax.plot(track[:,0],track[:,1],color='#bf6c13',lw=1)
        for key,label,color in [('pickup_center_world_y_up','Pickup','#317db5'),('setdown_center_world_y_up','Set-down','#53964b')]:
            point=(np.array(metrics[key])-origin)@heading
            ax.scatter(*point,color=color,label=label,zorder=4)
        ax.scatter(0,0,marker='^',color='#206f9c',label='Person at start')
        ax.set(xlabel='Forward from initial person (m)',ylabel='Right (m)');ax.axis('equal');ax.legend(fontsize=8)
        dims=' × '.join(f'{d*100:g}' for d in sorted(t['box_geometry']['dimensions_m']))
        fig.suptitle(f"{t['label']} · {t['trial_id']} · {t['actor_id']}\n"
                     f"Box {dims} cm | pickup→set-down {metrics['pickup_setdown_displacement_m']:.2f} m | {metrics['duration_s']:.2f} s")
        fig.savefig(output/f"{t['label']}.png",dpi=110);plt.close(fig)
        rows.append(dict(label=t['label'],trial_id=t['trial_id'],actor=t['actor_id'],
            box=t['object_id'],box_dimensions_m=t['box_geometry']['size_group'],
            start_frame=a,stop_frame=b,duration_s=round(metrics['duration_s'],3),
            pickup_setdown_m=round(metrics['pickup_setdown_displacement_m'],3),
            start_forward_m=round(metrics['start_box_forward_right_m'][0],3),
            start_right_m=round(metrics['start_box_forward_right_m'][1],3),
            start_pelvis_height_m=round(metrics['start_pelvis_height_m'],3),
            carry_direction_deg=round(metrics['carry_direction_body_deg'],1),
            end_release_confirmed=metrics['end_released_stable_s']>=.3,
            retarget_input=t['retarget_input']))
        print('Rendered',t['label'],flush=True)
    with (args.manifest.parent/'selected_trials.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cards='\n'.join(f'<article><h2>{r["label"]} — {html.escape(r["trial_id"])}</h2>'
        f'<p>{r["box_dimensions_m"]} m · {r["pickup_setdown_m"]} m displacement · {r["duration_s"]} s</p>'
        f'<img loading="lazy" src="{r["label"]}.png" width="100%" alt="Source poses and geometric measurements">'
        f'<p><a href="../retarget_inputs/{r["label"]}.npz">Retargeting input</a></p></article>' for r in rows)
    (output/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>HiPHI carry pilot</title>'
        '<style>body{max-width:1500px;margin:30px auto;font-family:system-ui;background:#f5f7fa;color:#172c3b}'
        'article{background:white;padding:20px;margin:25px 0;border-radius:12px}a{color:#206f9c}</style>'
        '<h1>HiPHI rectangular-box carry pilot</h1><p>Geometric candidates: supported pickup, sustained lift, '
        'at least 1 m between pickup and stable set-down. Frames are zero-based, stop exclusive. '
        'Source human motion; no robot retargeting or physical execution certification.</p>'
        '<p><a href="../selected_trials.json">Manifest</a> · <a href="../selected_trials.csv">CSV</a></p>'+cards+'</html>')


if __name__=='__main__':main(tyro.cli(Args))
