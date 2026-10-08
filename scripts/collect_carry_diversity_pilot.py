#!/usr/bin/env python3
"""Selectively extract a few original carrying recordings per cuboid size.

Scans existing local archives, retaining metadata for Bringing-box candidates.
Copies only selected packages, never modifies a source. No downloads or mirrors.
"""
import csv
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from dataclasses import dataclass
from collections import Counter

import tyro
import zstandard
from curate_box_carries import cuboid_geometry, sha256, save_json
from collect_stage1 import storage_check


@dataclass
class Args:
    output: Path
    run: Path
    parts: tuple[int,...] = (22,21,23)
    per_size: int = 5
    expanded_size_group: str = ''
    expanded_size_count: int = 20
    raw_root: Path = Path('/data/sky/projects/hiphi/datasets/HiPHI')
    seed_root: Path = Path('/data/sky/projects/hiphi/datasets/stage1_v1/HiPHI')
    manifest: Path = Path('/data/sky/projects/hiphi/runs/full_release_b50ee4b/release_manifest.json')


def main(args):
    if not args.output.resolve().is_relative_to(Path('/data')): raise ValueError('Use /data')
    if args.output.exists() or args.run.exists(): raise FileExistsError('Use fresh output paths')
    storage_check(2*2**30)
    print('Extraction PID',os.getpid(),'output',args.output,flush=True)
    args.output.mkdir(parents=True);args.run.mkdir(parents=True)
    (args.run/'pid').write_text(str(os.getpid())+'\n')
    shapes={p.name:cuboid_geometry(p) for p in args.raw_root.joinpath('object_meshes').glob('Box*.obj') if '__mirror' not in p.name}
    save_json(args.run/'box_geometry.json',shapes)
    eligible_sizes={v['size_group'] for v in shapes.values() if v['is_rectangular_prism']}
    counts=Counter(); selected={}; catalog=[]; verified=[]
    def quota(group):
        return args.expanded_size_count if group==args.expanded_size_group else args.per_size
    def copy_companion(m):
        for obj in m['objects']:
            relative=obj['mesh_path'];dest=args.output/relative
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(args.raw_root/relative,dest)
    for path in sorted(args.seed_root.joinpath('data').glob('*/*/*/metadata.json')):
        m=json.loads(path.read_text())
        if m['mirrored'] or m['frame']!='Bringing' or len(m['objects'])!=1:continue
        obj=m['objects'][0]
        geometry=shapes.get(Path(obj['mesh_path']).name)
        if not geometry or not geometry['is_rectangular_prism']:continue
        group=geometry['size_group']
        if counts[group]>=quota(group):continue
        destination=args.output/path.parent.relative_to(args.seed_root)
        shutil.copytree(path.parent,destination);copy_companion(m)
        selected[m['motion_id']]=dict(metadata=m,origin='existing_pilot',geometry=geometry)
        counts[group]+=1
    inventory=json.loads(args.manifest.read_text())
    hashes={r['path']:r['sha256'] for r in inventory['files']}
    for part in args.parts:
        if all(counts[s]>=quota(s) for s in eligible_sizes):break
        archive=args.raw_root/'data'/f'HiPHI_data_part_{part:04d}_of_0032.tar.zst'
        print('Verifying then scanning',archive.name,'size counts',dict(counts),flush=True)
        if sha256(archive)!=hashes[str(archive.relative_to(args.raw_root))]:raise ValueError('Archive hash mismatch')
        verified.append(str(archive))
        chosen_packages={}
        with archive.open('rb') as f,zstandard.ZstdDecompressor().stream_reader(f) as z,tarfile.open(fileobj=z,mode='r|') as tar:
            for member in tar:
                path=PurePosixPath(member.name)
                if not member.isfile() or '..' in path.parts or path.is_absolute():continue
                parts=path.parts
                if len(parts)<6 or parts[:2]!=('HiPHI','data'):continue
                mid=parts[4]
                if path.name=='metadata.json' and '__mirror' not in mid:
                    m=json.load(tar.extractfile(member))
                    if m['frame']!='Bringing' or len(m['objects'])!=1 or m['objects'][0]['object_category']!='box':continue
                    obj=m['objects'][0];geometry=shapes.get(Path(obj['mesh_path']).name)
                    catalog.append(dict(metadata=m,archive=str(archive),geometry=geometry))
                    if mid in selected or not geometry or not geometry['is_rectangular_prism']:continue
                    group=geometry['size_group']
                    if counts[group]>=quota(group):continue
                    package=args.output/'data'/m['frame']/m['lu']/mid
                    package.mkdir(parents=True,exist_ok=False)
                    save_json(package/'metadata.json',m);copy_companion(m)
                    chosen_packages[mid]=m;selected[mid]=dict(metadata=m,origin=str(archive),geometry=geometry)
                    counts[group]+=1
                    print('Selected',mid,obj['mesh_id'],group,flush=True)
                elif mid in chosen_packages:
                    m=chosen_packages[mid]
                    tail=PurePosixPath(*parts[5:])
                    allowed={'motion_actor.bvh',m['objects'][0]['trajectory_path']}
                    if str(tail) not in allowed:continue
                    if member.size>100*1024**2:raise ValueError('Unexpected oversized pilot file')
                    destination=args.output/'data'/m['frame']/m['lu']/mid/str(tail)
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with destination.open('xb') as out:shutil.copyfileobj(tar.extractfile(member),out)
        print('Scanned',part,'counts',dict(counts),flush=True)
    for mid,item in selected.items():
        m=item['metadata'];base=args.output/'data'/m['frame']/m['lu']/mid
        for relative in ('metadata.json','motion_actor.bvh',m['objects'][0]['trajectory_path']):
            if not (base/relative).is_file():raise ValueError(f'Missing {mid}/{relative}')
        item['source_sha256']={r:sha256(base/r) for r in ('metadata.json','motion_actor.bvh',m['objects'][0]['trajectory_path'])}
    save_json(args.run/'archive_catalog.json',catalog)
    save_json(args.run/'collection.json',dict(selected=selected,size_counts=dict(counts),verified_archives=verified,
        status='complete',scope='Small source pool for ten diverse mini-sequences; catalog covers scanned archives only'))
    print('COMPLETE',len(selected),'recordings',dict(counts),flush=True)
    storage_check()


if __name__=='__main__': main(tyro.cli(Args))
