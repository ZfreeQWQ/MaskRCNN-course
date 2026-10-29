#!/usr/bin/env python3
"""Exhaustive public-label validation and train-only depth statistics.

All masks are decoded. Test labels are never accessed. Depth percentiles use every
valid training pixel, with per-image calibration preserved through grouped histograms.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from prepare_data import EXPECTED, image_identity, write_json


def inspect_shard(task):
    path, source, split = map(str, task)
    source = Path(source)
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    images = {im['id']: im for im in data['images']}
    assert len(images) == len(data['images'])
    assert len(data['categories']) == 1 and data['categories'][0]['id'] == 1
    areas, types, count, histograms = [], Counter(), Counter(), defaultdict(lambda: np.zeros(65536, dtype=np.uint64))
    depths_total = depths_invalid = decoded = 0
    ann_ids = []
    for ann in data['annotations']:
        im = images[ann['image_id']]
        assert ann['category_id'] == 1 and ann['iscrowd'] == 0, ann['id']
        rle = ann['segmentation']
        assert rle['size'] == [im['height'], im['width']], ann['id']
        mask = mask_util.decode(rle)
        assert mask.shape == (im['height'], im['width']) and mask.dtype == np.uint8, ann['id']
        area = int(np.count_nonzero(mask))
        assert area > 0 and area == ann['area'] == int(mask_util.area(rle)), (ann['id'], 'area')
        bbox = mask_util.toBbox(rle).tolist()
        assert np.allclose(bbox, ann['bbox'], rtol=0, atol=1e-6), (ann['id'], 'bbox')
        x,y,w,h = bbox
        assert x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= im['width'] and y+h <= im['height']
        areas.append(area); types[ann.get('semantic_type', '<missing>')] += 1
        count[ann['image_id']] += 1; ann_ids.append(ann['id']); decoded += 1
    for im in data['images']:
        iid, scene, view = image_identity(im['file_name'])
        assert iid == im['id'] and scene == im['scene_id'] and view == im['view']
        assert im['split'] == split and Path(im['file_name']).name == Path(im['depth_file_name']).name
        assert np.isfinite(im['depth_scale_m']) and im['depth_scale_m'] > 0
        with Image.open(source / im['file_name']) as rgb:
            assert rgb.size == (im['width'], im['height']) and rgb.mode == 'RGB', im['id']
            rgb.verify()
        with Image.open(source / im['depth_file_name']) as dep:
            depth = np.array(dep)
        assert depth.dtype == np.uint16 and depth.shape == (im['height'], im['width']), im['id']
        if split == 'train':
            h = np.bincount(depth.ravel(), minlength=65536).astype(np.uint64)
            invalid_code = im['depth_invalid_code']
            assert isinstance(invalid_code, int) and 0 <= invalid_code <= 65535
            depths_total += depth.size; depths_invalid += int(h[invalid_code]); h[invalid_code] = 0
            histograms[float(im['depth_scale_m'])] += h
    return {'path': Path(path).name, 'images': [(im['id'], im['scene_id']) for im in data['images']],
            'annotations': ann_ids, 'areas': areas, 'types': dict(types),
            'counts': [count[im['id']] for im in data['images']], 'decoded': decoded,
            'depth_total': depths_total, 'depth_invalid': depths_invalid,
            'histograms': [(scale, np.flatnonzero(h), h[h>0]) for scale,h in histograms.items()]}


def summary(values):
    a = np.asarray(values)
    return {'min': float(a.min()), 'p25': float(np.percentile(a,25)),
            'median': float(np.median(a)), 'p75': float(np.percentile(a,75)),
            'max': float(a.max()), 'mean': float(a.mean())}


def weighted_percentile(values, counts, percentiles):
    # Match numpy percentile(method='linear') over the full multiset without materializing it.
    order = np.argsort(values); values = values[order]; counts = counts[order]
    cumulative = np.cumsum(counts, dtype=np.uint64)
    n = int(cumulative[-1]); result = []
    for p in percentiles:
        rank = (n - 1) * p / 100
        lo, hi = int(np.floor(rank)), int(np.ceil(rank))
        a = values[np.searchsorted(cumulative, lo, side='right')]
        b = values[np.searchsorted(cumulative, hi, side='right')]
        result.append(float(a + (b-a) * (rank-lo)))
    return result


def visualize(source, out, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for split in ('train','val'):
        shards = sorted((source / 'annotations' / split).glob('*.json.gz'))
        fig, axs = plt.subplots(3,3,figsize=(15,15))
        for row,index in enumerate([0,len(shards)//2,len(shards)-1]):
            with gzip.open(shards[index], 'rt') as f:
                data = json.load(f)
            im = data['images'][0]; anns = [a for a in data['annotations'] if a['image_id']==im['id']]
            rgb = np.array(Image.open(source / im['file_name']))
            raw = np.array(Image.open(source / im['depth_file_name']))
            depth = raw.astype(np.float64)*im['depth_scale_m']; depth[raw==im['depth_invalid_code']] = np.nan
            overlay = rgb.astype(np.float32)/255
            for j,ann in enumerate(anns):
                mask = mask_util.decode(ann['segmentation']).astype(bool)
                color = np.array(plt.get_cmap('tab20')(j%20)[:3]); overlay[mask] = .45*overlay[mask]+.55*color
            axs[row,0].imshow(rgb); axs[row,0].set_title(f'{split}: {im["id"]} RGB')
            artist=axs[row,1].imshow(depth,vmin=stats['p1'],vmax=stats['p99'],cmap='viridis')
            axs[row,1].set_title('Depth (m), training p1/p99 color scale'); fig.colorbar(artist,ax=axs[row,1],fraction=.046)
            axs[row,2].imshow(overlay); axs[row,2].set_title(f'Public masks ({len(anns)} instances)')
            for ax in axs[row]: ax.axis('off')
        fig.tight_layout();fig.savefig(out/f'data_{split}_examples.png',dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'blender')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--no-visualize',action='store_true')
    args=parser.parse_args(); source=args.source.resolve(); project=Path(__file__).resolve().parents[1]
    started=time.time(); report={'test_annotations_read':False,'mask_validation':'Every train/val instance decoded; pixel area, COCO area, tight bbox and dimensions checked.', 'splits':{}}
    hist=defaultdict(lambda:np.zeros(65536,dtype=np.uint64));total=invalid=0;split_scenes={};split_ids={}
    for split in ('train','val'):
        files=sorted((source/'annotations'/split).glob('*.json.gz'))
        ids=set();ann_ids=set();scenes=set();areas=[];counts=[];types=Counter();decoded=0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n,result in enumerate(pool.map(inspect_shard, [(p,source,split) for p in files]),1):
                for iid,scene in result['images']:
                    assert iid not in ids, iid
                    ids.add(iid);scenes.add(scene)
                for aid in result['annotations']:
                    assert aid not in ann_ids, aid
                    ann_ids.add(aid)
                areas.extend(result['areas']);counts.extend(result['counts']);types.update(result['types']);decoded+=result['decoded']
                total+=result['depth_total'];invalid+=result['depth_invalid']
                for scale,index,values in result['histograms']:hist[scale][index]+=values
                if n%100==0 or n==len(files):print(f'{split}: {n}/{len(files)} shards; {decoded} masks decoded',flush=True)
        assert (len(ids),len(ann_ids))==EXPECTED[split]
        split_scenes[split]=scenes;split_ids[split]=ids
        a=np.asarray(areas)
        report['splits'][split]={'images':len(ids),'annotations':len(ann_ids),'all_masks_decoded':decoded,
          'scenes':len(scenes),'scene_min':min(scenes),'scene_max':max(scenes),'area_px':summary(areas),
          'instances_per_image':summary(counts),'coco_area_counts':{'small_lt_1024':int((a<1024).sum()),
           'medium_1024_to_9216':int(((a>=1024)&(a<9216)).sum()),'large_ge_9216':int((a>=9216).sum())},
          'semantic_types':dict(types)}
    assert not split_scenes['train']&split_scenes['val'] and not split_ids['train']&split_ids['val']
    values=np.concatenate([np.flatnonzero(h).astype(np.float64)*scale for scale,h in hist.items()])
    frequencies=np.concatenate([h[h>0] for h in hist.values()])
    p1,p50,p99=weighted_percentile(values,frequencies,[1,50,99])
    stats={'p1':p1,'p99':p99,'median':p50,'unit':'m','split':'train','train_only':True,
      'method':'Exact all-valid-training-pixel histogram by per-image depth_scale_m; linear percentile (NumPy convention). Invalid code excluded per image.',
      'images':EXPECTED['train'][0],'total_pixels':total,'valid_pixels':total-invalid,'invalid_pixels':invalid,
      'invalid_fraction':invalid/total,'minimum_m':float(values.min()),'maximum_m':float(values.max()),
      'depth_scale_values':sorted(hist),'scope':'Descriptive training depth statistics only; model preprocessing is defined by mapper/config.'}
    assert p99>p1
    write_json(project/'datasets/blender/annotations/depth_stats.json',stats)
    report['train_depth']=stats;report['scene_disjoint']=True;report['elapsed_seconds']=time.time()-started
    prep=json.loads((project/'reports/data/data_preparation.json').read_text())
    for split in ('train','val'):
        path=project/'datasets/blender/annotations'/f'instances_{split}.json';digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==prep['merged_sha256'][split], 'Merged annotation changed after preparation'
    report['merged_files_match_preparation_sha256']=True
    report['test_metadata']=prep['test']
    write_json(project/'reports/data/data_validation.json',report)
    if not args.no_visualize:
        out=project/'reports/data/figures';out.mkdir(parents=True,exist_ok=True);visualize(source,out,stats)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
