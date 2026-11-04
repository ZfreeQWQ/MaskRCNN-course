"""Shared own RLE data and COCO evaluator; no architecture imports.
Extracted from our tested train_official_m2f.py and official_m2f_fit.py.
"""
import copy,gzip,json,random,sys,time
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from PIL import Image
from detectron2.structures import Instances,BitMasks
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import settings
from course_data import BlenderIndex
from io_utils import atomic_json,sha
base=SimpleNamespace(B=settings.DATA_ROOT.parent,BlenderIndex=BlenderIndex,atomic_json=atomic_json,sha=sha)



class Data(torch.utils.data.Dataset):

    def __init__(self, split, seed, limit=0):
        self.index = base.BlenderIndex(settings.DATA_ROOT, split)
        if limit:
            self.index.records = self.index.records[:limit]
        self.split, self.seed = (split, seed)
        keys = ('image_id', 'category_id', 'segmentation', 'area', 'bbox', 'iscrowd')
        self.annotations = []
        for i in range(len(self)):
            self.annotations.append([{k: a[k] for k in keys if k in a} for a in self.index.annotations(i) if a['area'] > 0])
        self.index._cache.clear()

    def __len__(self):
        return len(self.index.records)

    def item(self, i, targets=True, flip=None):
        r = self.index.records[i]
        with Image.open(self.index.path(r['file_name'])) as image:
            rgb = np.array(image.convert('RGB'), copy=True)
        assert rgb.shape == (r['height'], r['width'], 3)
        assert rgb.shape[:2] == (1024, 1024), 'Native-size baseline requires 1024 images'
        if flip is None:
            flip = targets and self.split == 'train' and (random.Random(self.seed + i).random() < 0.5)
        if flip:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        result = dict(image=torch.from_numpy(rgb).permute(2, 0, 1).float(), height=r['height'], width=r['width'], image_id=r['id'])
        if targets:
            anns = [a for a in self.annotations[i] if not a.get('iscrowd', 0)]
            assert all((a['category_id'] == 1 for a in anns))
            masks = mu.decode([a['segmentation'] for a in anns]) if anns else np.zeros((r['height'], r['width'], 0), dtype=np.uint8)
            assert masks.shape == (r['height'], r['width'], len(anns))
            assert np.array_equal(masks.sum((0, 1)), [a['area'] for a in anns])
            if flip:
                masks = masks[:, ::-1]
            masks = torch.from_numpy(masks.transpose(2, 0, 1).copy()).bool()
            instances = Instances((r['height'], r['width']))
            instances.gt_classes = torch.zeros(len(masks), dtype=torch.long)
            instances.gt_masks = masks
            instances.gt_boxes = BitMasks(masks).get_bounding_boxes()
            result['instances'] = instances
        return result

    def __getitem__(self, i):
        return self.item(i)

def collate(batch):
    return batch

class TestData(Data):
    """Image-only inference metadata: never loads test annotations."""
    def __init__(self,limit=0):
        raw=json.loads(settings.IMAGE_METADATA.read_text())
        assert set(raw).issubset({'images','categories','info','licenses'}), 'Use image-only metadata'
        self.records=raw['images'][:limit or None]
        assert len({r['id'] for r in self.records})==len(self.records)
        self.root=settings.DATA_ROOT;self.annotations=[[] for r in self.records]
        self.index=self;self.split='test';self.seed=0
    def path(self,relative):
        p=(self.root/relative).resolve();p.relative_to(self.root);return p
    def item(self,i,targets=False,flip=False):
        assert not targets and not flip
        return super().item(i,False,False)
    def __getitem__(self,i):return self.item(i,False,False)

def optimizer(model, cfg):
    groups = []
    memo = set()
    norm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm)
    for name, module in model.named_modules():
        for key, p in module.named_parameters(recurse=False):
            if not p.requires_grad or p in memo:
                continue
            memo.add(p)
            lr = 0.0001 * (0.1 if 'backbone' in name else 1.0)
            wd = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm):
                wd = cfg.SOLVER.WEIGHT_DECAY_NORM
            if isinstance(module, nn.Embedding):
                wd = cfg.SOLVER.WEIGHT_DECAY_EMBED
            if 'relative_position_bias_table' in key or 'absolute_pos_embed' in key:
                wd = 0.0
            groups.append(dict(params=[p], lr=lr, weight_decay=wd))
    return torch.optim.AdamW(groups, lr=0.0001)
