"""Additional official MaskDINO RGB / independently adapted RGB-depth trials.

Official detection+segmentation losses, hybrid matching, query selection and DN
are retained. Public method is attributed, not claimed as our innovation.
"""
import argparse,json,math,os,random,sys,time,traceback
from pathlib import Path
import settings
B=settings.DATA_ROOT.parent;ENV=settings.WEIGHTS
setup={'commit':settings.UPSTREAM_COMMIT}
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
import independent_seg_data as d


class RGBDepthStem(nn.Module):
    """Zero residual keeps the official RGB stem exact at initialization.

Depth is an independent2-channel convolution added after RGB normalization,
before the existing ReLU/pooling. No teammate geometry/fusion implementation.
"""
    def __init__(self,stem):
        super().__init__();self.rgb_conv=stem.conv1
        self.depth_adapter=nn.Conv2d(2,64,7,stride=2,padding=3,bias=False)
        nn.init.zeros_(self.depth_adapter.weight)
        self.in_channels=5;self.out_channels=64;self.stride=4
    def forward(self,x):
        y=self.rgb_conv(x[:,:3])+self.depth_adapter(x[:,3:])
        return F.max_pool2d(F.relu_(y),3,stride=2,padding=1)


def build(out,modality):
    cfg=get_cfg();add_deeplab_config(cfg);add_maskdino_config(cfg)
    upstream=settings.MASKDINO_SRC
    cfg.merge_from_file(str(upstream/'configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml'))
    MetadataCatalog.get('independent_dino_train').set(thing_classes=['component'],thing_dataset_id_to_contiguous_id={1:0})
    cfg.DATASETS.TRAIN=('independent_dino_train',);cfg.MODEL.DEVICE='cpu'
    model=build_model(cfg)
    checkpoint=torch.load(ENV/'official_instance_r50.pth',map_location='cpu',weights_only=False)
    weights=checkpoint['model'];expected=model.state_dict();aliases={}
    assert not (set(weights)-set(expected)), 'Unexpected official checkpoint keys'
    # Official checkpoints omit duplicated names for shared box-head tensors.
    # Recover only aliases proven to share the SAME live tensor storage; never
    # fill a genuinely absent parameter with its random initialization.
    for key in set(expected)-set(weights):
        target=expected[key]
        candidates=[other for other in weights if expected[other].data_ptr()==target.data_ptr()
                    and expected[other].shape==target.shape and expected[other].stride()==target.stride()]
        if not candidates:raise RuntimeError('Uncovered pretrained parameter: '+key)
        assert all(torch.equal(weights[candidates[0]],weights[other]) for other in candidates)
        aliases[key]=candidates[0];weights[key]=weights[candidates[0]]
    expected.update(weights);model.load_state_dict(expected,strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    loaded=len(weights);del checkpoint
    predictor=model.sem_seg_head.predictor
    assert predictor.class_embed.out_features==80 and predictor.label_enc.num_embeddings==80
    cls=nn.Linear(predictor.class_embed.in_features,1)
    labels=nn.Embedding(1,predictor.label_enc.embedding_dim)
    with torch.no_grad():
        cls.weight.copy_(predictor.class_embed.weight.mean(0,keepdim=True))
        cls.bias.copy_(predictor.class_embed.bias.mean(0,keepdim=True))
        labels.weight.copy_(predictor.label_enc.weight.mean(0,keepdim=True))
    predictor.class_embed=cls;predictor.label_enc=labels;predictor.num_classes=1
    model.sem_seg_head.num_classes=1;model.criterion.num_classes=1
    model.criterion.empty_weight=torch.tensor([1.,cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT])
    excluded={'sem_seg_head.predictor.class_embed.weight','sem_seg_head.predictor.class_embed.bias',
              'sem_seg_head.predictor.label_enc.weight','criterion.empty_weight'}
    retained=[k for k in weights if k not in excluded]
    assert all(torch.equal(weights[k],model.state_dict()[k]) for k in retained)
    del weights
    if modality=='rgbd':
        original=model.backbone.stem;replacement=RGBDepthStem(original)
        probe=torch.randn(1,5,64,64)
        with torch.no_grad():assert torch.equal(original(probe[:,:3]),replacement(probe))
        model.backbone.stem=replacement
        model.pixel_mean=torch.cat([model.pixel_mean,torch.zeros(2,1,1)])
        model.pixel_std=torch.cat([model.pixel_std,torch.ones(2,1,1)])
    receipt=dict(upstream='IDEA-Research/MaskDINO',commit=setup['commit'],
                 checkpoint_sha256=d.sha(ENV/'official_instance_r50.pth'),strict_loaded_tensors=loaded,
                 exactly_retained_tensors=len(retained),proven_shared_tensor_aliases=aliases,
                 class_adaptation='Average official80 foreground rows to1 sigmoid class and label embedding',
                 queries=model.num_queries,modality=modality,depth_residual_zero_check=modality=='rgbd',
                 custom_geometry_or_teammate_modules=False)
    d.atomic_json(out/'initialization.json',receipt)
    (out/'official_config.yaml').write_text(cfg.dump())
    return model.cuda(),cfg


class InputData(torch.utils.data.Dataset):
    def __init__(self,split,modality,limit=0):
        self.data=d.TestData(limit) if split=='test' else d.Data(split,121,limit)
        self.index=self.data.index;self.annotations=self.data.annotations
        self.split=split;self.modality=modality;self.epoch=0
    def __len__(self):return len(self.data)
    def item(self,i,targets=False):
        item=self.data.item(i,targets=targets,flip=False)
        r=self.index.records[i]
        if self.modality=='rgbd':
            with Image.open(self.index.path(r['depth_file_name'])) as im:raw=np.asarray(im)
            assert raw.shape==(r['height'],r['width']) and np.issubdtype(raw.dtype,np.integer)
            valid=raw!=r['depth_invalid_code']
            metric=raw.astype(np.float32)*float(r['depth_scale_m'])
            assert r['depth_scale_m']>0 and np.isfinite(metric).all()
            z=np.where(valid,np.log1p(np.clip(metric,0,10))/math.log(11),0).astype(np.float32)
            depth=torch.from_numpy(np.stack([z,valid.astype(np.float32)]))
            item['image']=torch.cat([item['image'],depth])
        if targets:
            rng=random.Random(121+self.epoch*len(self)+i)
            masks=item['instances'].gt_masks;image=item['image']
            if rng.random()<.5:image=image.flip(-1);masks=masks.flip(-1)
            size=round(1024*rng.uniform(.8,1.2))
            rgb=F.interpolate(image[None,:3],size=(size,size),mode='bilinear',align_corners=False)[0]
            if self.modality=='rgbd':
                depth=F.interpolate(image[None,3:],size=(size,size),mode='nearest')[0]
                image=torch.cat([rgb,depth])
            else:image=rgb
            masks=F.interpolate(masks[None].float(),size=(size,size),mode='nearest')[0].bool() if len(masks) else torch.zeros((0,size,size),dtype=torch.bool)
            x=rng.randrange(max(0,size-1024)+1);y=rng.randrange(max(0,size-1024)+1)
            image=image[:,y:y+1024,x:x+1024];masks=masks[:,y:y+1024,x:x+1024]
            pad=(0,1024-image.shape[-1],0,1024-image.shape[-2])
            rgb=F.pad(image[:3],pad,value=128.)
            image=torch.cat([rgb,F.pad(image[3:],pad,value=0.)]) if self.modality=='rgbd' else rgb
            masks=F.pad(masks,pad,value=False);masks=masks[masks.flatten(1).any(1)]
            instances=d.Instances((1024,1024));instances.gt_masks=masks
            instances.gt_classes=torch.zeros(len(masks),dtype=torch.long)
            instances.gt_boxes=d.BitMasks(masks).get_bounding_boxes()
            item.update(image=image,instances=instances)
        return item
    def __getitem__(self,i):return self.item(i,targets=True)
