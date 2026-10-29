"""Pretrained local decoder; independent GT-piloted training queries, no test GT."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from pycocotools import mask as mu

TRAINABLE=('transformer_decoder.','query_embed.','query_feat.','cls_embed.','mask_embed.')
WINDOWS=((0,0),(512,0),(0,512),(512,512))
SEED=20261013

def clone_head(original):
    # Do not deepcopy bound runtime hooks (which recursively own the full model).
    memo={id(original.pixel_decoder):original.pixel_decoder}
    for module in original.modules():
        for hook in list(module._forward_hooks.values())+list(module._forward_pre_hooks.values()):memo[id(hook)]=hook
    head=copy.deepcopy(original,memo).requires_grad_(False)
    # The pixel decoder is intentionally shared, frozen, and never optimized.
    for n,p in head.named_parameters():p.requires_grad_(n.startswith(TRAINABLE))
    assert any(p.requires_grad for p in head.parameters())
    return head

def decoder(head,pixels,memories,labels=None,targets=None):
    """Same ordinary path as upstream; auxiliary path never shares self-attention."""
    batch=pixels.shape[0];assert batch==1
    inputs=[];positions=[]
    for i in range(head.num_transformer_feat_level):
        z=head.decoder_input_projs[i](memories[i]).flatten(2).permute(0,2,1)
        inputs.append(z+head.level_embed.weight[i].view(1,1,-1))
        empty=z.new_zeros((batch,)+memories[i].shape[-2:],dtype=torch.bool)
        positions.append(head.decoder_positional_encoding(empty).flatten(2).permute(0,2,1))
    guided=labels is not None
    if guided:
        assert targets is not None and len(labels)==len(targets) and len(labels)>0
        # A detached pretrained class embedding, no randomly learned instance head.
        query=head.cls_embed.weight.detach()[labels][None].clone()
        position=torch.zeros_like(query)
    else:
        query=head.query_feat.weight[None].repeat(batch,1,1)
        position=head.query_embed.weight[None].repeat(batch,1,1)
    cs=[];ms=[]
    c,m,attention=head._forward_head(query,pixels,memories[0].shape[-2:]);cs.append(c);ms.append(m)
    for i,layer in enumerate(head.transformer_decoder.layers):
        level=i%head.num_transformer_feat_level
        if guided:
            # Foreground occupancy prevents a one-pixel object vanishing in the guide.
            allowed=F.adaptive_max_pool2d(targets[:,None].float(),memories[level].shape[-2:])[:,0]>0
            assert allowed.flatten(1).any(1).all()
            attention=(~allowed).flatten(1)[None].repeat(head.num_heads,1,1).detach()
        else:attention=attention & (attention.sum(-1)!=attention.shape[-1]).unsqueeze(-1)
        query=layer(query=query,key=inputs[level],value=inputs[level],query_pos=position,
            key_pos=positions[level],cross_attn_mask=attention,query_key_padding_mask=None,key_padding_mask=None)
        c,m,attention=head._forward_head(query,pixels,memories[(i+1)%head.num_transformer_feat_level].shape[-2:])
        cs.append(c);ms.append(m)
    return cs,ms

def guided_loss(head,pixels,memories,gt):
    n=len(gt.labels)
    if not n:return sum(p.sum()*0 for p in head.parameters() if p.requires_grad)
    cs,ms=decoder(head,pixels,memories,gt.labels,gt.masks)
    total=pixels.new_zeros(())
    for c,m in zip(cs,ms):
        total=total+head.loss_cls(c[0],gt.labels,avg_factor=n)
        pred=m[0]
        with torch.no_grad():
            coords=get_uncertain_point_coords_with_randomness(pred[:,None],None,head.num_points,head.oversample_ratio,head.importance_sample_ratio)
            target=point_sample(gt.masks[:,None].float(),coords).squeeze(1)
        pred=point_sample(pred[:,None],coords).squeeze(1)
        total=total+head.loss_dice(pred,target,avg_factor=n)+head.loss_mask(pred.reshape(-1),target.reshape(-1),avg_factor=n*head.num_points)
    return total

def crop_targets(coco,ident,xy):
    x,y=xy;assert xy in WINDOWS
    rows=[];masks=[]
    for a in sorted(coco.loadAnns(coco.getAnnIds(imgIds=[ident])),key=lambda a:a['id']):
        assert not a.get('ignore',False) and not a.get('iscrowd',False),'Unexpected ignored train instance'
        full=coco.annToMask(a);assert full.shape==(1024,1024)
        m=np.ascontiguousarray(full[y:y+512,x:x+512],dtype=np.uint8)
        if m.any():
            rows.append(dict(gt_id=int(a['id']),original_area=float(a['area']),visible_area=int(m.sum()),clipped=int(m.sum())!=int(full.sum())))
            masks.append(m)
    return (np.stack(masks) if masks else np.zeros((0,512,512),np.uint8)),rows

def fit_matches(rows,targets,threshold=.5):
    if not rows or not targets:return []
    ious=mu.iou([r['segmentation'] for r in rows],[r['segmentation'] for r in targets],[0]*len(targets));used=set();pairs=[]
    for idx in sorted(range(len(rows)),key=lambda j:(-rows[j]['score'],j)):
        eligible=[j for j in range(len(targets)) if j not in used and ious[idx,j]>=threshold]
        if not eligible:continue
        j=max(eligible,key=lambda k:(ious[idx,k],-k));used.add(j)
        pairs.append(dict(prediction_index=idx,gt_id=int(targets[j]['gt_id']),iou=float(ious[idx,j])))
    return pairs

def gate(zero,current):
    return bool(current['matched50']>=zero['matched50'] and (current['matched75']>zero['matched75'] or current['matched50_iou_sum']>zero['matched50_iou_sum']+1e-9))
