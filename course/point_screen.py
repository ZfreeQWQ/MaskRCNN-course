"""PointRend-inspired query-conditioned nonlinear RGB-D point residual.

Not a PointRend reproduction: use Mask2Former queries, a bounded zero-initialized
residual, original final-loss sampling, coarse-only matching/auxiliaries, and
separate native coarse interpolation to guarantee initial inference identity.
"""
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY, META_ARCH_REGISTRY
from detectron2.projects.point_rend.point_features import point_sample, get_uncertain_point_coords_with_randomness
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.criterion import SetCriterion, calculate_uncertainty, sigmoid_ce_loss_jit, dice_loss_jit
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from course.modeling_v2 import GeometryResNet, build_geometry_resnet_backbone
from course.refinement_v3 import DetailStem
from course.query_screen import SeededQueryDecoder


class PointGeometryResNet(GeometryResNet):
    def forward(self,x):
        features=super().forward(x)
        half=self.stem.half_feature
        self.stem.half_feature=None
        keep=(self.geometry_fusion.last_keep if self.geometry_fusion.enabled else x.new_zeros(len(x),1,1,1))
        features['point_rgbd']=half
        features['point_geometry']=F.interpolate(x[:,4:]*keep,size=half.shape[-2:],mode='nearest')
        return features

    def output_shape(self):
        shape=super().output_shape()
        shape['point_rgbd']=ShapeSpec(channels=self.stem.out_channels,stride=2)
        shape['point_geometry']=ShapeSpec(channels=7,stride=2)
        return shape


@BACKBONE_REGISTRY.register()
def build_point_geometry_backbone(cfg,input_shape):
    base=build_geometry_resnet_backbone(cfg,input_shape)
    model=PointGeometryResNet(stem=DetailStem(base.stem),
        stages=[list(getattr(base,n).children()) for n in base.stage_names],
        out_features=cfg.MODEL.RESNETS.OUT_FEATURES,freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
    model.geometry_fusion=base.geometry_fusion
    return model


@TRANSFORMER_DECODER_REGISTRY.register()
class PointEmbeddingDecoder(SeededQueryDecoder):
    def forward(self,x,mask_features,mask=None,query_geometry=None):
        self._point_calls=0
        self._point_embedding=None
        try:
            result=super().forward(x,mask_features,mask,query_geometry=query_geometry)
            assert self._point_embedding is not None
            result['point_embedding']=self._point_embedding
            return result
        finally:
            self._point_embedding=None

    def forward_prediction_heads(self,output,mask_features,attn_mask_target_size):
        result=super().forward_prediction_heads(output,mask_features,attn_mask_target_size)
        self._point_calls+=1
        if self._point_calls==self.num_layers+1:
            self._point_embedding=self.mask_embed(self.decoder_norm(output).transpose(0,1))
        return result


class NonlinearPointRefiner(nn.Module):
    def __init__(self,rgb_channels=64,embedding_channels=256,channels=16,hidden=64,geometry=False,gain=1.):
        super().__init__()
        if channels%4 or hidden<=0 or not 0<gain<=2:
            raise ValueError('Invalid point dimensions or residual gain')
        self.rgb=nn.Sequential(nn.Conv2d(rgb_channels,channels,1,bias=False),nn.GroupNorm(4,channels),nn.ReLU())
        self.features=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),
                                    nn.GroupNorm(4,channels),nn.ReLU())
        self.query=nn.Linear(embedding_channels,channels)
        self.mlp=nn.Sequential(nn.Conv1d(2*channels+1,hidden,1),nn.ReLU(),
                               nn.Conv1d(hidden,hidden,1),nn.ReLU(),nn.Conv1d(hidden,1,1))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
        # Add optional geometry AFTER common weights so paired candidates share
        # identical RGB, query and MLP initialization, not just zero outputs.
        self.geometry=(nn.Sequential(nn.Conv2d(7,channels,1,bias=False),nn.GroupNorm(4,channels),nn.ReLU())
                       if geometry else None)
        self.gain=gain
        self.last_diagnostics={}

    def fine_features(self,rgbd,geometry=None):
        rgbd=self.rgb(rgbd)
        if self.geometry is not None:
            if geometry is None:raise ValueError('Aligned geometry required')
            rgbd=rgbd+self.geometry(geometry)*geometry[:,6:7].clamp(0,1)
        return self.features(rgbd)

    def render(self,fine,embedding,coarse,coords,image_indices):
        """N matched queries x P points; never repeat full feature maps per GT."""
        n,p=coords.shape[:2]
        if embedding.shape[0]!=n or coarse.shape!=(n,p) or image_indices.shape!=(n,):
            raise ValueError('Point/query alignment mismatch')
        if (image_indices<0).any() or (image_indices>=len(fine)).any():
            raise ValueError('Point image index out of range')
        output=coarse.new_zeros(n,p)
        for b in range(len(fine)):
            rows=(image_indices==b).nonzero().flatten()
            if not len(rows):continue
            sampled=point_sample(fine[b:b+1],coords[rows].reshape(1,-1,2),align_corners=False)
            sampled=sampled.reshape(fine.shape[1],len(rows),p).permute(1,0,2)
            query=self.query(embedding[rows])[:,:,None].expand(-1,-1,p)
            inputs=torch.cat([sampled,query,coarse[rows,None].clamp(-12,12)/6.],1)
            output[rows]=self.gain*self.mlp(inputs).squeeze(1).tanh()
        self.last_diagnostics=dict(residual_rms=output.detach().square().mean().sqrt(),
                                    residual_mean=output.detach().mean(),
                                    residual_std=output.detach().std(unbiased=False),
                                    rendered_points=n*p)
        return output

    def refine(self,coarse,fine,embedding,points=2048,chunk=16):
        """Two subdivisions; interpolate ONLY residual, native coarse stays exact.

        At each stage select uncertain current pixels, but condition the point
        MLP on original coarse logits as in training. Residual is an absolute
        correction to the original coarse field, not added repeatedly.
        """
        if points<=0 or chunk<=0:raise ValueError('Positive point budget/chunk required')
        batch,queries=coarse.shape[:2]
        half=tuple(fine.shape[-2:]);full=tuple(2*s for s in half)
        images=[]
        for b in range(batch):
            pieces=[]
            for start in range(0,queries,chunk):
                c=coarse[b,start:start+chunk];e=embedding[b,start:start+chunk]
                delta=None
                for size in (half,full):
                    native=F.interpolate(c[None],size=size,mode='bilinear',align_corners=False)[0]
                    delta=(torch.zeros_like(native) if delta is None else
                           F.interpolate(delta[None],size=size,mode='bilinear',align_corners=False)[0])
                    h,w=size
                    ids=(-(native+delta).abs()).flatten(1).topk(min(points,h*w),dim=1).indices
                    coords=torch.stack(((ids%w+.5)/w,(ids//w+.5)/h),-1)
                    sampled=point_sample(c[:,None],coords,align_corners=False).squeeze(1)
                    residual=self.render(fine[b:b+1],e,sampled,coords,
                                          torch.zeros(len(c),device=c.device,dtype=torch.long))
                    delta=delta.flatten(1).scatter(1,ids,residual).reshape_as(native)
                pieces.append(native+delta)
            images.append(torch.cat(pieces))
        return torch.stack(images)


@SEM_SEG_HEADS_REGISTRY.register()
class PointRefinementHead(MaskFormerHead):
    @configurable
    def __init__(self,*,point_settings,rgb_channels,**kwargs):
        super().__init__(**kwargs)
        if point_settings.MODE not in ('rgbd','rgbd_geometry'):
            raise ValueError('Point head needs an explicit enabled mode')
        with torch.random.fork_rng(devices=[]):
            self.point_refiner=NonlinearPointRefiner(rgb_channels,self.predictor.mask_embed.layers[-1].out_features,
                point_settings.CHANNELS,point_settings.HIDDEN,point_settings.MODE=='rgbd_geometry',point_settings.GAIN)
        self.points=point_settings.INFERENCE_POINTS;self.chunk=point_settings.QUERY_CHUNK

    @classmethod
    def from_config(cls,cfg,input_shape):
        args=super().from_config(cfg,input_shape)
        args.update(point_settings=cfg.COURSE.POINT,rgb_channels=input_shape['point_rgbd'].channels)
        return args

    def layers(self,features,mask=None):
        low,_,multi=self.pixel_decoder.forward_features(features)
        result=self.predictor(multi,low,mask)
        embedding=result.pop('point_embedding')
        fine=self.point_refiner.fine_features(features['point_rgbd'],features['point_geometry'])
        if self.training:
            result['point_fine']=fine;result['point_embedding']=embedding
        else:
            result['pred_masks']=self.point_refiner.refine(result['pred_masks'],fine,embedding,self.points,self.chunk)
        return result


class PointCriterion(SetCriterion):
    def __init__(self,*args,renderer,**kwargs):
        super().__init__(*args,**kwargs)
        # One registered owner: sem_seg_head.point_refiner. Weak reference avoids
        # duplicate state keys/optimizer parameters. Models rebuild from config
        # and load state_dict, never pickle/deepcopy live model objects.
        self._renderer_ref=weakref.ref(renderer)

    def loss_masks(self,outputs,targets,indices,num_masks):
        if 'point_fine' not in outputs:
            return super().loss_masks(outputs,targets,indices,num_masks)
        source=self._get_src_permutation_idx(indices)
        coarse=outputs['pred_masks'][source][:,None]
        if not len(coarse):
            zero=(outputs['pred_masks'].sum()+outputs['point_fine'].sum())*0
            return dict(loss_mask=zero,loss_dice=zero)
        masks=torch.cat([t['masks'][j] for t,(_,j) in zip(targets,indices)]).to(coarse)[:,None]
        with torch.no_grad():
            coords=get_uncertain_point_coords_with_randomness(coarse,calculate_uncertainty,self.num_points,
                                                              self.oversample_ratio,self.importance_sample_ratio)
            truth=point_sample(masks,coords,align_corners=False).squeeze(1)
        logits=point_sample(coarse,coords,align_corners=False).squeeze(1)
        renderer=self._renderer_ref()
        if renderer is None:raise RuntimeError('Point refiner owner was destroyed')
        delta=renderer.render(outputs['point_fine'],outputs['point_embedding'][source],logits,coords,source[0].to(coarse.device))
        logits=logits+delta
        return dict(loss_mask=sigmoid_ce_loss_jit(logits,truth,num_masks),loss_dice=dice_loss_jit(logits,truth,num_masks))


@META_ARCH_REGISTRY.register()
class PointMaskFormer(MaskFormer):
    @classmethod
    def from_config(cls,cfg):
        assert cfg.COURSE.COVERAGE.MATCH_POINTS==cfg.COURSE.COVERAGE.LOSS_POINTS==0
        assert cfg.COURSE.QUERY.MODE=='p2'
        args=super().from_config(cfg);old=args['criterion']
        args['criterion']=PointCriterion(old.num_classes,old.matcher,old.weight_dict,old.eos_coef,
            old.losses,old.num_points,old.oversample_ratio,old.importance_sample_ratio,
            renderer=args['sem_seg_head'].point_refiner)
        return args
