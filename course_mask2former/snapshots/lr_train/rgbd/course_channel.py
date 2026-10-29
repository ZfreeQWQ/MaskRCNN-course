"""Opt-in identity-initialized channel calibration on existing depth residuals."""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.registry import MODELS
from rgbd.multiscale import SpatialFusion,MultiScaleRGBDMask2Former

class ChannelSpatialFusion(SpatialFusion):
    def attach_channels(self):
        if hasattr(self,'channel_mlps'):raise ValueError('Already attached')
        # Preserve all inherited tensors and the caller's subsequent RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.channel_mlps=nn.ModuleList([nn.Sequential(nn.Linear(2*p.out_channels,16),
                nn.SiLU(),nn.Linear(16,p.out_channels)) for p in self.projections])
            for mlp in self.channel_mlps:
                nn.init.zeros_(mlp[-1].weight);nn.init.zeros_(mlp[-1].bias)
        self.channel_calls={'train':0,'eval':0,'train_samples':0,'eval_samples':0}

    def forward(self,rgb_features,planes,cue,use_gate):
        if len(rgb_features)!=4 or planes.shape[1]!=2 or cue.shape[1]!=1 or not use_gate:
            raise ValueError('Calibrated arm requires four RGB-D levels and original spatial gate')
        x=self.stem(planes);levels=[x]
        for stage in self.stages:x=stage(x);levels.append(x)
        result=[]
        for i,(rgb,depth) in enumerate(zip(rgb_features,levels)):
            if rgb.shape[1]!=self.projections[i].out_channels:raise ValueError('Channel mismatch')
            depth=F.interpolate(depth,rgb.shape[-2:],mode='bilinear',align_corners=False)
            residual=self.projections[i](depth)
            pooled=torch.cat((rgb.mean((2,3)),residual.mean((2,3))),1)
            channel=2*torch.sigmoid(self.channel_mlps[i](pooled))[:,:,None,None]
            residual=residual*channel
            measured=F.interpolate(cue,rgb.shape[-2:],mode='area')
            context=torch.tanh(rgb.mean(1,keepdim=True))
            spatial=2*torch.sigmoid(self.gates[i](torch.cat([depth,context,measured],1)))
            result.append(rgb+residual*spatial)
        mode='train' if self.training else 'eval'
        self.channel_calls[mode]+=1;self.channel_calls[mode+'_samples']+=planes.shape[0]
        return tuple(result)

@MODELS.register_module()
class ChannelRGBDMask2Former(MultiScaleRGBDMask2Former):
    def __init__(self,*args,course_channel=None,**kwargs):
        opts=dict(course_channel or {})
        if set(opts)-{'enabled'} or type(opts.get('enabled',False)) is not bool:
            raise ValueError('Only a boolean enabled switch')
        enabled=opts.get('enabled',False)
        super().__init__(*args,**kwargs)
        if enabled:
            if self.multiscale_options!=dict(enabled=True,reliability_gate=True,source='depth'):
                raise ValueError('Original depth+spatial-gate base is required')
            # Add modules to the existing fusion; do not reconstruct or redraw it.
            self.spatial_fusion.__class__=ChannelSpatialFusion
            self.spatial_fusion.attach_channels()
