"""Independent gated residual geometry pyramid for our existing MaskDINO.

References: SA-Gate/CMX multimodal feature fusion and teammate RGB-D audits.
This is an adaptation of shared ideas, not an exact paper reproduction or a
claim of novel architecture. No teammate implementation is imported.
"""
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.modeling.backbone import Backbone

def conv(cin,cout,stride):
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False),nn.GroupNorm(8,cout),nn.GELU())

class GeometryBackbone(Backbone):
    def __init__(self,rgb):
        super().__init__();self.rgb=rgb
        self.levels=('res2','res3','res4','res5');self.dropout=.2
        self.entry=conv(7,24,2)
        self.stages=nn.ModuleList();self.outputs=nn.ModuleDict();self.gates=nn.ModuleDict()
        previous=24
        for name,width in zip(self.levels,(32,64,128,256)):
            self.stages.append(nn.Sequential(conv(previous,width,2),conv(width,width,1)))
            channels=rgb.output_shape()[name].channels
            self.outputs[name]=nn.Conv2d(width,channels,1)
            nn.init.zeros_(self.outputs[name].weight);nn.init.zeros_(self.outputs[name].bias)
            self.gates[name]=nn.Sequential(nn.Conv2d(channels+width,16,1),nn.GELU(),nn.Conv2d(16,1,1),nn.Sigmoid())
            previous=width
        self.last_activity={};self.dropout_calls=0
    @property
    def size_divisibility(self):return self.rgb.size_divisibility
    @property
    def padding_constraints(self):return self.rgb.padding_constraints
    def output_shape(self):return self.rgb.output_shape()
    def forward(self,x):
        assert x.shape[1]==10
        features=self.rgb(x[:,:3]);g=self.entry(x[:,3:]);keep=g.new_ones((len(g),1,1,1))
        if self.training:
            generator=torch.Generator(device=g.device).manual_seed(20261102+self.dropout_calls)
            keep=(torch.rand(keep.shape,device=g.device,generator=generator)>=self.dropout).to(g);self.dropout_calls+=1
        activity={}
        for level,stage in zip(self.levels,self.stages):
            g=stage(g);rgb=features[level]
            aligned=F.interpolate(g,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
            gate=self.gates[level](torch.cat((rgb,aligned),1))
            delta=self.outputs[level](aligned)*gate*keep
            features[level]=rgb+delta
            activity[level]=dict(residual_rms=delta.detach().square().mean().sqrt(),rgb_rms=rgb.detach().square().mean().sqrt(),gate=gate.detach().mean())
        self.last_activity=activity
        return features

def install(model):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261102)
        model.backbone=GeometryBackbone(model.backbone).to(model.device)
    model.pixel_mean=torch.cat((model.pixel_mean,model.pixel_mean.new_zeros((7,1,1))))
    model.pixel_std=torch.cat((model.pixel_std,model.pixel_std.new_ones((7,1,1))))
    return model
