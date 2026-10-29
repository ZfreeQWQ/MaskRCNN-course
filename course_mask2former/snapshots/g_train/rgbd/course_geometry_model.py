"""Seven-channel geometry plus four-scale spatial/channel residual fusion."""
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from rgbd.multiscale import block
from rgbd.course_tiny_matching import TinyMatchedRGBDMask2Former

class GeometryFusion(nn.Module):
    def __init__(self):
        super().__init__();widths=(32,64,128,256);outs=(256,512,1024,2048)
        self.stem=nn.Sequential(block(7,16,2),block(16,32,2),block(32,32))
        self.stages=nn.ModuleList([nn.Sequential(block(a,b,2),block(b,b)) for a,b in zip(widths,widths[1:])])
        self.rgb_context=nn.ModuleList([nn.Conv2d(c,16,1) for c in outs])
        self.spatial=nn.ModuleList([nn.Conv2d(c+16,1,1) for c in widths])
        self.channel=nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c+16,max(16,c//4),1),nn.SiLU(),nn.Conv2d(max(16,c//4),c,1)) for c in widths])
        self.projections=nn.ModuleList([nn.Conv2d(c,d,1,bias=False) for c,d in zip(widths,outs)])
        for p in self.projections:nn.init.zeros_(p.weight)
        for p,c in zip(self.spatial,self.channel):nn.init.zeros_(p.weight);nn.init.zeros_(p.bias);nn.init.zeros_(c[-1].weight);nn.init.zeros_(c[-1].bias)
    def forward(self,rgb,g,keep):
        x=self.stem(g);levels=[x]
        for stage in self.stages:x=stage(x);levels.append(x)
        output=[]
        for i,(r,x) in enumerate(zip(rgb,levels)):
            x=F.interpolate(x,r.shape[-2:],mode='bilinear',align_corners=False);joint=torch.cat((x,self.rgb_context[i](r)),1)
            residual=self.projections[i](x*(2*torch.sigmoid(self.channel[i](joint))))*(2*torch.sigmoid(self.spatial[i](joint)))
            output.append(r+residual*keep)
        return tuple(output)

@MODELS.register_module()
class GeometryMatchedMask2Former(TinyMatchedRGBDMask2Former):
    def __init__(self,*args,geometry_options=None,**kw):
        super().__init__(*args,**kw);assert geometry_options==dict(enabled=True,dropout=.2,seed=20261013)
        self.geometry_options=dict(geometry_options)
        with torch.random.fork_rng(devices=[]):torch.manual_seed(64027);self.geometry_fusion=GeometryFusion()
        self.geometry_activity=dict(train_calls=0,train_samples=0,dropped_samples=0,eval_calls=0)
        self._geometry=None
    def forward(self,inputs,data_samples=None,mode='tensor'):
        assert data_samples and all('geometry' in x for x in data_samples)
        self._geometry=torch.stack([x.geometry.data for x in data_samples])
        try:return super().forward(inputs,data_samples,mode)
        finally:self._geometry=None
    def extract_feat(self,inputs):
        base=super().extract_feat(inputs)
        if not self.geometry_options['enabled']:return base
        assert self._geometry is not None and self._geometry.shape[:2]==(inputs.shape[0],7)
        keep=inputs.new_ones((inputs.shape[0],1,1,1))
        if self.training:
            a=self.geometry_activity;gen=torch.Generator(device='cpu').manual_seed(self.geometry_options['seed']+999983*a['train_calls'])
            k=torch.rand(inputs.shape[0],generator=gen)>=self.geometry_options['dropout'];keep=k.to(inputs.device,inputs.dtype)[:,None,None,None]
            a['train_calls']+=1;a['train_samples']+=inputs.shape[0];a['dropped_samples']+=int((~k).sum())
        else:self.geometry_activity['eval_calls']+=1
        return self.geometry_fusion(base,self._geometry,keep)
