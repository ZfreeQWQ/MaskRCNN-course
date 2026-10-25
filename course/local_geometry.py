"""Label-free local depth/normal contrast, thin-edge preserving P2 residual."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead


@torch.no_grad()
def relative_geometry_features(geometry, mean, std, size, reliable):
    """Derive finite10ch maps; geometry channel6 is binary validity, not z-score."""
    valid = geometry[:,6:7] > .5
    raw = (geometry[:,:6] * std[:,:6] + mean[:,:6]) * valid
    z,height,normals,edge = raw[:,:1],raw[:,1:2],raw[:,2:5],raw[:,5:6]
    normal_good = (normals.square().sum(1,keepdim=True) > .25) & valid
    pair_x = valid[:,:,:,:-1] & valid[:,:,:,1:]
    pair_y = valid[:,:,:-1,:] & valid[:,:,1:,:]
    dx = (z[:,:,:,1:] - z[:,:,:,:-1]) / z[:,:,:,:-1].clamp_min(1e-4)
    dy = (z[:,:,1:,:] - z[:,:,:-1,:]) / z[:,:,:-1,:].clamp_min(1e-4)
    dx = F.pad(dx * pair_x,(0,1,0,0));dy = F.pad(dy * pair_y,(0,0,0,1))
    nx = (normals[:,:,:,1:] - normals[:,:,:,:-1]).square().sum(1,keepdim=True).sqrt()
    ny = (normals[:,:,1:,:] - normals[:,:,:-1,:]).square().sum(1,keepdim=True).sqrt()
    normal_pair_x,normal_pair_y = pair_x,pair_y
    if reliable:
        normal_pair_x = normal_pair_x & normal_good[:,:,:,:-1] & normal_good[:,:,:,1:]
        normal_pair_y = normal_pair_y & normal_good[:,:,:-1,:] & normal_good[:,:,1:,:]
    nx=F.pad(nx*normal_pair_x,(0,1,0,0));ny=F.pad(ny*normal_pair_y,(0,0,0,1))
    hx=F.pad((height[:,:,:,1:]-height[:,:,:,:-1]).abs()*pair_x,(0,1,0,0))
    hy=F.pad((height[:,:,1:,:]-height[:,:,:-1,:]).abs()*pair_y,(0,0,0,1))
    coverage=F.avg_pool2d(valid.float(),3,stride=1,padding=1)
    signals=torch.cat([dx.relu()/.05,(-dx).relu()/.05,dy.relu()/.05,(-dy).relu()/.05,
                       nx/2,ny/2,edge.clamp_min(0)/.05,torch.maximum(hx,hy)/.005],1).clamp(0,5)
    if reliable:
        # Depth edges remain useful at occlusion boundaries; suppress missing
        # depth support, NOT every high-contrast edge. Unreliable normals alone
        # are excluded by the normal-pair validity rule above.
        signals=signals*coverage
    contrast=F.adaptive_max_pool2d(signals,size)  # don't average away1px fragments
    quality=F.adaptive_avg_pool2d(torch.cat([valid.float(),normal_good.float()],1),size)
    return torch.cat([contrast,quality],1)


class LocalGeometryResidual(nn.Module):
    def __init__(self, mask_channels, mean, std, channels=32, reliable=False):
        super().__init__()
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32).view(1,7,1,1),persistent=False)
        self.register_buffer("std",torch.as_tensor(std,dtype=torch.float32).view(1,7,1,1),persistent=False)
        self.reliable=reliable
        self.features=nn.Sequential(nn.Conv2d(10,channels,3,padding=1,bias=False),
            nn.GroupNorm(8,channels),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False),
            nn.GroupNorm(8,channels),nn.GELU())
        self.output=nn.Conv2d(channels,mask_channels,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.last_diagnostics={}

    def forward(self,low,geometry):
        if geometry is None:raise ValueError("Local geometry requires aligned7ch input")
        context=relative_geometry_features(geometry,self.mean,self.std,low.shape[-2:],self.reliable)
        residual=self.output(self.features(context))
        self.last_diagnostics=dict(residual_rms=residual.detach().square().mean().sqrt(),
            valid_fraction=context[:,8].detach().mean(),normal_reliable_fraction=context[:,9].detach().mean())
        return low+residual


@SEM_SEG_HEADS_REGISTRY.register()
class LocalGeometryHead(MaskFormerHead):
    @configurable
    def __init__(self, *, geometry_mean,geometry_std,local_mode,local_channels,**kwargs):
        super().__init__(**kwargs)
        assert local_mode in ("contrast","reliable")
        with torch.random.fork_rng(devices=[]):
            self.local_geometry=LocalGeometryResidual(self.predictor.mask_embed.layers[-1].out_features,
                geometry_mean,geometry_std,local_channels,local_mode=="reliable")

    @classmethod
    def from_config(cls,cfg,input_shape):
        args=super().from_config(cfg,input_shape)
        statistics=json.loads(Path("data_cache/geometry_v2_stats.json").read_text())
        assert statistics["train_only"]
        args.update(geometry_mean=statistics["mean"],geometry_std=statistics["std"],
                    local_mode=cfg.COURSE.LOCAL.MODE,local_channels=cfg.COURSE.LOCAL.CHANNELS)
        return args

    def layers(self,features,mask=None):
        assert self.transformer_in_feature=="multi_scale_pixel_decoder"
        low,_,multi=self.pixel_decoder.forward_features(features)
        low=self.local_geometry(low,features.get("query_geometry"))
        return self.predictor(multi,low,mask)
