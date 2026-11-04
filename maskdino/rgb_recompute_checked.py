"""Recompute RGB residual stages during backward, retaining FP32 parameters/losses."""
import torch
from torch.utils.checkpoint import checkpoint

def install(model):
    rgb=model.backbone.rgb
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats for m in rgb.modules()),'Running-stat BatchNorm cannot be recomputed without extra handling'
    for name in ('res2','res3','res4','res5'):
        stage=getattr(rgb,name);original=stage.forward
        def forward(x,original=original,stage=stage):
            if stage.training and torch.is_grad_enabled():return checkpoint(original,x,use_reentrant=False,preserve_rng_state=True)
            return original(x)
        stage.forward=forward

def check_and_install(model):
    old=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    rgb=model.backbone.rgb;rgb.train();x=torch.randn(1,3,256,256,device=model.device)
    results=[]
    for changed in (False,True):
        if changed:install(model)
        rgb.zero_grad(set_to_none=True);out=rgb(x);loss=sum(y.square().mean() for y in out.values());loss.backward()
        results.append(({k:v.detach().cpu() for k,v in out.items()},{k:p.grad.detach().cpu() for k,p in rgb.named_parameters() if p.grad is not None}))
        del out,loss
    assert results[0][0].keys()==results[1][0].keys() and results[0][1].keys()==results[1][1].keys()
    for group in (0,1):
        for key in results[0][group]:assert torch.allclose(results[0][group][key],results[1][group][key],rtol=1e-5,atol=1e-7),(group,key)
    rgb.zero_grad(set_to_none=True)
    torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=old
    return dict(real_RGB_forward_and_parameter_gradients_match=True,rtol=1e-5,atol=1e-7,stages=['res2','res3','res4','res5'],precision='FP32')
