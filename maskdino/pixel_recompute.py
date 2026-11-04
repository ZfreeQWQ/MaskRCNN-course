"""Recompute multi-scale pixel encoder activations in FP32."""
import torch
from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
def install(model):
    module=model.sem_seg_head.pixel_decoder;original=module.forward_features
    def forward(*args,**kwargs):
        if module.training and torch.is_grad_enabled():
            with set_checkpoint_early_stop(False):return checkpoint(original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
        return original(*args,**kwargs)
    module.forward_features=forward
def check_and_install(model):
    module=model.sem_seg_head.pixel_decoder;model.train()
    with torch.no_grad():features=model.backbone(torch.randn(1,10,256,256,device=model.device))
    saved_cpu=torch.get_rng_state();saved_cuda=torch.cuda.get_rng_state_all()
    def flatten(x):
        if isinstance(x,torch.Tensor):return [x]
        if isinstance(x,(tuple,list)):return sum((flatten(v) for v in x),[])
        return []
    results=[]
    for changed in (False,False,True):
        if changed:install(model)
        torch.set_rng_state(saved_cpu);torch.cuda.set_rng_state_all(saved_cuda);module.zero_grad(set_to_none=True)
        values=flatten(module.forward_features(features,None));sum(v.square().mean() for v in values).backward()
        results.append(([v.detach().cpu() for v in values],{k:p.grad.detach().cpu() for k,p in module.named_parameters() if p.grad is not None}));del values
    assert all(torch.allclose(a,b,rtol=1e-5,atol=1e-7) for a,b in zip(results[0][0],results[2][0]))
    noise=change=0.
    for name,base in results[0][1].items():
        ref=float((base-results[1][1][name]).norm()/base.norm().clamp_min(1e-30))
        diff=float((base-results[2][1][name]).norm()/base.norm().clamp_min(1e-30))
        assert diff<=max(1e-5,3*ref+1e-8),(name,ref,diff)
        noise=max(noise,ref);change=max(change,diff)
    module.zero_grad(set_to_none=True)
    return dict(outputs_close=True,gradients_within_baseline_numerical_tolerance=True,max_reference_repeat_relative_l2=noise,max_checkpoint_relative_l2=change,relative_tolerance='max(1e-5,3*reference_repeat+1e-8)',input_side=256,precision='FP32')
