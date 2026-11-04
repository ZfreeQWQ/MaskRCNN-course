"""Avoid redundant full-resolution mask stack in official MaskDINO DN splitting.

Preserves every decoder-layer mask and loss, using per-layer views instead of
allocating a second stacked mask tensor. No changes to installed vendor files.
"""
import inspect,textwrap,types,hashlib
import torch

def split_views(self,outputs_class,outputs_coord,mask_dict,outputs_mask):
    n=mask_dict['pad_size'];assert n>0
    kc=outputs_class[:,:,:n,:];c=outputs_class[:,:,n:,:]
    kb=outputs_coord[:,:,:n,:];b=outputs_coord[:,:,n:,:]
    km=[m[:,:n] for m in outputs_mask];m=[m[:,n:] for m in outputs_mask]
    mask_dict['output_known_lbs_bboxes']=dict(pred_logits=kc[-1],pred_boxes=kb[-1],pred_masks=km[-1],aux_outputs=self._set_aux_loss(kc,km,kb))
    return c,b,m

def install(model):
    decoder=model.sem_seg_head.predictor
    original=type(decoder).forward;source=textwrap.dedent(inspect.getsource(original))
    needle='predictions_mask=torch.stack(predictions_mask)'
    assert source.count(needle)==1
    source=source.replace(needle,'# Masks remain per-layer tensors; split_views avoids a duplicate stack.')
    scope={};exec(compile(source,'<mask_memory_forward>','exec'),original.__globals__,scope)
    decoder.forward=types.MethodType(scope['forward'],decoder)
    decoder.dn_post_process=types.MethodType(split_views,decoder)
    return dict(vendor_forward_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),patched_forward_sha256=hashlib.sha256(source.encode()).hexdigest())

def check_split(decoder):
    # Compare all retained/denoising outputs and gradients to official stack path.
    torch.manual_seed(721)
    seeds=[torch.randn(3,1,7,2),torch.randn(3,1,7,4),torch.randn(3,1,7,5,6)]
    def flatten(x):
        if isinstance(x,torch.Tensor):return [x]
        if isinstance(x,dict):return sum((flatten(x[k]) for k in sorted(x)),[])
        if isinstance(x,(tuple,list)):return sum((flatten(v) for v in x),[])
        return []
    records=[]
    for patched in (False,True):
        c,b,m=[x.clone().requires_grad_() for x in seeds];state={'pad_size':2}
        out=split_views(decoder,c,b,state,list(m)) if patched else type(decoder).dn_post_process(decoder,c,b,state,m)
        normalized=(out[0],out[1],list(out[2]),state)
        values=flatten(normalized);sum((i+1)*v.square().sum() for i,v in enumerate(values)).backward()
        records.append(([v.detach().clone() for v in values],[x.grad.clone() for x in (c,b,m)]))
    assert all(torch.equal(a,b) for a,b in zip(records[0][0],records[1][0]))
    assert all(torch.equal(a,b) for a,b in zip(records[0][1],records[1][1]))
    return dict(outputs_bitwise_equal=True,gradients_bitwise_equal=True)
