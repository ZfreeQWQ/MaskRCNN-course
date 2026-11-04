"""Sample identical GT points in chunks, avoiding dense repeated DN target masks."""
import inspect,textwrap,types,hashlib
import torch

def install(model):
    criterion=model.criterion;original=type(criterion).loss_masks
    source=textwrap.dedent(inspect.getsource(original))
    for line in ('        target_masks = target_masks.to(src_masks)\n','        target_masks = target_masks[tgt_idx]\n','        target_masks = target_masks[:, None]\n'):
        needle=textwrap.dedent(line).strip();assert source.count(needle)==1;source=source.replace(needle,'# GT selection/casting deferred to bounded chunks.')
    old='''point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)'''
    # getsource dedents the method by four spaces.
    old=textwrap.dedent(old)
    start=source.index('        point_labels = point_sample(')
    end=source.index(').squeeze(1)',start)+len(').squeeze(1)')
    replacement='''        parts=[]
        for begin in range(0,len(point_coords),8):
            selected=target_masks[tuple(i[begin:begin+8] for i in tgt_idx)].to(src_masks)[:,None]
            parts.append(point_sample(selected,point_coords[begin:begin+8],align_corners=False).squeeze(1))
        point_labels=torch.cat(parts,dim=0) if parts else src_masks.new_empty((0,self.num_points))'''
    source=source[:start]+replacement+source[end:]
    scope={};exec(compile(source,'<chunked_target_loss>','exec'),original.__globals__,scope)
    criterion.loss_masks=types.MethodType(scope['loss_masks'],criterion)
    return dict(vendor_loss_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),patched_loss_sha256=hashlib.sha256(source.encode()).hexdigest(),target_chunk=8)

def check_and_install(model):
    torch.manual_seed(749);torch.cuda.manual_seed_all(749)
    device=torch.device("cpu");seed=torch.randn(1,20,32,32,device=device)
    targets=[{'masks':torch.rand(7,128,128,device=device)>.5}]
    indices=[(torch.arange(18,device=device),torch.arange(18,device=device)%7)]
    saved=torch.get_rng_state();results=[]
    for changed in (False,True):
        if changed:meta=install(model)
        torch.set_rng_state(saved);m=seed.clone().requires_grad_()
        values=model.criterion.loss_masks({'pred_masks':m},targets,indices,18)
        sum(values.values()).backward();results.append(({k:float(v.detach()) for k,v in values.items()},m.grad.detach().clone()))
    assert results[0][0]==results[1][0],results[0][0]
    delta=(results[0][1]-results[1][1]);relative=float(delta.norm()/results[0][1].norm().clamp_min(1e-30))
    assert torch.allclose(results[0][1],results[1][1],rtol=1e-5,atol=1e-8),(float(delta.abs().max()),relative)
    return dict(**meta,losses_bitwise_equal=True,gradients_close=True,gradient_max_abs_difference=float(delta.abs().max()),gradient_relative_l2=relative,repeated_target_indices_tested=True,check_device="cpu")
