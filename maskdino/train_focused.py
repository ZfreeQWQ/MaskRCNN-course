"""Continue validated2048 weights for1500 updates on the next3000 training images."""
import argparse,json,math,os,sys,time,traceback
from pathlib import Path
import settings
B=settings.WORKDIR;ROOT=B/'training'
import geometry_maskdino as r
from focused_data import GeometryData
from mask_memory import install as mask_install
from rgb_recompute_checked import install as rgb_install
from mask_target_chunks import install as target_install
from pixel_recompute import install as pixel_install
t,d,torch=r.t,r.d,r.torch
CK=settings.WEIGHTS/'parent.pth';STATS=settings.GEOMETRY_STATS

def main(a):
    assert a.arm=='highres' and a.phase=='train'
    side=2048
    out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    source_hashes={p.name:d.sha(p) for p in Path(__file__).parent.glob('*.py')}
    state=dict(status='initializing',pid=os.getpid(),arm=a.arm,phase=a.phase,training_side=side,evaluation_side=2048,precision='FP32',planned_updates=4 if a.phase=='gate' else 1500,completed_updates=0,source_hashes=source_hashes,parent_checkpoint=str(CK),parent_sha256=d.sha(CK),geometry_stats_sha256=d.sha(STATS),history=[],policy='Continue focused crop checkpoint and AdamW for1500 updates; train-only small-instance enriched native640 crops resized2048, fixed parent global2048 predictions; gated validation then fullVAL+TEST; validation selection only.')
    def write():d.atomic_json(out/'run.json',state)
    write()
    step=0;stage='initializing';batch_rng=None;batch_dropout=None
    try:
        assert state['parent_sha256']=='9f26b613b6eaef04ac23350c56890f8bf57caa199f715699d12b734570c54ec7'
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.66)
        model,cfg=r.build(out,'geometry');parent=torch.load(CK,map_location='cpu',weights_only=False);model.load_state_dict(parent['model'],strict=True)
        parent_rng=parent['rng'];parent_optimizer=parent['optimizer'];parent_dropout=parent['run_state']['dropout_calls'];del parent
        state['mask_memory']=mask_install(model);rgb_install(model);state['target_chunking']=target_install(model);pixel_install(model)
        opt,rates,_=r.make_optimizer(model,cfg);rates=[min(x,1e-5) for x in rates]
        opt.load_state_dict(parent_optimizer);del parent_optimizer
        model.backbone.dropout_calls=parent_dropout
        train=GeometryData('train','geometry',STATS,side=side);train.epoch=4
        eligible=[i for i in range(len(train)) if any(0<a['area']<=1024 for a in train.annotations[i])]
        assert len(eligible)>=3000
        perm=torch.randperm(len(eligible),generator=torch.Generator().manual_seed(20261106)).tolist()
        order=[eligible[i] for i in perm[:3000]]
        train.focus_indices=set(order)
        state['focused_training']={'native_crop':640,'crop_input_side':2048,'eligible_images':len(eligible),'crop_slots':3000,'epoch':4,'policy':'Train-only small-instance enriched sampling; all slots request focused crop; raster-empty fallback retained'}
        state['training_order']={'seed':20261106,'images':len(train),'indices_sha256':__import__('hashlib').sha256(json.dumps(order).encode()).hexdigest()}
        d.atomic_json(out/'training_indices.json',order)
        if a.phase=='gate':
            # Include dense scenes before spending a formal training budget.
            dense=sorted(range(len(train)),key=lambda i:len(train.annotations[i]),reverse=True)[:4]
            order=dense+order[:4]
        r.restore_rng(parent_rng);del parent_rng;step=0
        if a.resume:
            ck=torch.load(out/'latest.pth',map_location='cpu',weights_only=False);prior=ck['run_state']
            assert prior['source_hashes']==source_hashes and prior['training_side']==side
            model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);step=ck['step'];r.restore_rng(ck['rng']);model.backbone.dropout_calls=prior['dropout_calls'];state['completed_updates']=step;del ck
        loader=torch.utils.data.DataLoader(train,batch_sampler=[order[i:i+2] for i in range(step*2,len(order),2)],collate_fn=d.collate,num_workers=4,prefetch_factor=1,generator=torch.Generator().manual_seed(8123))
        model.train();start=time.monotonic();state['status']='training';write()
        for batch in loader:
            batch_rng=r.rng_state();batch_dropout=model.backbone.dropout_calls;stage='forward_backward'
            scale=(.1+.9*min(1.,step/100))*(.1+.9*.5*(1+math.cos(math.pi*step/1500)))
            for group,rate in zip(opt.param_groups,rates):group['lr']=rate*scale
            opt.zero_grad(set_to_none=True);total=0.
            for sample in batch:
                assert tuple(sample['image'].shape)==(10,2048,2048)
                key='focused_samples' if sample.get('_focused_crop',False) else 'full_samples';state[key]=state.get(key,0)+1
                losses=model([sample]);loss=sum(losses.values())/len(batch);assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());del losses,loss
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.01,error_if_nonfinite=True);assert norm>0
            stage='optimizer_step';opt.step();step+=1;stage='completed_update'
            stop=(out/'STOP').exists() or (ROOT/'STOP').exists()
            if step%20==0 or step==state['planned_updates'] or a.phase=='gate' or stop:
                state.update(completed_updates=step,loss=total,gradient_norm=float(norm),seconds=time.monotonic()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,dropout_calls=model.backbone.dropout_calls);write();print('TRAIN',a.arm,step,'/',state['planned_updates'],flush=True)
            if step==1 or step%50==0 or step==1500 or stop:
                state.update(completed_updates=step,dropout_calls=model.backbone.dropout_calls,checkpoint_step=step)
                r.save(out/'latest.pth',model,opt,step,state);write()
                print('CHECKPOINT',step,flush=True)
            if stop:state['status']='stopped';write();return
        checkpoint=out/'step_1500.pth'
        if not checkpoint.exists():r.save(checkpoint,model,opt,1500,state)
        state.update(checkpoint=str(checkpoint),checkpoint_sha256=d.sha(checkpoint),evaluation='Server pipeline will compare crop inference against current best on validation subsets before full VAL+TEST')
        state['status']='completed';write()
    except torch.OutOfMemoryError:
        evidence=traceback.format_exc()
        state.update(status='oom',error=evidence,failure_stage=stage,last_complete_update=step)
        # Forward/backward cannot partially update AdamW parameters. Restore the batch-start RNG
        # and dropout counter, discard partial gradients, then save the consistent last update.
        if stage=='forward_backward' and batch_rng is not None:
            try:
                opt.zero_grad(set_to_none=True);model.backbone.dropout_calls=batch_dropout;r.restore_rng(batch_rng)
                state.update(completed_updates=step,dropout_calls=batch_dropout,checkpoint_step=step)
                r.save(out/'latest.pth',model,opt,step,state)
                state['emergency_checkpoint_saved']=True
            except Exception:state['emergency_checkpoint_error']=traceback.format_exc()
        state['recoverable_checkpoint_exists']=(out/'latest.pth').exists();write();raise
    except Exception:
        state.update(status='failed',error=traceback.format_exc());write();raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['control','highres'],required=True);p.add_argument('--phase',choices=['gate','train'],required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');main(p.parse_args())
