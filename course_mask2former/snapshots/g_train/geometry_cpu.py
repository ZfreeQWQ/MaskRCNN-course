"""CPU-only physical-input and common-recipe checks; no test metadata."""
import copy,json,random,time,hashlib
from pathlib import Path
import numpy as np

def prepare(w):
    import mmcv
    from geometry_signal import geometry
    from rgbd.course_geometry_data import load_index
    root=w.ROOT/'data/course_blender_20261013/prepared';images={};train=[];ann_sha={}
    for split in ('train','val'):
        path=root/'annotations'/('instances_'+split+'.json');ann_sha[split]=w.sha(path);doc=w.read(path)
        for im in doc['images']:
            assert im['depth_semantics']=='camera_axis_z'
            camera=im['camera'];assert camera['pixel_center_convention']=='integer'
            depth=(root/im['depth_file_name']).resolve();rgb=(root/im['file_name']).resolve()
            assert depth.is_relative_to(root/'depth'/split) and rgb.is_relative_to(root/'rgb'/split)
            identity=int(im['id']);assert identity not in images
            images[identity]=dict(id=identity,K=camera['K'],depth_scale_m=im['depth_scale_m'],depth_invalid_code=im['depth_invalid_code'],depth_file=str(depth),img_path=str(rgb),height=im['height'],width=im['width'],split=split)
            if split=='train':train.append(identity)
    assert len(train)==14432 and len(images)==17016
    w.save(w.OUT/'geometry_index.json',dict(schema='train_val_K_depth_only_v1',images=images,annotation_SHA=ann_sha,no_pose_or_support_or_objects=True))
    selected=sorted(train,key=lambda x:hashlib.sha256(('G-normalization-'+str(x)).encode()).hexdigest())[:256]
    sums=np.zeros(7,np.float64);squares=np.zeros(7,np.float64);count=0;records=[]
    for i,identity in enumerate(selected):
        w.w.deadline(w.spec.CPU_END);v=images[identity];raw=mmcv.imread(v['depth_file'],flag='unchanged');valid=(raw!=v['depth_invalid_code'])&(raw>0);z=raw.astype(np.float32)*v['depth_scale_m'];g,a=geometry(z,valid,v['K']);flat=g[valid]
        assert len(flat)>0;flat=flat[np.linspace(0,len(flat)-1,min(4096,len(flat)),dtype=int)].astype(np.float64)
        sums+=flat.sum(0);squares+=(flat*flat).sum(0);count+=len(flat)
        records.append(dict(image_id=identity,depth_sha256=w.sha(v['depth_file']),geometry_sha256=hashlib.sha256(g.tobytes()).hexdigest(),**a))
        if i%8==0:w.w.mark('geometry_train_only_normalization',images=i+1,total=256)
    empirical_mean=sums/count;std=np.sqrt(np.maximum(squares/count-empirical_mean**2,1e-8))
    # Height/normal/edge zero has physical meaning; preserve it. Only Z is centered.
    mean=[float(empirical_mean[0]),0.,0.,0.,0.,0.,0.];scale=[float(max(std[0],.1)),float(max(std[1],.01)),1.,1.,1.,float(max(std[5],.01)),1.]
    w.save(w.OUT/'geometry_normalization.json',dict(status='passed',mean=mean,std=scale,fit_images=selected,samples=count,records=records,training_only=True,empirical_mean=empirical_mean.tolist(),empirical_std=std.tolist(),normal_frame='source_camera',plane='depth_only_dominant_plane_64_fixed_RANSAC_5mm_40percent_support',dataset_index_sha256=w.sha(w.OUT/'geometry_index.json')))

def actual_checks(w,old,new,oldcfg,cfg):
    import torch
    from mmengine.dataset import Compose,pseudo_collate
    from mmdet.registry import DATASETS
    from rgbd.course_geometry_data import adapt_pipeline
    from course_resolution_probe import safe_request
    # The exact same original augmentation draws, RGB-D, GT and CP choices.
    oldds=DATASETS.build(oldcfg.train_dataloader.dataset);newds=DATASETS.build(cfg.train_dataloader.dataset);records=[]
    for index in range(16):
        seed=20261013+index
        def one(ds):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);sample=ds.prepare_data(index);return sample,torch.get_rng_state(),np.random.get_state(),random.getstate()
        a,ar,an,ap=one(oldds);b,br,bn,bp=one(newds)
        assert (a is None)==(b is None)
        if a is not None:
            assert torch.equal(a['inputs'],b['inputs']) and torch.equal(ar,br) and np.array_equal(an[1],bn[1]) and an[2:]==bn[2:] and ap==bp
            sa,sb=a['data_samples'],b['data_samples'];assert sa.metainfo==sb.metainfo
            assert np.array_equal(sa.gt_instances.masks.masks,sb.gt_instances.masks.masks)
            assert torch.equal(sa.gt_instances.labels,sb.gt_instances.labels)
            assert sb.geometry.data.shape[0]==7 and torch.isfinite(sb.geometry.data).all()
            records.append(dict(index=index,image_id=int(sb.img_id),copy_paste=sb.metainfo.get('course_copy_paste_activity'),geometry_shape=list(sb.geometry.data.shape)))
        w.w.mark('geometry_augmentation_parity',images=index+1,total=16)
    assert len(records)==16 and sum(x['copy_paste']['pasted_instances'] for x in records)>0
    w.save(w.OUT/'augmentation_binding.json',dict(status='passed',records=records,original_RGB_depth_GT_RNG_equal=True))
    del oldds,newds
    # Actual pretrained full detector prediction equality; no labels in requests.
    old.eval();new.eval();doc=w.read(oldcfg.val_evaluator.ann_file);rows=[]
    for im in doc['images'][:2]:
        p=[copy.deepcopy(dict(t)) for t in oldcfg.val_dataloader.dataset.pipeline if t['type']!='LoadAnnotations'];p[1]['scale']=(128,128)
        pa=Compose(p);pb=Compose(adapt_pipeline(p,str(w.OUT/'geometry_index.json')));request=safe_request(oldcfg.val_dataloader.dataset.data_root,im)
        samples=[pa(copy.deepcopy(request)),pb(copy.deepcopy(request))]
        for sample in samples:
            if 'gt_instances' in sample['data_samples']:assert len(sample['data_samples'].gt_instances)==0;del sample['data_samples'].gt_instances
        with torch.inference_mode():
            torch.manual_seed(5);a=old.test_step(pseudo_collate([samples[0]]))[0].pred_instances;rng=torch.get_rng_state().clone()
            torch.manual_seed(5);b=new.test_step(pseudo_collate([samples[1]]))[0].pred_instances
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(getattr(a,k),getattr(b,k)) for k in ('masks','scores','labels','bboxes'))
        rows.append(dict(image_id=im['id'],all_predictions_equal=True,GT_sent=False,input_size=128))
    # Original ordinary + pilot/noise + matching loss and global RNG at zero residual.
    from mmengine.structures import InstanceData
    from mmdet.structures.mask import BitmapMasks
    processed=[]
    for model,sample in ((old,samples[0]),(new,samples[1])):
        data=model.data_preprocessor(pseudo_collate([sample]),training=False);ds=data['data_samples'][0]
        h,wid=data['inputs'].shape[-2:];masks=np.zeros((2,h,wid),np.uint8);masks[0,7,9]=1;masks[1,20:60,20:60]=1
        ds.gt_instances=InstanceData(labels=torch.zeros(2,dtype=torch.long),masks=BitmapMasks(masks,h,wid));processed.append(data)
    old.train();new.train();torch.manual_seed(77)
    a=old(**processed[0],mode='loss');rng=torch.get_rng_state().clone();torch.manual_seed(77);b=new(**processed[1],mode='loss')
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) and torch.isfinite(b[k]).all() for k in a) and torch.equal(rng,torch.get_rng_state())
    total=sum(b.values());total.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.geometry_fusion.parameters())
    assert sum(float(p.weight.grad.abs().sum()) for p in new.geometry_fusion.projections)>0
    old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
    w.save(w.OUT/'initial_prediction_parity.json',dict(status='passed',images=rows,original_COCO=True,zero_residual=True,RNG_equal=True,ordinary_and_auxiliary_loss_equal=True,geometry_projection_gradient_finite_nonzero=True,CPU_only=True))
