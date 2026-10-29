"""Paired geometry sidecar; original five-channel RGB-D path stays unchanged."""
import copy,json
from pathlib import Path
from functools import lru_cache
import mmcv,numpy as np,torch
from mmengine.structures import PixelData
from mmdet.registry import TRANSFORMS,MODELS
from mmdet.datasets.transforms import RandomCrop
from rgbd.course_data import LoadCourseRGBDImage
from rgbd.data import RGBDResize,RGBDRandomFlip,PackRGBDDetInputs,_check_aligned
from rgbd.course_copy_paste import CourseRGBDSmallCopyPaste
from rgbd.model import RGBDDataPreprocessor
from geometry_signal import geometry

@lru_cache(maxsize=4)
def load_index(path):
    d=json.loads(Path(path).read_text());assert d['schema']=='train_val_K_depth_only_v1'
    return {int(k):v for k,v in d['images'].items()}

@TRANSFORMS.register_module()
class LoadGeometryRGBD(LoadCourseRGBDImage):
    def __init__(self,geometry_index,**kw):super().__init__(**kw);self.geometry_index=geometry_index
    def transform(self,r):
        r=super().transform(r);v=load_index(self.geometry_index)[int(r['img_id'])]
        assert Path(r['depth_file']).resolve()==Path(v['depth_file']).resolve() and r['depth_scale_m']==v['depth_scale_m']
        assert r['rgbd_depth'].shape==(v['height'],v['width'])
        r['geometry'],r['geometry_audit']=geometry(r['rgbd_depth'],r['rgbd_raw_valid'],v['K']);return r

@TRANSFORMS.register_module()
class GeometryCopyPaste(CourseRGBDSmallCopyPaste):
    def __init__(self,geometry_index,**kw):super().__init__(**kw);self.geometry_index=geometry_index
    def _load_object(self,obj):
        donor=super()._load_object(obj);image=self._images[obj['image_id']];v=load_index(self.geometry_index)[int(obj['image_id'])]
        assert Path(image['_depth']).resolve()==Path(v['depth_file']).resolve()
        raw=mmcv.imread(str(image['_depth']),flag='unchanged');valid=(raw!=image['depth_invalid_code'])&(raw>0)
        depth=np.where(valid,raw.astype(np.float32)*image['depth_scale_m'],0)
        from pycocotools import mask as mu
        seg=obj['segmentation'];h,w=raw.shape
        if isinstance(seg,list):seg=mu.merge(mu.frPyObjects(seg,h,w))
        elif isinstance(seg.get('counts'),list):seg=mu.frPyObjects(seg,h,w)
        ys,xs=np.where(mu.decode(seg));region=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
        donor['geometry'],_=geometry(depth,valid,v['K'],region=region);return donor
    def _paste(self,r,donor,y,x,activity,obj):
        accepted=super()._paste(r,donor,y,x,activity,obj)
        if accepted:
            h,w=donor['mask'].shape;r['geometry'][y:y+h,x:x+w][donor['mask']]=donor['geometry'][donor['mask']]
        return accepted

@TRANSFORMS.register_module()
class GeometryFlip(RGBDRandomFlip):
    def _flip(self,r):
        super()._flip(r);r['geometry']=mmcv.imflip(r['geometry'],direction=r['flip_direction']).copy()
        # This is a pixel permutation, not a reflection of the source camera frame.

@TRANSFORMS.register_module()
class GeometryResize(RGBDResize):
    def _resize_img(self,r):
        super()._resize_img(r);h,w=r['img'].shape[:2]
        r['geometry']=mmcv.imresize(r['geometry'],(w,h),interpolation='nearest')

@TRANSFORMS.register_module()
class GeometryCrop(RandomCrop):
    def _crop_data(self,r,crop_size,allow_negative_crop):
        _check_aligned(r);dtype=r['img'].dtype
        r['img']=np.concatenate([r['img'],r['rgbd_depth'][...,None],r['rgbd_raw_valid'][...,None],r['rgbd_label_valid'][...,None],r['geometry']],axis=-1)
        try:return super()._crop_data(r,crop_size,allow_negative_crop)
        finally:
            v=r['img'];r['img']=v[...,:3].astype(dtype,copy=False);r['rgbd_depth']=v[...,3].astype(np.float32);r['rgbd_raw_valid']=v[...,4].astype(bool);r['rgbd_label_valid']=v[...,5].astype(bool);r['geometry']=v[...,6:].astype(np.float32)

@TRANSFORMS.register_module()
class PackGeometryRGBD(PackRGBDDetInputs):
    def transform(self,r):
        assert r['geometry'].shape==(*r['img'].shape[:2],7) and np.isfinite(r['geometry']).all()
        output=super().transform(r);output['data_samples'].geometry=PixelData(data=torch.from_numpy(np.ascontiguousarray(r['geometry'].transpose(2,0,1))));return output

@MODELS.register_module()
class GeometryDataPreprocessor(RGBDDataPreprocessor):
    def __init__(self,geometry_mean,geometry_std,**kw):
        super().__init__(**kw);assert len(geometry_mean)==len(geometry_std)==7
        assert all(np.isfinite(geometry_mean)) and all(np.isfinite(geometry_std)) and min(geometry_std)>0
        self.geometry_mean=tuple(geometry_mean);self.geometry_std=tuple(geometry_std)
    def forward(self,data,training=False):
        # Parent handles the same 5 input channels and original RGB normalization.
        out=super().forward(data,training);h,w=out['inputs'].shape[-2:]
        for ds in out['data_samples']:
            g=ds.geometry.data.to(device=out['inputs'].device,dtype=torch.float32);assert g.shape[0]==7 and torch.isfinite(g).all()
            valid=g[6:7]>.5;mean=g.new_tensor(self.geometry_mean)[:,None,None];std=g.new_tensor(self.geometry_std)[:,None,None]
            g=torch.where(valid,(g-mean)/std,0)
            ds.geometry=PixelData(data=torch.nn.functional.pad(g,(0,w-g.shape[-1],0,h-g.shape[-2]),value=0))
        return out

def adapt_pipeline(pipe,index):
    names={'LoadCourseRGBDImage':'LoadGeometryRGBD','CourseRGBDSmallCopyPaste':'GeometryCopyPaste','RGBDRandomFlip':'GeometryFlip','RGBDResize':'GeometryResize','RGBDRandomCrop':'GeometryCrop','PackRGBDDetInputs':'PackGeometryRGBD'}
    out=copy.deepcopy(pipe)
    for t in out:
        if t['type'] in names:t['type']=names[t['type']]
        if t['type'] in ('LoadGeometryRGBD','GeometryCopyPaste'):t['geometry_index']=index
        if t['type']=='RandomResize':assert t['resize_type']=='RGBDResize';t['resize_type']='GeometryResize'
    return out
