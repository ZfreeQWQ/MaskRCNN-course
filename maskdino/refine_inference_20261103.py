"""Portable subset of the selected release's evaluation and merge primitives."""
import collections,copy,json
from pathlib import Path
import settings
import geometry_maskdino as r
from scale_data import GeometryData
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
torch,np,d,F=r.torch,r.np,r.d,r.t.F
B=settings.WORKDIR;STATS=settings.GEOMETRY_STATS
def write(path,value): d.atomic_json(path,value)
def read(path): return json.loads(path.read_text())
def grouped(path):
    result=collections.defaultdict(list)
    for p in read(path): result[p['image_id']].append(p)
    return result

def score(data,indices,predictions,out):
    assert data.split=='val', 'Public evaluator accepts validation only'
    images=[dict(id=data.index.records[i]['id'],height=1024,width=1024) for i in indices]
    anns=[dict(a) for i in indices for a in data.annotations[i]]
    for i,a in enumerate(anns): a['id']=i+1
    gt=COCO();gt.dataset=dict(info={},images=images,annotations=anns,categories=[dict(id=1,name='component')]);gt.createIndex()
    if predictions: dt=gt.loadRes(copy.deepcopy(predictions))
    else:
        dt=COCO();dt.dataset=dict(gt.dataset,annotations=[]);dt.createIndex()
    ev=COCOeval(gt,dt,'segm');ev.params.imgIds=[a['id'] for a in images];ev.params.catIds=[1]
    ev.evaluate();ev.accumulate();ev.summarize()
    value=dict(scope=('full_val' if len(indices)==2584 else 'validation_subset'),images=len(images),instances=len(anns),original_mask_area=True,max_detections=100,
        segm=dict(zip(('AP','AP50','AP75','APs','APm','APl'),[float(v*100) for v in ev.stats[:6]])))
    write(out,value);return value

def enc(mask):
    e=mu.encode(np.asfortranarray(mask.astype(np.uint8)));e['counts']=e['counts'].decode('ascii');return e

def encode_pred(pred,image_id):
    result=[]
    for mask,s,cls in zip(pred.pred_masks,pred.scores,pred.pred_classes):
        assert int(cls)==0
        m=mask.numpy()
        if m.any():result.append(dict(image_id=image_id,category_id=1,score=float(s),segmentation=enc(m)))
    assert len(result)<=100
    return result

def merge(base,extra):
    items=sorted(base+extra,key=lambda p:p['score'],reverse=True)
    if not items:return []
    overlap=mu.iou([p['segmentation'] for p in items],[p['segmentation'] for p in items],[0]*len(items))
    keep=[]
    for i in range(len(items)):
        if not keep or max(overlap[i,keep])<.5:keep.append(i)
        if len(keep)==100:break
    return [items[i] for i in keep]

def self_checks():
    a=np.zeros((1024,1024),np.uint8);a[30:40,40:50]=1
    b=np.zeros_like(a);b[80:90,80:90]=1
    p=dict(image_id=1,category_id=1,score=.9,segmentation=enc(a))
    q=dict(image_id=1,category_id=1,score=.8,segmentation=enc(b))
    assert len(merge([p],[dict(p),q]))==2
    assert np.array_equal(mu.decode(enc(a)),a)
