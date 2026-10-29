"""CPU-only lossless mask-only export; drop only proven zero-area slots."""
import csv,hashlib,json,math
from pathlib import Path
SCOPE='authorized_test_once'
callback=lambda:None

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ids_checked(ids):
 assert all(type(i) is int for i in ids) and len(ids)==len(set(ids));return ids

def filter_rows(rows,ident):
 import numpy as np
 from pycocotools import mask as mu
 assert isinstance(rows,list) and len(rows)<=100
 keep=[];empty=0
 for row in rows:
  assert set(row)>={'image_id','category_id','score','segmentation'}
  assert row['image_id']==ident and row['category_id']==1
  score=row['score'];assert type(score) in (int,float) and math.isfinite(score) and 0<=score<=1
  seg=row['segmentation'];assert set(seg)=={'size','counts'} and seg['size']==[1024,1024]
  assert isinstance(seg['counts'],str) and seg['counts'].isascii()
  mask=mu.decode(seg);assert mask.shape==(1024,1024) and np.isin(mask,[0,1]).all()
  encoded=mu.encode(np.asfortranarray(mask));encoded['counts']=encoded['counts'].decode('ascii');assert encoded==seg
  if 'bbox' in row:assert np.array_equal(mu.toBbox(seg),row['bbox'])
  area=int(mu.area(seg));assert area==int(mask.sum())
  if area==0:empty+=1;continue
  keep.append({k:row[k] for k in ('image_id','category_id','score','segmentation')})
 return keep,empty

def export_manifest(pred_path,input_path,csv_path):
 pred_path,input_path,csv_path=map(Path,(pred_path,input_path,csv_path));root=pred_path.parent
 pred=json.loads(pred_path.read_text());inp=json.loads(input_path.read_text());ids=ids_checked(inp['ids'])
 assert len(ids)==2112 and pred['scope']==SCOPE and pred['test_label_access'] is False
 assert pred['input_manifest_sha256']==sha(input_path) and ids==[x['image_id'] for x in pred['images']]
 assert not csv_path.exists();tmp=csv_path.with_suffix('.csv.validating');assert not tmp.exists()
 expected=[];removed=0;total=0;kept=0;audit=[]
 with tmp.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['image_id','PredictionString']);writer.writeheader()
  for x in pred['images']:
   i=x['image_id'];path=root/x['predictions'];assert path.resolve().is_relative_to(root.resolve()) and sha(path)==x['sha256']
   rows=json.loads(path.read_text());assert len(rows)==x['instances']
   out,n=filter_rows(rows,i);text=json.dumps(out,separators=(',',':'),allow_nan=False);assert json.loads(text)==out
   writer.writerow(dict(image_id=i,PredictionString=text));expected.append(hashlib.sha256(text.encode()).hexdigest())
   removed+=n;total+=len(rows);kept+=len(out);audit.append(dict(image_id=i,original=len(rows),removed_zero_area=n,kept=len(out)));callback()
 with tmp.open(newline='') as f:
  reader=csv.DictReader(f);assert reader.fieldnames==['image_id','PredictionString'];count=0
  for row,i,h in zip(reader,ids,expected,strict=True):
   assert list(row)==['image_id','PredictionString'] and int(row['image_id'])==i
   text=row['PredictionString'];assert hashlib.sha256(text.encode()).hexdigest()==h
   parsed=json.loads(text);assert isinstance(parsed,list) and len(parsed)<=100
   assert all(set(x)=={'image_id','category_id','score','segmentation'} for x in parsed);count+=1
 assert count==2112 and total==kept+removed
 tmp.replace(csv_path)
 (root/'export_filter_audit.json').write_text(json.dumps(audit,indent=2))
 return dict(images=count,instances=kept,original_instances=total,removed_zero_area_instances=removed,remaining_masks_scores_order_unchanged=True,all_RLE_decode_reencode=True,CSV_JSON_roundtrip=True,mask_only=True,csv_sha256=sha(csv_path),csv_bytes=csv_path.stat().st_size)
