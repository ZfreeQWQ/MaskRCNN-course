"""CPU synthetic evidence for error_analysis.py; no real annotations or GPU access."""
from pathlib import Path
import copy,json,subprocess,sys
import numpy as np
from PIL import Image
from pycocotools import mask as mu
project=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project/'tools'))
from error_analysis import analyze
root=project/'reports/error_analysis_synthetic';root.mkdir(exist_ok=True)
def item(box,**fields):
 x,y,w,h=box;mask=np.zeros((256,256),dtype=np.uint8);mask[y:y+h,x:x+w]=1
 rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii')
 return dict(image_id=1,category_id=1,bbox=list(box),segmentation=rle,**fields)
boxes=[(10,10,10,10),(25,10,10,10),(50,50,40,50),(120,120,100,100),(0,200,32,32),(150,0,96,96)]
gt=[item(box,id=11+i,area=box[2]*box[3],iscrowd=0,semantic_type='synthetic') for i,box in enumerate(boxes)]
info=dict(id=1,file_name='rgb.png',depth_file_name='depth.png',height=256,width=256,
          split='val',depth_scale_m=.001,depth_invalid_code=0)
data=dict(images=[info],annotations=gt,categories=[dict(id=1,name='component')])
preds=[item(boxes[0],score=.1),item((10,10,25,10),score=.95,id=55),
       item((0,100,10,10),score=.9),item((65,50,40,50),score=.85)]
result=analyze(data,preds)
assert result['all_area_counts']==dict(missed=6,boundary_candidate=3,false_positive=3,adjacent_merge_candidate=1),result['all_area_counts']
assert result['population_by_area']['gt']==dict(S=3,M=3,L=2)
assert result['counts_by_area']['missed']==dict(S=3,M=3,L=2)
assert result['counts']['small_missed']==3
merge=result['stratified_cases']['adjacent_merge_candidate'][0]
assert merge['prediction_index']==1 and merge['prediction_id']==55
assert merge['covered_annotation_ids']==[11,12]
assert all(x['coverage']==1 for x in merge['covered_annotations'])
assert all(case['prediction_index']!=0 for rows in result['stratified_cases'].values() for case in rows)
assert result==analyze(data,preds),'candidate generation must be deterministic'
cap_data=dict(data,images=[dict(info,id=i) for i in range(1,31)],
              annotations=[dict(gt[0],id=i,image_id=i) for i in range(1,31)])
capped=analyze(cap_data,[],max_cases=20)
assert capped['all_area_counts']['missed']==30
assert len(capped['stratified_cases']['missed'])==20
assert len({x['case_id'] for x in capped['stratified_cases']['missed']})==20
for maximum in [0,21]:
 try:analyze(data,preds,max_cases=maximum)
 except ValueError:pass
 else:raise AssertionError('case budget must be bounded')
try:analyze(dict(data,images=[dict(info,split='test')]),preds)
except ValueError:pass
else:raise AssertionError('test must be refused')
raw=np.full((256,256),1000,dtype=np.uint16);raw[10:20,10:15]=0
Image.fromarray(raw).save(root/'depth.png')
rgb=np.zeros((256,256,3),dtype=np.uint8);rgb[...,0]=np.arange(256,dtype=np.uint8)[None,:];rgb[...,1]=100
Image.fromarray(rgb).save(root/'rgb.png')
for name,obj in [('annotations.json',data),('predictions.json',preds),('b0.json',[dict(gt[0],score=.8)]),('m1.json',[dict(row,score=.9) for row in gt[:3]])]:
 (root/name).write_text(json.dumps(obj))
command=[sys.executable,str(project/'tools/error_analysis.py'),'--annotations',str(root/'annotations.json'),
 '--predictions',str(root/'predictions.json'),'--baseline-predictions',str(root/'b0.json'),
 '--m1-predictions',str(root/'m1.json'),'--data-root',str(root),'--output-dir',str(root/'output'),'--render']
subprocess.run(command,check=True)
output=json.loads((root/'output/error_candidates.json').read_text())
miss=next(row for row in output['stratified_cases']['missed'] if row['annotation_id']==11)
assert miss['depth_evidence']['invalid_fraction']==.5,miss['depth_evidence']
assert miss['depth_evidence']['median_m']==1
assert output['sources']['M1_predictions']==str(root/'m1.json')
assert output['sources']['selected_predictions']==str(root/'predictions.json')
assert miss['candidate_model']=='Selected' and miss['review_status']=='pending_human_review'
with Image.open(root/'output'/miss['sheet']) as sheet:assert sheet.size==(384*6,96+384*2+24)
assert miss['roi_xyxy'][2]-miss['roi_xyxy'][0]<256,'local enlarged ROI required'
review=json.loads((root/'output/human_review_template.json').read_text())
assert all(row['confirmed_error'] is None and row['review_status']=='pending_human_review' for row in review['cases'])
print(json.dumps(dict(status='passed',synthetic_only=True,gpu_used=False,
 groups=output['population_by_area']['gt'],counts=output['all_area_counts'],
 stable_original_prediction_index=merge['prediction_index'],depth_invalid_fraction=.5,
 cases_capped_at=20,model_columns=['RGB','Depth','GT','B0','M1','Selected'],
 review_status='pending_human_review',fixture_directory=str(root)),indent=2))
