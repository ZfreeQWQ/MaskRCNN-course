"""Full-image + four-crop inference for the frozen ccc MaskDINO release."""
import argparse,csv,json,math
from pathlib import Path
import settings
import refine_inference_20261103 as h
import crop2048_impl

EXPECTED = {
    'global.pth': '312654e16f6d413e06d91ef3674e21de3d3bfa2a8a3189f753ace04069da9268',
    'crop.pth': '472463ef32044a3525417bcfe4c5ff162af424aa6e704609799bde1242ec88de',
}

def load(model,name):
    path=settings.WEIGHTS/name
    assert h.d.sha(path)==EXPECTED[name], 'Checkpoint does not belong to this release: '+name
    ck=h.torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    model.eval()

def export(predictions,ids,path):
    grouped={str(i):[] for i in ids}
    for p in predictions:
        item={k:p[k] for k in ('category_id','score','segmentation')}
        assert item['category_id']==1 and math.isfinite(item['score']) and 0<=item['score']<=1
        assert item['segmentation']['size']==[1024,1024]
        assert isinstance(item['segmentation']['counts'],str)
        grouped[str(p['image_id'])].append(item)
    assert len(ids)==len(set(map(str,ids)))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['image_id','PredictionString'])
        for i in ids:
            assert len(grouped[str(i)])<=100
            writer.writerow([i,json.dumps(grouped[str(i)],separators=(',',':'))])

@h.torch.inference_mode()
def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    h.torch.set_num_threads(4);h.torch.cuda.set_per_process_memory_fraction(.66)
    data=h.GeometryData(a.split,'geometry',settings.GEOMETRY_STATS,side=1024,limit=a.limit)
    ids=[row['id'] for row in data.index.records]
    manifest={'split':a.split,'ids':ids,'weights':EXPECTED,'geometry_stats_sha256':h.d.sha(settings.GEOMETRY_STATS),
              'source_hashes':{p.name:h.d.sha(p) for p in Path(__file__).parent.glob('*.py')},
              'global_side':2048,'crop_native':640,'crop_side':2048,'nms_iou':.5,'max_detections':100}
    if (out/'manifest.json').exists():assert h.read(out/'manifest.json')==manifest
    else:h.write(out/'manifest.json',manifest)
    model,_=h.r.build(out,'geometry')
    for phase in ['global','merged']:
        dest=out/phase;dest.mkdir(exist_ok=True)
        load(model,'global.pth' if phase=='global' else 'crop.pth')
        for i,image_id in enumerate(ids):
            path=dest/f'{i:05d}.json'
            if path.exists():continue
            if phase=='global':
                item=data.item(i,False);image=item['image']
                item['image']=h.torch.cat((h.F.interpolate(image[None,:3],(2048,2048),mode='bilinear',align_corners=False)[0],
                                          h.F.interpolate(image[None,3:],(2048,2048),mode='nearest')[0]))
                pred=h.encode_pred(model([item])[0]['instances'].to('cpu'),image_id)
            else:
                base=h.read(out/'global'/f'{i:05d}.json')
                pred=crop2048_impl.inference(model,data,i,'crop640',base)
            h.write(path,pred)
            print(phase,i+1,'/',len(ids),flush=True)
    predictions=[p for i in range(len(ids)) for p in h.read(out/'merged'/f'{i:05d}.json')]
    h.write(out/'predictions.json',predictions)
    if a.split=='val':h.score(data,list(range(len(ids))),predictions,out/'validation.json')
    if a.template:
        with Path(a.template).open(newline='',encoding='utf-8-sig') as f:
            template_ids=[row['image_id'] for row in csv.DictReader(f)]
        assert set(template_ids)==set(map(str,ids)) and len(template_ids)==len(ids)
        ids=template_ids
    if a.split=='test':assert a.template, 'Competition export requires template ID order'
    export(predictions,ids,out/'submission.csv')
    print(json.dumps({'status':'completed','images':len(ids),'csv_sha256':h.d.sha(out/'submission.csv')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--template',help='Competition template CSV; required for test export')
    main(p.parse_args())
