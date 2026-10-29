def error_analysis():
    """Post-hoc one-to-one IoU analysis; never COCO AP or model inference."""
    from collections import defaultdict
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    from types import SimpleNamespace
    q=SimpleNamespace(DATA='data/course_blender_20261013/prepared')
    coco=COCO(str(ROOT/q.DATA/'annotations/instances_val.json'))
    paths={'baseline':ROOT/s.REFERENCE/'predictions.segm.json',s.ARM:OUT/'evaluate_1536/predictions.segm.json'}
    tables={};stats={};matched={};areas={}
    for arm,path in paths.items():
        table=defaultdict(list)
        for row in read(path):table[row['image_id']].append(row)
        tables[arm]=table;matched[arm]={'.5':set(),'.75':set()}
        stats[arm]={'FP_all':0,'small_FP_all':0,'FP_score_ge_0.3':0,'small_FP_ge_0.3':0,
            'small_FP_ge_0.9':0,'small_FP_large_GT_containment_ge_0.8':0,'ordinary_duplicate_FP':0}
    per_image=[]
    for number,image in enumerate(coco.dataset['images'],1):
        deadline(s.DEADLINE-60)
        ident=image['id'];anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident])) if not a.get('iscrowd',0)]
        gt=[coco.annToRLE(a) for a in anns]
        for a in anns:areas[a['id']]=a['area']
        entry={'image_id':ident}
        for arm,table in tables.items():
            rows=table[ident];rles=[r['segmentation'] for r in rows]
            ious=mu.iou(rles,gt,[0]*len(gt));area=mu.area(rles) if rows else []
            order=sorted(range(len(rows)),key=lambda j:(-rows[j]['score'],j));entry[arm]={}
            for label,threshold in (('.5',.5),('.75',.75)):
                used=set();fps=[]
                for k in order:
                    candidates=[j for j in range(len(anns)) if j not in used and ious[k,j]>=threshold]
                    if candidates:used.add(max(candidates,key=lambda j:(ious[k,j],-j)))
                    else:fps.append(k)
                ids={anns[j]['id'] for j in used};matched[arm][label].update(ids)
                entry[arm][label]=sorted(ids)
                if label=='.5':
                    st=stats[arm];st['FP_all']+=len(fps)
                    small=[k for k in fps if area[k]<=1024]
                    st['small_FP_all']+=len(small)
                    st['FP_score_ge_0.3']+=sum(rows[k]['score']>=.3 for k in fps)
                    st['small_FP_ge_0.3']+=sum(rows[k]['score']>=.3 for k in small)
                    st['small_FP_ge_0.9']+=sum(rows[k]['score']>=.9 for k in small)
                    st['ordinary_duplicate_FP']+=sum(any(ious[k,j]>=.5 for j in range(len(anns))) for k in fps)
                    large=[gt[j] for j,a in enumerate(anns) if a['area']>1024]
                    if small and large:
                        containment=mu.iou([rles[k] for k in small],large,[1]*len(large))
                        st['small_FP_large_GT_containment_ge_0.8']+=sum(bool((v>=.8).any()) for v in containment)
        per_image.append(entry)
        if number%50==0:mark('CPU_paired_error_analysis',images=number)
    bins={'tiny_1_16':(0,16),'tiny_17_64':(16,64),'small_65_256':(64,256),'small_257_1024':(256,1024),'all_small':(0,1024)}
    summary={}
    for label in ('.5','.75'):
        a=matched['baseline'][label];b=matched[s.ARM][label];summary[label]={}
        for name,(low,high) in bins.items():
            chosen={k for k,v in areas.items() if low<v<=high}
            summary[label][name]=dict(GT=len(chosen),baseline=len(a&chosen),new=len(b&chosen),
                rescued=sorted((b-a)&chosen),lost=sorted((a-b)&chosen))
    save(OUT/'paired_error_analysis.json',dict(status='passed',images=2584,summary=summary,
        FP=stats,per_image=per_image,input_caches_sha256={k:sha(v) for k,v in paths.items()},
        metric_scope='score-ordered one-to-one class-matched mask IoU; not AP; GT only post-hoc',
        new_AP_runs=0,new_inference=0))
