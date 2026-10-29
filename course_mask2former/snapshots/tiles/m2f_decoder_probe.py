"""Read-only, GT-free inference capture; separate post-inference CPU analysis."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

from course_batch_predict import VAL_SHA, atomic_json
from course_demo import sha256, verify_run
from course_resolution_probe import safe_request, scaled_pipeline, now

MANIFEST = 'outputs/course_mrcnn_stages_20261018_v2/manifest.json'
MANIFEST_SHA = '6a81ce934d435ee97d2eb5b5b3738d8aa1cfe1565d955f0f2e587e4f2c6de3fe'
ARMS = {
    'copy_paste': dict(run='outputs/course_directions_20261015_v1/formal/runs/copy_paste_long',
        report_sha='790a771d893437dba23e41180efbf149be4cec5398eab733ad924e2d23d14394',
        weight_sha='bdea9c0f87864941e9ba34abceb561328fc4a950051ea4e0ad59cf81219ccd1a',
        reference='outputs/course_m2f_copypaste1536_20261018_v1/evaluate_1536',
        reference_sha='b4e4c3dc6ebfbe3a63e88be36be6065388d63987e92323f7dbde0e57100451ee',
        parameters=44400414, state_bytes=177814584),
    'cp_detail': dict(run='outputs/course_m2f_cpdetail_20261018_v1/formal/runs/copy_paste_detail_long',
        report_sha='4286fb459d7638b4f3b5c8555b3dbe18da187a6cef56144e2bd6b0356c34d359',
        weight_sha='26f52e8c1769a288250f2a5c86300512561b8160f29e0bc8130a15637fe17dfc',
        reference='outputs/course_m2f_cpdetail_20261018_v1/evaluate_1536',
        reference_sha='f0b84e3d75ee61900f51c490bf6354e2cc87b8a7b139c18a7f3f2d4e5f23fdbe',
        parameters=44402718, state_bytes=177823800),
}
CAP_BYTES = 5 * 2**30


def read(path):
    return json.loads(Path(path).read_text())


def manifest(workspace, images):
    path = Path(workspace)/MANIFEST
    if sha256(path) != MANIFEST_SHA:
        raise ValueError('Fixed256 manifest changed')
    m = read(path)
    expected = sorted((im['id'] for im in images),
        key=lambda i: hashlib.sha256(f'20261013:{i}'.encode()).hexdigest())[:256]
    if m['count'] != 256 or m['image_ids'] != expected or m['val_sha256'] != VAL_SHA:
        raise ValueError('Not the original GT-independent256 selection')
    return m


def budget(root, deadline, extra_bytes=0):
    if time.time() >= deadline:
        raise TimeoutError('Original2h deadline; no extension')
    size = sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    if size + extra_bytes > CAP_BYTES:
        raise RuntimeError('5GiB output cap reached; no truncation/subsample fallback')
    return size


class Recorder:
    """Hook holds outputs without mutation or extra forward/model call."""
    def __init__(self):
        self.output = None
        self.calls = 0

    def __call__(self, module, inputs, output):
        if self.output is not None:
            raise ValueError('Unconsumed decoder outputs')
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError('Unexpected decoder output')
        self.output = output
        self.calls += 1
        return None

    def take(self):
        if self.output is None:
            raise ValueError('Decoder recorder not invoked')
        value, self.output = self.output, None
        return value


def map_logits(mask, meta):
    """Exactly head.predict then fusion.predict geometry; not direct resize."""
    import torch.nn.functional as F
    x = F.interpolate(mask, size=meta['batch_input_shape'][:2], mode='bilinear', align_corners=False)
    h, w = meta['img_shape'][:2]
    x = x[0, :, :h, :w]
    return F.interpolate(x[:, None], size=meta['ori_shape'][:2], mode='bilinear', align_corners=False)[:, 0]


def stage_values(cls, masks, meta, query_count=100):
    import torch
    if cls.shape != (1, query_count, 2) or masks.ndim != 4 or masks.shape[:2] != (1, query_count):
        raise ValueError('Require batch1, one thing plus background, unchanged queries')
    if not torch.isfinite(cls).all() or not torch.isfinite(masks).all():
        raise ValueError('Nonfinite decoder output')
    mapped = map_logits(masks, meta)
    binary = mapped > 0
    probability = cls[0].softmax(-1)[:, 0]
    quality = (mapped.sigmoid() * binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6)
    scores = probability * quality
    return binary, probability, quality, scores


def assert_same_prediction(a, b):
    import torch
    for key in ('masks', 'labels', 'bboxes'):
        if not torch.equal(getattr(a, key), getattr(b, key)):
            raise ValueError('Recorder changes final '+key)
    if not torch.allclose(a.scores, b.scores, rtol=0, atol=1e-6):
        raise ValueError('Recorder changes final scores')


def assert_final(binary, probability, scores, pred):
    import torch
    order = probability.topk(len(probability), sorted=False).indices
    if (not torch.equal(binary[order], pred.masks) or not (pred.labels == 0).all()
            or not torch.allclose(scores[order], pred.scores, rtol=0, atol=1e-6)):
        raise ValueError('Reconstructed final layer differs from production output')


def encode_layer(binary, probability, quality, scores):
    import numpy as np
    from pycocotools import mask as mu
    masks = binary.cpu().numpy()
    values = [v.cpu().tolist() for v in (probability, quality, scores)]
    result = []
    for q, mask in enumerate(masks):
        rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
        if not np.array_equal(mu.decode(rle).astype(bool), mask):
            raise ValueError('RLE pixel roundtrip failure')
        rle['counts'] = rle['counts'].decode('ascii')
        result.append(dict(query=q, class_probability=values[0][q], mask_quality=values[1][q],
            score=values[2][q], area=int(mask.sum()), segmentation=rle))
    return result


def capture(root, arm, deadline):
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    workspace = Path('/workspace')
    spec = ARMS[arm]
    run = workspace/spec['run']
    reference = workspace/spec['reference']
    if sha256(run/'report.json') != spec['report_sha'] or sha256(reference/'report.json') != spec['reference_sha']:
        raise ValueError('Accepted arm report changed')
    report, checkpoint, source = verify_run(run)
    if report['checkpoint_sha256'] != spec['weight_sha'] or report['final_optimizer_steps'] != [30000]:
        raise ValueError('Not accepted30k fixed weight')
    sys.path.insert(0, str(source))
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(run/'config.py'))
    import_modules_from_strings(**cfg.custom_imports)
    pipeline = scaled_pipeline(cfg.val_dataloader.dataset.pipeline, 1536)
    previous = Config.fromfile(str(reference/'inference_config.py'))
    if cfg.model.to_dict() != previous.model.to_dict() or pipeline != previous.val_dataloader.dataset.pipeline:
        raise ValueError('Production model/inference recipe drift')
    ann_file = Path(cfg.val_evaluator.ann_file)
    if ann_file.name != 'instances_val.json' or sha256(ann_file) != VAL_SHA:
        raise ValueError('Only official val metadata')
    # Only image metadata enters this phase. Annotations are not used until analyze.
    images = read(ann_file)['images']
    if len(images) != 2584 or len({x['id'] for x in images}) != 2584:
        raise ValueError('Wrong official val population')
    m = manifest(workspace, images)
    by_id = {x['id']: x for x in images}
    old_hashes = {x['image_id']: x for x in read(reference/'input_hashes.json')}
    out = Path(root)/arm
    out.mkdir()
    state = dict(status='running', phase='initialization', arm=arm, completed_images=0,
        started_at_utc=now(), training_started=False, gt_sent_to_model=False,
        test_inputs_used=False, test_annotations_used=False, ap_reevaluated=False,
        checkpoint_sha256=spec['weight_sha'], training_report_sha256=spec['report_sha'],
        reference_report_sha256=spec['reference_sha'], manifest_sha256=MANIFEST_SHA,
        val_annotations_sha256=VAL_SHA, input_scale=1536, original_shape=[1024, 1024])
    atomic_json(out/'status.json', state)
    saved = torch.load(checkpoint, map_location='cpu')
    if any(not torch.isfinite(t).all() for t in saved['state_dict'].values() if t.is_floating_point()):
        raise ValueError('Nonfinite fixed weight')
    model = MODELS.build(cfg.model)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    if model.active_parameter_count() != spec['parameters'] or sum(t.numel()*t.element_size() for t in model.state_dict().values()) != spec['state_bytes']:
        raise ValueError('Model size differs from accepted arm')
    if model.panoptic_head.num_queries != 100 or model.panoptic_fusion_head.num_classes != 1:
        raise ValueError('Production queries/classes changed')
    layers = model.panoptic_head.num_transformer_decoder_layers + 1
    if layers != 10 or model.panoptic_fusion_head.test_cfg['max_per_image'] != 100:
        raise ValueError('Expected initial plus nine decoder layers / unchanged100')
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    compose = Compose(pipeline)
    recorder = Recorder()
    train_calls = [0]
    def forbid_training(module, inputs):
        if module.training:
            train_calls[0] += 1
            raise ValueError('No training is authorized')
    model.register_forward_pre_hook(forbid_training)
    completed, hashes, files = [], [], {}
    begin = time.monotonic()
    with torch.inference_mode():
        for idx, identity in enumerate(m['image_ids']):
            budget(root, deadline)
            req = safe_request(cfg.val_dataloader.dataset.data_root, by_id[identity])
            ih = dict(image_id=identity, rgb_sha256=sha256(req['img_path']), depth_sha256=sha256(req['depth_file']))
            if ih != old_hashes[identity]:
                raise ValueError('Input identity differs from prior full1536 evaluation')
            value = compose(copy.deepcopy(req))
            ds = value['data_samples']
            if 'gt_instances' in ds:
                if len(ds.gt_instances):
                    raise ValueError('GT may not enter inference')
                del ds.gt_instances
            if tuple(value['inputs'].shape) != (5, 1536, 1536) or tuple(ds.ori_shape) != (1024, 1024) or not torch.isfinite(value['inputs']).all():
                raise ValueError('Packed RGB-D/geometry/finite check')
            batch = pseudo_collate([value])
            if idx == 0:
                off = model.test_step(copy.deepcopy(batch))[0].pred_instances
            handle = model.panoptic_head.register_forward_hook(recorder)
            try:
                result = model.test_step(batch)[0]
            finally:
                handle.remove()
            pred = result.pred_instances
            if model.training or len(pred) != 100:
                raise ValueError('Unexpected production mode/output count')
            if idx == 0:
                assert_same_prediction(off, pred)
                del off
            cls_layers, mask_layers = recorder.take()
            if len(cls_layers) != layers or len(mask_layers) != layers:
                raise ValueError('Incomplete decoder capture')
            recorded = []
            for stage, (cls, masks) in enumerate(zip(cls_layers, mask_layers)):
                binary, probability, quality, scores = stage_values(cls, masks, result.metainfo)
                if stage == layers - 1:
                    assert_final(binary, probability, scores, pred)
                rows = encode_layer(binary, probability, quality, scores)
                recorded.append(dict(stage=stage, label='pre_decoder' if stage == 0 else f'decoder_{stage}', queries=rows))
                del binary, probability, quality, scores
            payload = dict(image_id=identity, input_hash=ih, original_shape=[1024,1024], layers=recorded,
                gt_sent_to_model=False, final_output_equivalent=True)
            encoded = gzip.compress(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
            budget(root, deadline, len(encoded))
            path = out/f'{identity}.json.gz'
            with path.open('xb') as f:
                f.write(encoded)
            files[path.name] = sha256(path)
            hashes.append(ih);completed.append(identity)
            if (idx+1) % 16 == 0 or idx == 255:
                state.update(phase='capture', completed_images=idx+1, last_image_id=identity,
                    elapsed_seconds=time.monotonic()-begin, updated_at_utc=now())
                atomic_json(out/'status.json', state)
                print(json.dumps(state), flush=True)
            del batch, value, ds, pred, result, cls_layers, mask_layers, recorded, payload, encoded
    if completed != m['image_ids'] or recorder.calls != 256 or train_calls[0] != 0:
        raise ValueError('Coverage/recorder/train audit')
    if sha256(checkpoint) != spec['weight_sha'] or sha256(ann_file) != VAL_SHA:
        raise ValueError('Fixed inputs changed')
    atomic_json(out/'input_hashes.json', hashes)
    state.update(status='passed', phase='complete', finished_at_utc=now(), layer_count=layers,
        query_count=100, image_ids=completed, input_hashes_sha256=sha256(out/'input_hashes.json'),
        prediction_sha256=files, recorder_calls=recorder.calls, model_eval_calls=257,
        train_calls=0, first_image_off_on_equivalent=True, all_final_layers_equivalent=True,
        rle_roundtrip_verified=True, active_parameters=spec['parameters'], state_tensor_bytes=spec['state_bytes'])
    budget(root, deadline)
    atomic_json(out/'report.json', state);atomic_json(out/'status.json', state)


def match_iou(ious):
    """Maximum-cardinality IoU>=.5 bipartite matching, then IoU tie-break."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not ious.size:
        return {}
    q, g = linear_sum_assignment(-((ious >= .5).astype(float) + ious/(min(ious.shape)+1)))
    return {int(j): int(i) for i,j in zip(q,g) if ious[i,j] >= .5}


def analyze_one(layers, anns, gt_rles):
    import numpy as np
    from pycocotools import mask as mu
    records = [dict(annotation_id=a['id'], area=a['area'], layers=[]) for a in anns]
    matrices = []
    for layer in layers:
        rows = layer['queries']
        ious = np.asarray(mu.iou([r['segmentation'] for r in rows], gt_rles, [0]*len(anns)), dtype=float)
        if ious.size == 0:
            ious = np.zeros((len(rows), len(anns)))
        matrices.append(ious)
        strict = match_iou(ious)
        display = {}
        for q in sorted((q for q,r in enumerate(rows) if r['score'] >= .3), key=lambda q: -rows[q]['score']):
            candidates = [g for g in range(len(anns)) if g not in display]
            if candidates:
                g = max(candidates, key=lambda g: ious[q,g])
                if ious[q,g] >= .5:
                    display[g] = q
        rank = {q:n+1 for n,q in enumerate(sorted(range(len(rows)), key=lambda q:-rows[q]['score']))}
        for g, record in enumerate(records):
            q = int(ious[:,g].argmax())
            best = rows[q]
            record['layers'].append(dict(stage=layer['stage'], best_query=q, best_iou=float(ious[q,g]),
                score=best['score'], class_probability=best['class_probability'], mask_quality=best['mask_quality'],
                score_rank=rank[q], covering_queries=int((ious[:,g]>=.5).sum()),
                one_to_one_query=strict.get(g), display_query=display.get(g)))
    for g, record in enumerate(records):
        coverage = [x['best_iou'] >= .5 for x in record['layers']]
        record.update(first_covered_stage=next((i for i,v in enumerate(coverage) if v),None),
            coverage_loss_transitions=[i for i in range(1,len(coverage)) if coverage[i-1] and not coverage[i]],
            never_covered=not any(coverage), early_covered_final_lost=any(coverage[:-1]) and not coverage[-1],
            final_covered=coverage[-1], final_one_to_one=record['layers'][-1]['one_to_one_query'] is not None,
            final_display_matched=record['layers'][-1]['display_query'] is not None,
            early_covering_query_ids=sorted({int(q) for matrix in matrices[:-1] for q in np.where(matrix[:,g]>=.5)[0]}))
        record['early_queries_final_iou'] = {str(q):float(matrices[-1][q,g]) for q in record['early_covering_query_ids']}
    return records


def summarize(records, n_layers):
    def group(items):
        missed = [r for r in items if not r['final_covered']]
        lost = [r for r in missed if r['early_covered_final_lost']]
        return dict(gt=len(items), final_no_coverage=len(missed), early_covered_final_lost=len(lost),
            early_loss_fraction=len(lost)/len(missed) if missed else 0., never_covered=sum(r['never_covered'] for r in items),
            final_covered_not_display=sum(r['final_covered'] and not r['final_display_matched'] for r in items),
            stages=[dict(stage=i, covered=sum(r['layers'][i]['best_iou']>=.5 for r in items),
                one_to_one=sum(r['layers'][i]['one_to_one_query'] is not None for r in items),
                display_matched=sum(r['layers'][i]['display_query'] is not None for r in items),
                lost_from_previous=sum(i in r['coverage_loss_transitions'] for r in items)) for i in range(n_layers)])
    small = [r for r in records if r['area']<=1024]
    results = {'small':group(small), 'tiny':group([r for r in small if r['area']<=256]),
        'small_257_1024':group([r for r in small if r['area']>256])}
    s=results['small']
    results['early_layer_hypothesis_supported'] = s['gt']>=100 and s['early_covered_final_lost']>=30 and s['early_loss_fraction']>=.3
    return results


def analyze(root, deadline):
    from pycocotools.coco import COCO
    root=Path(root)
    plan=read(root/'plan.json')
    ann=Path('/workspace')/plan['val_annotations']
    if sha256(ann)!=VAL_SHA:
        raise ValueError('Val annotation identity drift')
    coco=COCO(str(ann))
    out=root/'analysis';out.mkdir()
    reports={a:read(root/a/'report.json') for a in ARMS}
    if any(r['status']!='passed' or r['completed_images']!=256 for r in reports.values()):
        raise ValueError('Both full fixed256 captures required')
    if read(root/'copy_paste/input_hashes.json') != read(root/'cp_detail/input_hashes.json'):
        raise ValueError('Arms input mismatch')
    comparison={}
    all_rows={}
    for arm,r in reports.items():
        records=[]
        for identity in r['image_ids']:
            budget(root,deadline)
            path=root/arm/f'{identity}.json.gz'
            if sha256(path)!=r['prediction_sha256'][path.name]:
                raise ValueError('Captured prediction changed')
            payload=json.loads(gzip.decompress(path.read_bytes()))
            anns=coco.loadAnns(coco.getAnnIds(imgIds=[identity],iscrowd=False))
            rows=analyze_one(payload['layers'],anns,[coco.annToRLE(a) for a in anns])
            records += [dict(row,image_id=identity) for row in rows]
        result=summarize(records,r['layer_count'])
        comparison[arm]=result;all_rows[arm]=records
        atomic_json(out/f'{arm}_per_gt.json',records)
    old={x['annotation_id']:x for x in all_rows['copy_paste'] if x['area']<=1024}
    new={x['annotation_id']:x for x in all_rows['cp_detail'] if x['area']<=1024}
    if old.keys()!=new.keys():raise ValueError('GT pairing mismatch')
    paired=[dict(annotation_id=k,image_id=old[k]['image_id'],area=old[k]['area'],
        old_final_iou=old[k]['layers'][-1]['best_iou'],new_final_iou=new[k]['layers'][-1]['best_iou'],
        gained=not old[k]['final_covered'] and new[k]['final_covered'],
        lost=old[k]['final_covered'] and not new[k]['final_covered']) for k in old]
    atomic_json(out/'paired_small.json',paired)
    result=dict(status='passed',finished_at_utc=now(),arms=comparison,
        next_route='propose_single_early_candidate_retention_validation' if comparison['copy_paste']['early_layer_hypothesis_supported'] else 'early_layer_hypothesis_not_supported_select_from_measured_failures',
        original_small_gt=len(old),new_coverage_gains=sum(x['gained'] for x in paired),new_coverage_losses=sum(x['lost'] for x in paired),
        manifest_sha256=MANIFEST_SHA,report_sha256={a:sha256(root/a/'report.json') for a in ARMS},
        gt_used_only_after_inference=True,training_started=False,test_inputs_used=False,test_annotations_used=False,
        ap_reevaluated=False,limitations='GT oracle coverage is a diagnostic upper bound, not AP or a deployable gain. Stages0=predecoder,1..9=decoder layers. Strict matching maximizes cardinality; display match uses score>=.3/IoU>=.5. Single fixed subset, no new performance experiment authorized.')
    budget(root,deadline)
    atomic_json(out/'comparison.json',result)
    atomic_json(out/'report.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','analyze']);p.add_argument('--root',required=True);p.add_argument('--arm',choices=list(ARMS));p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    if a.mode=='capture':capture(Path(a.root),a.arm,a.deadline)
    else:analyze(Path(a.root),a.deadline)
