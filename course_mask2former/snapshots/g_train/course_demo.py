"""CLI RGB-D comparison demo; annotated official val or unlabelled custom PNGs.

Each model is loaded in a fresh process from its own verified training snapshot.
GT is used only by the parent renderer, never sent to either inference worker.
No web server, upload, test-label access, model training, or Kaggle submission.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checked_image_path(path):
    path = Path(path).resolve()
    if 'test' in path.parts or not path.is_file() or path.suffix.lower() != '.png':
        raise ValueError('Demo accepts existing PNGs outside test directories only')
    return path


def check_inputs(rgb_path, depth_path, scale, invalid_code):
    import cv2
    import numpy as np
    from PIL import Image
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite per-image depth scale required')
    if not isinstance(invalid_code, int) or not 0 <= invalid_code <= 65535:
        raise ValueError('Depth invalid code must be uint16')
    rgb_path, depth_path = checked_image_path(rgb_path), checked_image_path(depth_path)
    with Image.open(rgb_path) as im:
        rgb_size = im.size
    depth = cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Depth must be single-channel uint16 PNG')
    if depth.shape[::-1] != rgb_size or depth.size > 4096**2:
        raise ValueError('RGB/depth dimensions must match and be at most 16 MP')
    return dict(img_path=str(rgb_path), depth_file=str(depth_path),
        depth_scale_m=scale, depth_invalid_code=invalid_code,
        rgb_sha256=sha256(rgb_path), depth_sha256=sha256(depth_path),
        original_shape=list(depth.shape))


def encode_predictions(masks, scores, image_id):
    import numpy as np
    from pycocotools import mask as mask_utils
    masks, scores = np.asarray(masks), np.asarray(scores)
    if masks.ndim != 3 or len(scores) != len(masks) or len(scores) > 100:
        raise ValueError('Expected at most 100 aligned masks and scores')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid prediction scores')
    rows = []
    for mask, score in zip(masks, scores):
        mask = mask.astype(bool)
        if not mask.any():
            continue
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        box = mask_utils.toBbox(rle).tolist()
        rle['counts'] = rle['counts'].decode('ascii')
        rows.append(dict(image_id=int(image_id), category_id=1, bbox=box,
                         score=float(score), segmentation=rle))
    return sorted(rows, key=lambda row: -row['score'])


def verify_run(run):
    run = Path(run).resolve()
    report = json.loads((run/'report.json').read_text())
    if (report['status'] != 'passed' or report['args']['functional_smoke']
            or report['protocol'] != 'component-official-train-val-standard-coco-mask-maxdets100'
            or report['test_annotations_used'] or report['test_inputs_used']):
        raise ValueError('Require a passed full course train/val run')
    checkpoint = run/Path(report['checkpoint']).name
    if sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    root = run.parents[2]
    plan = json.loads((root/'plan.json').read_text())
    source = root/'source'
    for relative, expected in plan['source_sha256'].items():
        if sha256(source/relative) != expected:
            raise ValueError('Frozen source hash mismatch: '+relative)
    return report, checkpoint, source


def worker(args):
    report, checkpoint, source = verify_run(args.run)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    request = json.loads(Path(args.request).read_text())
    checked = check_inputs(request['img_path'], request['depth_file'],
                           request['depth_scale_m'], request['depth_invalid_code'])
    if any(checked[k] != request[k] for k in checked):
        raise ValueError('Inputs changed after request creation')
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(Path(args.run)/'config.py'))
    import_modules_from_strings(**cfg.custom_imports)
    # Annotation-free inference, using exactly the trained validation geometry.
    pipeline = [dict(t) for t in cfg.val_dataloader.dataset.pipeline
                if t['type'] != 'LoadAnnotations']
    if [t['type'] for t in pipeline] != ['LoadCourseRGBDImage','RGBDResize','PackRGBDDetInputs']:
        raise ValueError('Unsupported inference pipeline')
    sample = Compose(pipeline)(dict(request, scene='demo'))
    if 'gt_instances' in sample['data_samples']:
        if len(sample['data_samples'].gt_instances):
            raise ValueError('GT must not enter model inference')
        del sample['data_samples'].gt_instances
    saved = torch.load(checkpoint, map_location='cpu')
    trained_cfg = Config.fromstring(saved['meta'].get('cfg',saved['meta'].get('config')), '.py')
    if trained_cfg.model.to_dict() != cfg.model.to_dict():
        raise ValueError('Checkpoint/config model mismatch')
    model = MODELS.build(cfg.model)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        pred = model.test_step(pseudo_collate([sample]))[0].pred_instances.cpu()
    if pred.labels.numel() and not bool((pred.labels == 0).all()):
        raise ValueError('Expected component-only predictions')
    if list(pred.masks.shape[-2:]) != request['original_shape']:
        raise ValueError('Predictions were not restored to the original image grid')
    rows = encode_predictions(pred.masks.numpy(), pred.scores.numpy(), request['img_id'])
    output = Path(args.output)
    save_json(output, rows)
    save_json(output.with_suffix('.provenance.json'), dict(
        checkpoint_sha256=report['checkpoint_sha256'], training_report_sha256=sha256(Path(args.run)/'report.json'),
        config_sha256=sha256(Path(args.run)/'config.py'), frozen_source_sha256=sha256(source/'run_course_experiment.py'),
        source=str(source), training_metrics=report['metrics'], input_rgb_sha256=request['rgb_sha256'],
        input_depth_sha256=request['depth_sha256'], instances=len(rows), gt_sent_to_model=False,
        note='Full model predictions up to 100, empty masks removed; no display-score filtering in JSON.'))


def blend(rgb, masks):
    import numpy as np
    result = np.asarray(rgb).astype(np.float32).copy()
    rng = np.random.default_rng(53)
    for mask in masks:
        color = rng.uniform(40,255,3)
        result[mask] = .5*result[mask]+.5*color
    return result.astype(np.uint8)


def render(output, request, gt, confidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils
    rgb = np.asarray(Image.open(request['img_path']).convert('RGB'))
    raw = np.asarray(Image.open(request['depth_file']))
    depth = np.where((raw != request['depth_invalid_code']) & (raw > 0),
                     raw.astype(float)*request['depth_scale_m'], np.nan)
    panels = [('RGB input',rgb,None),('Depth (metres)',depth,'viridis')]
    panels.append(('Ground truth' if gt is not None else 'GT unavailable (custom input)',
                   blend(rgb, [mask_utils.decode(a['segmentation']).astype(bool) for a in gt]) if gt is not None else rgb,None))
    for label in ('before','after'):
        rows = json.loads((output/f'{label}_predictions.json').read_text())
        masks = [mask_utils.decode(r['segmentation']).astype(bool) for r in rows if r['score'] >= confidence]
        overlay = blend(rgb,masks)
        Image.fromarray(overlay).save(output/f'{label}_prediction.png')
        panels.append((f'{label}: {len(masks)} instances at score >= {confidence}',overlay,None))
    fig, axes = plt.subplots(1,5,figsize=(20,4.4))
    for ax,(title,content,cmap) in zip(axes,panels):
        ax.imshow(content,cmap=cmap);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.suptitle(f'Image {request["img_id"]}; independent instance colours, not cross-panel identity')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=120);plt.close(fig)


def demo(args):
    output,workspace = Path(args.output).resolve(),Path(args.workspace).resolve()
    if output.exists() or not output.is_relative_to(workspace/'outputs'):
        raise ValueError('Use a NEW output directory under workspace/outputs')
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        raise ValueError('Display confidence must be in [0,1]')
    gt = None
    if args.image_id is not None:
        if any(value is not None for value in (args.rgb,args.depth,args.depth_scale,args.depth_invalid_code)):
            raise ValueError('Choose val image ID OR custom RGB/depth input')
        data = workspace/'data/course_blender_20261013/prepared'
        annotation = data/'annotations/instances_val.json'
        before_report = json.loads((Path(args.before_run)/'report.json').read_text())
        if sha256(annotation) != before_report['val_annotations_sha256']:
            raise ValueError('Official validation annotations changed')
        val = json.loads(annotation.read_text())
        selected = [im for im in val['images'] if im['id'] == args.image_id]
        if len(selected) != 1:
            raise ValueError('Image ID must belong to official val; no train/test lookup')
        im = selected[0]
        request = check_inputs(data/im['file_name'],data/im['depth_file_name'],
                               im['depth_scale_m'],im['depth_invalid_code'])
        gt = [a for a in val['annotations'] if a['image_id'] == args.image_id]
        request.update(img_id=args.image_id,input_scope='official_val',annotation_sha256=sha256(annotation))
    else:
        if any(value is None for value in (args.rgb,args.depth,args.depth_scale,args.depth_invalid_code)):
            raise ValueError('Custom input requires RGB, uint16 depth, per-image scale and invalid code')
        request = check_inputs(args.rgb,args.depth,args.depth_scale,args.depth_invalid_code)
        request.update(img_id=args.custom_image_id,input_scope='custom_unlabelled')
    output.mkdir(parents=True)
    save_json(output/'request.json',request)
    for label, run in (('before',args.before_run),('after',args.after_run)):
        with (output/f'{label}.log').open('xb') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker',
                '--run',str(Path(run).resolve()),'--request',str(output/'request.json'),
                '--output',str(output/f'{label}_predictions.json')],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    render(output,request,gt,args.confidence)
    manifest = dict(status='complete',input=request,before_run=str(args.before_run),after_run=str(args.after_run),
        display_confidence=args.confidence,ground_truth_available=gt is not None,gt_sent_to_model=False,
        test_annotations_used=False,script_sha256=sha256(__file__),
        before=json.loads((output/'before_predictions.provenance.json').read_text()),
        after=json.loads((output/'after_predictions.provenance.json').read_text()),
        note='Demo outputs, not AP evaluation or a complete test submission; after does not imply better.')
    save_json(output/'manifest.json',manifest)
    print(json.dumps(dict(status='complete',output=str(output),ground_truth_available=gt is not None)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    p = sub.add_parser('compare')
    p.add_argument('--workspace',default='/workspace')
    p.add_argument('--before-run',default='/workspace/outputs/course_component_20261013_v2/baseline/runs/rgb')
    p.add_argument('--after-run',default='/workspace/outputs/course_component_20261013_v2/baseline/runs/gated_depth')
    p.add_argument('--image-id',type=int)
    p.add_argument('--rgb');p.add_argument('--depth');p.add_argument('--depth-scale',type=float)
    p.add_argument('--depth-invalid-code',type=int)
    p.add_argument('--custom-image-id',type=int,default=1)
    p.add_argument('--confidence',type=float,default=.3)
    p.add_argument('--output',required=True)
    p = sub.add_parser('worker')
    p.add_argument('--run',required=True);p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    args = parser.parse_args()
    worker(args) if args.action == 'worker' else demo(args)


if __name__ == '__main__':
    main()
