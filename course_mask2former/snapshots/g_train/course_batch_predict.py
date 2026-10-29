"""Bounded, annotation-free model inference on 1..8 official val images.

Not a test exporter or Kaggle submission tool. Uses bundled, hash-pinned source;
weights and prepared data stay external. Resume never changes the run identity.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from course_demo import check_inputs, encode_predictions, sha256

VAL_SHA = '71612ebf1d548eb85e146e97de4d211578d2f87d5273f9b0f5e2d48b1af2ea2d'
PROTOCOL = 'component-official-train-val-standard-coco-mask-maxdets100'


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inside(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):
        raise ValueError('Unsafe relative path: ' + str(relative))
    return path


def verify_bundle(bundle, model):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/'bundle_manifest.json').read_text())
    for relative, expected in manifest['file_sha256'].items():
        if sha256(inside(bundle, relative)) != expected:
            raise ValueError('Bundle file hash mismatch: ' + relative)
    entry = manifest['models'][model]
    metadata = inside(bundle, entry['metadata'])
    report = json.loads((metadata/'report.json').read_text())
    plan = json.loads((metadata/'plan.json').read_text())
    source = inside(bundle, entry['source'])
    for relative, expected in plan['source_sha256'].items():
        if sha256(inside(source, relative)) != expected:
            raise ValueError('Frozen source mismatch: ' + relative)
    n = report['final_iteration']
    if (report['status'] != 'passed' or report['args']['functional_smoke']
            or report['protocol'] != PROTOCOL or n != report['args']['iters']
            or report['final_optimizer_steps'] != [n]
            or report['test_inputs_used'] or report['test_annotations_used']
            or report['val_annotations_sha256'] != VAL_SHA):
        raise ValueError('Require accepted formal course model')
    return report, metadata/'config.py', source, sha256(bundle/'bundle_manifest.json')


def select_val_inputs(data_root, ids):
    if (not 1 <= len(ids) <= 8 or len(ids) != len(set(ids))
            or any(type(i) is not int or i <= 0 for i in ids)):
        raise ValueError('Choose 1..8 distinct positive official val image IDs')
    data_root = Path(data_root).resolve()
    if 'test' in data_root.parts:
        raise ValueError('Test paths are forbidden')
    ann = data_root/'annotations/instances_val.json'
    if sha256(ann) != VAL_SHA:
        raise ValueError('Official val annotation SHA mismatch')
    document = json.loads(ann.read_text())
    images = {im['id']: im for im in document['images']}
    if len(images) != 2584:
        raise ValueError('Unexpected val image count')
    requests = []
    for identity in ids:
        im = images.get(identity)
        if im is None:
            raise ValueError('ID not in official val; no fallback to train/test')
        rgb = inside(data_root, im['file_name'])
        depth = inside(data_root, im['depth_file_name'])
        if (not rgb.is_relative_to(data_root/'rgb/val')
                or not depth.is_relative_to(data_root/'depth/val')):
            raise ValueError('Input must stay within official val directories')
        request = check_inputs(rgb, depth, im['depth_scale_m'], im['depth_invalid_code'])
        if request['original_shape'] != [1024,1024]:
            raise ValueError('Expected course 1024x1024 input')
        request.update(img_id=identity, input_scope='official_val',
                       annotation_sha256=VAL_SHA)
        requests.append(request)
    # The annotations member is not used to prepare inference samples.
    return requests


def validate_rows(rows, image_id, shape):
    import numpy as np
    from pycocotools import mask as mu
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError('Expected a list of <=100 predictions')
    previous = 1.0
    for row in rows:
        if (type(row.get('image_id')) is not int or row['image_id'] != image_id
                or type(row.get('category_id')) is not int or row['category_id'] != 1):
            raise ValueError('Wrong image/category ID')
        score = row['score']
        if (type(score) not in (int,float) or not math.isfinite(score)
                or not 0 <= score <= 1 or score > previous):
            raise ValueError('Invalid or unsorted score')
        previous = score
        rle = row['segmentation']
        if (rle['size'] != list(shape) or not isinstance(rle['counts'], str)
                or not rle['counts'].isascii()):
            raise ValueError('Invalid RLE dimensions/counts')
        decoded = mu.decode(rle)
        if list(decoded.shape) != list(shape) or not decoded.any():
            raise ValueError('Empty or malformed mask')
        canonical = mu.encode(np.asfortranarray(decoded.astype(np.uint8)))
        if canonical['counts'].decode('ascii') != rle['counts']:
            raise ValueError('Noncanonical RLE')
        box = np.asarray(row['bbox'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not np.allclose(
                box, mu.toBbox(rle), rtol=0, atol=1e-6):
            raise ValueError('BBox does not agree with mask')
    return len(rows)


@contextmanager
def output_lock(output, resume):
    output = Path(output)
    if resume:
        if not output.is_dir() or not (output/'request.json').is_file():
            raise ValueError('Resume needs an existing initialized output')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield output
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class BatchStore:
    """Used only under output_lock. Per-image records commit atomically."""
    def __init__(self, output, identity, resume):
        self.output = Path(output)
        self.identity = identity
        self.fingerprint = stable_hash(identity)
        self.requests = {str(r['img_id']): r for r in identity['inputs']}
        if len(self.requests) != len(identity['inputs']):
            raise ValueError('Duplicate image IDs')
        header = dict(identity=identity, fingerprint=self.fingerprint)
        if resume:
            if json.loads((self.output/'request.json').read_text()) != header:
                raise ValueError('Resume identity changed')
        else:
            atomic_json(self.output/'request.json', header)
            (self.output/'images').mkdir()
        self.records = {}
        for file in sorted((self.output/'images').glob('*.json')):
            key = file.stem
            record = json.loads(file.read_text())
            if (key not in self.requests or record['fingerprint'] != self.fingerprint
                    or record['image_id'] != self.requests[key]['img_id']):
                raise ValueError('Foreign/corrupt image record')
            validate_rows(record['rows'], record['image_id'], self.requests[key]['original_shape'])
            if stable_hash(record['rows']) != record['rows_sha256']:
                raise ValueError('Prediction payload hash mismatch')
            self.records[key] = record
        self.initial_ids = sorted(self.records)

    def commit(self, request, rows):
        key = str(request['img_id'])
        if key in self.records:
            raise ValueError('Refusing to overwrite completed image')
        validate_rows(rows, request['img_id'], request['original_shape'])
        record = dict(image_id=request['img_id'], fingerprint=self.fingerprint,
                      rows_sha256=stable_hash(rows), rows=rows)
        atomic_json(self.output/'images'/f'{key}.json', record)
        self.records[key] = record
        self.progress('partial')

    def progress(self, status):
        value = dict(status=status, fingerprint=self.fingerprint,
            requested_ids=[r['img_id'] for r in self.identity['inputs']],
            completed_ids=[r['img_id'] for r in self.identity['inputs'] if str(r['img_id']) in self.records],
            resumed_existing_ids=[int(i) for i in self.initial_ids],
            test_inputs_used=False, test_annotations_used=False, gt_sent_to_model=False,
            note='Bounded val export only; not AP evaluation or final test predictions.')
        if len(self.records) == len(self.requests):
            rows = [row for r in self.identity['inputs'] for row in self.records[str(r['img_id'])]['rows']]
            path = self.output/'predictions.json'
            if path.exists():
                if json.loads(path.read_text()) != rows:
                    raise ValueError('Existing aggregate differs from image records')
            else:
                atomic_json(path, rows)
            value.update(status='complete', predictions_sha256=sha256(path),
                         images=len(self.records), instances=len(rows),
                         empty_ids=[int(k) for k,v in self.records.items() if not v['rows']])
        atomic_json(self.output/'status.json', value)
        return value


def predict(args):
    report, config, source, bundle_sha = verify_bundle(args.bundle, args.model)
    checkpoint = Path(args.checkpoint).resolve()
    if sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('External checkpoint SHA mismatch')
    requests = select_val_inputs(args.data_root, args.image_ids)
    if not 1 <= args.max_new_images <= 8:
        raise ValueError('max-new-images must be 1..8')
    identity = dict(model=args.model, bundle_sha256=bundle_sha,
        config_sha256=sha256(config), checkpoint_sha256=report['checkpoint_sha256'],
        script_sha256=sha256(__file__), inputs=requests, scope='official_val_only')
    with output_lock(Path(args.output).resolve(), args.resume) as output:
        store = BatchStore(output, identity, args.resume)
        pending = [r for r in requests if str(r['img_id']) not in store.records]
        if not pending:
            print(json.dumps(store.progress('complete')))
            return
        # Load only this model's bundled source; never the original run's source.
        if any(k == 'rgbd' or k.startswith('rgbd.') for k in sys.modules):
            raise RuntimeError('Use a fresh process per source snapshot')
        sys.path.insert(0, str(source))
        import torch
        from mmengine.config import Config
        from mmengine.dataset import Compose, pseudo_collate
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(str(config))
        import_modules_from_strings(**cfg.custom_imports)
        pipeline = [dict(t) for t in cfg.val_dataloader.dataset.pipeline if t['type'] != 'LoadAnnotations']
        if [t['type'] for t in pipeline] != ['LoadCourseRGBDImage','RGBDResize','PackRGBDDetInputs']:
            raise ValueError('Unexpected inference pipeline')
        compose = Compose(pipeline)
        state = torch.load(checkpoint, map_location='cpu')
        trained = Config.fromstring(state['meta'].get('cfg',state['meta'].get('config')),'.py')
        if trained.model.to_dict() != cfg.model.to_dict() or state['meta']['iter'] != report['final_iteration']:
            raise ValueError('Checkpoint/model configuration mismatch')
        model = MODELS.build(cfg.model)
        model.load_state_dict(state['state_dict'], strict=True)
        del state
        model.cuda().eval()
        torch.backends.cudnn.benchmark = False
        for request in pending[:args.max_new_images]:
            # Recheck content immediately before inference, including on resume.
            checked = check_inputs(request['img_path'],request['depth_file'],
                                   request['depth_scale_m'],request['depth_invalid_code'])
            if any(checked[k] != request[k] for k in checked):
                raise ValueError('Input changed during inference')
            sample = compose(dict(request, scene='val_export'))
            data = sample['data_samples']
            if 'gt_instances' in data:
                if len(data.gt_instances):
                    raise ValueError('GT must not enter inference')
                del data.gt_instances
            with torch.inference_mode():
                pred = model.test_step(pseudo_collate([sample]))[0].pred_instances.cpu()
            if pred.labels.numel() and not bool((pred.labels == 0).all()):
                raise ValueError('Non-component label')
            if list(pred.masks.shape[-2:]) != request['original_shape']:
                raise ValueError('Wrong output grid')
            rows = encode_predictions(pred.masks.numpy(),pred.scores.numpy(),request['img_id'])
            store.commit(request, rows)
        print(json.dumps(store.progress('partial')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--model',choices=['control_30k','boundary_30k'],default='control_30k')
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--image-ids',type=int,nargs='+',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-new-images',type=int,default=8)
    args=p.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
