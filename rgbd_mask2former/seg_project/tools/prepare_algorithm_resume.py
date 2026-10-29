#!/usr/bin/env python
"""Stage a full final checkpoint in a new output without altering the parent."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-output',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    source,output=Path(args.source_output),Path(args.output)
    checkpoint=source/'model_final.pth'
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    trainer=data['trainer']
    optimizer=trainer['_trainer']['optimizer']
    if not optimizer['state'] or not any('LRScheduler' in k for k in trainer['hooks']):
        raise AssertionError('Full optimizer/scheduler checkpoint is required')
    completed=data['iteration']+1
    if trainer['iteration']+1 != completed:
        raise AssertionError('Top-level and trainer iteration mismatch')
    output.mkdir(parents=True,exist_ok=False)
    seed=output/'resume_seed.pth'
    shutil.copy2(checkpoint,seed)
    expected=sha(checkpoint)
    if sha(seed)!=expected:
        raise AssertionError('Checkpoint staging checksum mismatch')
    copied=[]
    if not args.smoke:
        for name in ('best_metrics.json','best_segm_APs.pth','latest_eval_metrics.json'):
            if (source/name).exists():
                shutil.copy2(source/name,output/name)
                copied.append(name)
    (output/'last_checkpoint').write_text(seed.name)
    report={'valid':True,'source_checkpoint':str(checkpoint),'seed_checkpoint':str(seed),
            'sha256':expected,'completed_iterations':completed,'optimizer_states':len(optimizer['state']),
            'preserve_optimizer':True,'smoke':args.smoke,'copied_parent_best':copied}
    (output/'resume_seed.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
