"""Evaluate a fully completed setting while other GPU experiments are still running."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from run_round1 import ROOT,SETTINGS,SHARDS,ANNOTATIONS,merge,evaluate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('setting',choices=[s['name'] for s in SETTINGS])
    args=parser.parse_args()
    setting=next(s for s in SETTINGS if s['name']==args.setting)
    for i in range(SHARDS):
        if not (ROOT/'val'/setting['name']/f'shard{i}'/'complete.json').exists():
            parser.error('All inference shards must complete first')
    jobs=[]
    for variant in ['legacy','mask_first']+(['refined_small'] if setting['refine'] else []):
        name=setting['name']+'_'+variant
        output=ROOT/'merged_val'/(name+'.json')
        shards=[ROOT/'val'/setting['name']/f'shard{i}'/(variant+'.json') for i in range(SHARDS)]
        manifest=merge(shards,ANNOTATIONS,'val',output)
        evaluation=ROOT/'evaluation'/(name+'.json')
        evaluation.parent.mkdir(exist_ok=True)
        jobs.append((name,str(output),manifest,str(evaluation)))
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(evaluate,jobs))
