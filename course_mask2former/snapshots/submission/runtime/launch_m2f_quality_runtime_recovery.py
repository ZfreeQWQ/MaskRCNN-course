"""Same budget and same numerical source; deduct the previous GPU lease."""
import hashlib,inspect,json,shutil,sys,time
from pathlib import Path
START=1792574220;NAME='course_m2f_quality_runtime_20261021_v2'
NEW=('recover_m2f_quality_runtime.py','launch_m2f_quality_runtime_recovery.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path('/mnt/nfs/fenghaibin/AI_Class/outputs');out=root/NAME;prior=root/'course_m2f_quality_runtime_20261021_v1'
    assert time.time()<START+600 and not (out/'plan.json').exists()
    lease=json.loads((prior/'gpu_released.json').read_text());used=lease['seconds'];assert lease['original_lock_released'] and used<100
    inherited=json.loads((prior/'plan.json').read_text())['source_sha256']
    for n,h in inherited.items():
        assert sha(prior/'source'/n)==h and n not in NEW and not (out/'source'/n).exists(),n
        shutil.copy2(prior/'source'/n,out/'source'/n)
    assert len(list((out/'source').glob('*.py')))==67
    import launch_m2f_fine512_fullval as parent
    source=inspect.getsource(parent)
    changes={
        'from run_m2f_fine512_fullval import':'from recover_m2f_quality_runtime import',
        "['run_m2f_fine512_fullval.py',stage]":"['recover_m2f_quality_runtime.py',stage]",
        "'ai-m2f-512val-'":"'ai-m2f-qualityrecover-'",
        'gpu_max_seconds=5700':f'gpu_max_seconds={600-used!r},prior_gpu_seconds={used!r},cumulative_gpu_max_seconds=600',
        'm2f_calls=10633':'m2f_calls=297',
        'full_val_evaluations=1':'full_val_evaluations=0',
        'baseline_aps=.41399673960953054':'verified_candidate_aps=.4176790163964242',
        'acquired+5680':f'acquired+{580-used!r}',
        "end-acquired>=5100,'Insufficient conservative85min headroom'":"end-acquired>=330,'Insufficient conservative330sec runtime headroom'",
        "['seconds']<=5700":f"['seconds']<={600-used!r}",
    }
    for a,b in changes.items():assert source.count(a)==1,a;source=source.replace(a,b)
    ns={'__name__':'runtime_recovery_supervisor'};exec(compile(source,'<bounded-runtime-recovery>','exec'),ns)
    assert ns['START']==START and ns['DEADLINE']==START+1800
    ns['main']()
if __name__=='__main__':main()
