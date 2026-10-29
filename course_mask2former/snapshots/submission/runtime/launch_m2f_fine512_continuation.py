"""Reuse the accepted owner-scoped supervisor with immutable new bounds."""
import inspect
import launch_m2f_fine512_fullval as parent
import continue_m2f_fine512_fullval as continuation

def build():
    source=inspect.getsource(parent)
    changes={
        'from run_m2f_fine512_fullval import':'from continue_m2f_fine512_fullval import',
        "['run_m2f_fine512_fullval.py',stage]":"['continue_m2f_fine512_fullval.py',stage]",
        'gpu_max_seconds=5700':'gpu_max_seconds=8100',
        'm2f_calls=10633':'m2f_calls=10505,reused_capture_images=32,old_inferred_calls=128',
        'acquired+5680':'acquired+8080',
        "end-acquired>=5100,'Insufficient conservative85min headroom'":"end-acquired>=7800,'Insufficient conservative130min headroom'",
        "['seconds']<=5700":"['seconds']<=8100",
    }
    for a,b in changes.items():
        assert source.count(a)==1,a
        source=source.replace(a,b)
    ns={'__name__':'frozen_continuation_supervisor'}
    exec(compile(source,'<owner-scoped-continuation-supervisor>','exec'),ns)
    assert ns['START']==1792533242 and ns['DEADLINE']==1792544042
    return ns

if __name__=='__main__':build()['main']()
