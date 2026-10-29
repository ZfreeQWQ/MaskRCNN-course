"""Pure configuration contract: one declared numerical difference."""
import copy
KEYS=('model','train_cfg','optim_wrapper','param_scheduler','randomness','train_dataloader',
      'val_dataloader','val_evaluator','auto_scale_lr','env_cfg','custom_imports','custom_hooks')
def check_delta(old,new):
    old=copy.deepcopy(old);new=copy.deepcopy(new)
    assert old['optim_wrapper']['optimizer']['lr']==1e-5
    assert new['optim_wrapper']['optimizer']['lr']==3e-5
    new['optim_wrapper']['optimizer']['lr']=1e-5
    hooks=new['custom_hooks'];assert sum(h.get('type')=='LearningRateAuditHook' for h in hooks)==1
    new['custom_hooks']=[h for h in hooks if h.get('type')!='LearningRateAuditHook']
    changed=[k for k in KEYS if old[k]!=new[k]]
    assert not changed,changed
    return dict(status='passed',learning_rate_only=True,baseline_lr=1e-5,new_lr=3e-5,
                compared_keys=KEYS,allowed_numeric_path='optim_wrapper.optimizer.lr',
                additional_hook='LearningRateAuditHook (read-only; no RNG)')
