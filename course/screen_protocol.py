"""Audited, narrow exceptions for explicitly declared training-factor ablations."""
from copy import deepcopy


ALLOWED_TRAINING_CHANGES = {"INPUT.MIN_SCALE", "MODEL.MASK_FORMER.TRAIN_NUM_POINTS",
                          "MODEL.META_ARCHITECTURE", "COURSE.COVERAGE.MATCH_POINTS",
                          "COURSE.COVERAGE.LOSS_POINTS"}


def flatten(mapping, prefix=""):
    result={}
    for key,value in mapping.items():
        path=prefix+key
        if isinstance(value,dict):result.update(flatten(value,path+"."))
        else:result[path]=value
    return result


def compare_protocol(candidate,reference,allowed=()):
    allowed=set(allowed)
    if not allowed <= ALLOWED_TRAINING_CHANGES:
        raise ValueError("Cannot relax data split, seed, optimizer, LR, resolution or other protocol checks")
    ignored_model={"MODEL.WEIGHTS","MODEL.BACKBONE.NAME","MODEL.SEM_SEG_HEAD.NAME",
                   "MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME"}
    selected=[]
    for original in (candidate,reference):
        config=deepcopy(original)
        keys=("INPUT","SOLVER","DATALOADER","DATASETS","TEST","SEED","MODEL")
        values=flatten({k:config[k] for k in keys})
        values={k:v for k,v in values.items() if k not in ignored_model}
        # Query/local architecture settings are opt-in modules tested separately
        # by strict transfer, RNG and GPU parity. All data/loss opt-ins stay fixed.
        course={k:v for k,v in config["COURSE"].items() if k not in ("NAME","QUERY","LOCAL")}
        # Old immutable protocols predate this opt-in; absence means disabled.
        course.setdefault("COVERAGE",dict(MATCH_POINTS=0,LOSS_POINTS=0,MAX_MATCH_REPLACEMENTS=256))
        values.update(flatten(course,"COURSE."))
        selected.append(values)
    a,b=selected
    differences={k:{"candidate":a.get(k),"control":b.get(k)} for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    unexpected=set(differences)-allowed
    if unexpected:raise ValueError("Unexpected protocol changes: "+str(sorted(unexpected)))
    return {k:differences[k] for k in sorted(differences)}
