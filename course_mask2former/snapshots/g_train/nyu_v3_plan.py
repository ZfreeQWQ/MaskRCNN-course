"""Eight cells for three new switches, plus RGB and matched-capacity controls."""
from nyu_v2_plan import FLAGS as V2_FLAGS

NEW_FLAGS = ('relative_depth', 'modality_dropout', 'small_matching')
FLAGS = (*V2_FLAGS, *NEW_FLAGS)
ANCHOR = ('multiscale_depth', 'reliability_gate', 'foreground_points')
VARIANTS = [('rgb', 'RGB 基线', ())]
for mask in range(8):
    selected = tuple(flag for bit, flag in enumerate(NEW_FLAGS) if mask & (1 << bit))
    label = '门控深度+前景采样' + ''.join('+' + name for flag, name in
        zip(NEW_FLAGS, ('相对深度', '分支丢弃', '小目标匹配')) if flag in selected)
    VARIANTS.append((f'm{mask}', label, (*ANCHOR, *selected)))
VARIANTS.append(('rgb_capacity_control', 'RGB 同容量分支+三个新策略',
                 (*ANCHOR, *NEW_FLAGS, 'rgb_control')))
CONTRASTS = [('m0', 'rgb'), ('m1', 'm0'), ('m2', 'm0'), ('m4', 'm0'),
             ('m7', 'm0'), ('m7', 'rgb_capacity_control')]


def experiment_jobs(seeds):
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    jobs = []
    for seed in seeds:
        for variant, label, enabled in VARIANTS:
            identity = f's{seed}_{variant}'
            jobs.append(dict(id=identity, seed=seed, variant=variant, label=label, mask=0,
                flags={key: key in enabled for key in FLAGS}, output_rel='runs/' + identity))
    return jobs
