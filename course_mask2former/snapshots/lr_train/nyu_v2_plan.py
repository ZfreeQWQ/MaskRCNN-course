"""Predeclared v2 recipes. No scores are consulted when scheduling these jobs."""
FLAGS = ('depth_fusion', 'small_object_crop', 'small_object_loss', 'boundary_loss',
         'multiscale_depth', 'reliability_gate', 'rgb_control', 'foreground_points')
VARIANTS = [
    ('rgb', 'RGB 基线', ()),
    ('old_best', '上一轮最佳：早期深度+裁剪+加权', ('depth_fusion', 'small_object_crop', 'small_object_loss')),
    ('multiscale', '多尺度深度', ('multiscale_depth',)),
    ('gated_depth', '多尺度深度+空间门控', ('multiscale_depth', 'reliability_gate')),
    ('foreground', '小目标前景采样', ('foreground_points',)),
    ('gated_foreground', '门控深度+前景采样', ('multiscale_depth', 'reliability_gate', 'foreground_points')),
    ('gated_foreground_crop_weight', '门控深度+前景采样+裁剪+加权',
        ('multiscale_depth', 'reliability_gate', 'foreground_points', 'small_object_crop', 'small_object_loss')),
    ('rgb_capacity_control', '同容量 RGB 分支+前景采样+裁剪+加权',
        ('multiscale_depth', 'reliability_gate', 'foreground_points', 'small_object_crop', 'small_object_loss', 'rgb_control')),
]
CONTRASTS = [('multiscale', 'rgb'), ('gated_depth', 'multiscale'), ('foreground', 'rgb'),
             ('gated_foreground', 'gated_depth'), ('gated_foreground_crop_weight', 'gated_foreground'),
             ('gated_foreground_crop_weight', 'rgb_capacity_control'), ('old_best', 'rgb')]


def experiment_jobs(seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    result = []
    for seed in seeds:
        for variant, label, enabled in VARIANTS:
            flags = {key: key in enabled for key in FLAGS}
            identity = f's{seed}_{variant}'
            result.append(dict(id=identity, seed=seed, variant=variant, label=label,
                mask=sum(1 << i for i, key in enumerate(FLAGS[:4]) if flags[key]),
                flags=flags, output_rel='runs/' + identity))
    return result
