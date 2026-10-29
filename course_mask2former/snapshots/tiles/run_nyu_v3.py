"""v3 hypothesis tests; parameter-free representation, regularization, matching."""
import run_nyu_v2 as v2

NEW_FLAGS = ('relative_depth', 'modality_dropout', 'small_matching')


def parser():
    p = v2.parser()
    p.description = __doc__
    for flag in NEW_FLAGS:
        p.add_argument('--' + flag.replace('_', '-'), action='store_true')
    p.add_argument('--dropout-prob', type=float, default=.2)
    p.add_argument('--matching-fraction', type=float, default=.25)
    return p


def validate_args(args):
    v2.validate_args(args)
    if (args.relative_depth or args.modality_dropout) and not args.multiscale_depth:
        raise ValueError('Relative depth/dropout require multiscale fusion')
    if not 0 < args.dropout_prob < 1 or not 0 < args.matching_fraction <= .5:
        raise ValueError('Invalid dropout probability / matching fraction')


def build_config(args, categories):
    validate_args(args)
    cfg = v2.build_config(args, categories)
    cfg.model.type = 'AdvancedRGBDMask2Former'
    cfg.model.advanced_depth = dict(relative_depth=args.relative_depth,
        modality_dropout=args.modality_dropout, dropout_prob=args.dropout_prob)
    cfg.model.panoptic_head.type = 'SmallAwareMatchingHead'
    cfg.model.panoptic_head.small_matching = dict(enabled=args.small_matching,
        fraction=args.matching_fraction, area_threshold=args.small_area_threshold)
    cfg.custom_imports.imports += ['rgbd.advanced_model', 'rgbd.matching_head']
    return cfg


def evidence(runner, args):
    depth = dict(runner.model.advanced_counters)
    matching = dict(runner.model.panoptic_head.matching_counters)
    count = args.iters * args.batch_size
    if depth['relative_training_samples'] != (count if args.relative_depth else 0):
        raise RuntimeError('Relative representation execution count mismatch')
    if depth['dropout_training_samples'] != (count if args.modality_dropout else 0):
        raise RuntimeError('Dropout execution count mismatch')
    if not args.modality_dropout and depth['dropped_samples']:
        raise RuntimeError('Disabled dropout must not drop samples')
    if args.modality_dropout and args.iters >= 1000 and not 0 < depth['dropped_samples'] < count:
        raise RuntimeError('Long run must exercise both kept and dropped samples')
    if not args.small_matching and any(matching.values()):
        raise RuntimeError('Disabled matching was invoked')
    if args.small_matching and args.iters >= 1000 and matching['augmented_calls'] == 0:
        raise RuntimeError('Long run never exercised small-aware matching')
    return dict(advanced_depth_counters=depth, small_matching_counters=matching)


def main():
    args = parser().parse_args()
    validate_args(args)
    v2.main(args=args, config_builder=build_config, extra_sources=('run_nyu_v3.py',), report_callback=evidence)


if __name__ == '__main__':
    main()
