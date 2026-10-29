"""Independent opt-in switches, with an unchanged all-off baseline."""


def apply_optimizations(cfg, args):
    modules = []
    if args.small_object_crop:
        modules.append('optimizations.small_object_crop')
        pipeline = cfg.train_dataloader.dataset.pipeline
        for transform in pipeline:
            if transform['type'] == 'RandomCrop':
                transform.update(type='SmallObjectAwareCrop', prob=args.small_crop_prob,
                                 small_area_thr=args.small_area_threshold, area_mode='original')
                break
        else:
            raise ValueError('Small-object crop needs a RandomCrop in the training pipeline')
    if args.small_object_loss or args.boundary_loss:
        modules.append('optimizations.weighted_boundary_head')
        cfg.model.panoptic_head.type = 'WeightedBoundaryMask2FormerHead'
        cfg.model.panoptic_head.small_object_loss = dict(
            enabled=args.small_object_loss, area_threshold=args.small_area_threshold,
            max_weight=args.small_loss_max_weight)
        cfg.model.panoptic_head.boundary_loss = dict(
            enabled=args.boundary_loss, loss_weight=args.boundary_loss_weight,
            radius=1, min_target_peak=0.5)
    if modules:
        cfg.custom_imports = dict(imports=modules, allow_failed_imports=False)
    return cfg
