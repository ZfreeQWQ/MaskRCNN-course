auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            1024,
            1024,
        ),
        type='BatchFixedSizePad'),
]
custom_hooks = [
    dict(type='RejectNonfiniteLossHook'),
    dict(type='WholeRunPeakHook'),
    dict(type='CourseCurveHook'),
    dict(type='CourseDirectionsAuditHook'),
    dict(type='ChannelAuditHook'),
    dict(type='PointNoiseAuditHook'),
    dict(type='TinyMatchingAuditHook'),
    dict(type='LearningRateAuditHook'),
    dict(type='GeometryAuditHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'rgbd.data',
        'rgbd.model',
        'rgbd.void_head',
        'rgbd.metric',
        'rgbd.multiscale',
        'rgbd.foreground_head',
        'rgbd.course_data',
        'rgbd.course_copy_paste',
        'rgbd.course_pointnoise',
        'rgbd.course_tiny_matching',
        'rgbd.course_geometry_data',
        'rgbd.course_geometry_model',
    ])
data_preprocessor = dict(
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=False,
            size=(
                1024,
                1024,
            ),
            type='BatchFixedSizePad'),
    ],
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_seg=False,
    pad_size_divisor=32,
    seg_pad_value=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=6,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
dynamic_intervals = [
    (
        365001,
        368750,
    ),
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    1024,
    1024,
)
interval = 5000
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=10)
max_iters = 368750
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=None,
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    course_pilot=dict(
        enabled=True,
        max_queries=16,
        no_noise=False,
        point_noise_ratio=0.2,
        weight=0.5),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                img_pad_value=0,
                mask_pad_value=0,
                pad_mask=True,
                pad_seg=False,
                size=(
                    1024,
                    1024,
                ),
                type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        depth_max=10.0,
        geometry_mean=[
            0.3726963852020937,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        geometry_std=[
            0.24118472133914476,
            0.01687726006188841,
            1.0,
            1.0,
            1.0,
            0.01309585807441104,
            1.0,
        ],
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_seg=False,
        pad_size_divisor=32,
        seg_pad_value=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='GeometryDataPreprocessor'),
    depth_fusion=dict(enabled=False),
    geometry_options=dict(dropout=0.2, enabled=True, seed=20261013),
    init_cfg=None,
    multiscale_depth=dict(enabled=True, reliability_gate=True, source='depth'),
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes=1,
        type='MaskFormerFusionHead'),
    panoptic_head=dict(
        boundary_loss=dict(
            enabled=False, loss_weight=0.1, min_target_peak=0.5, radius=1),
        enforce_decoder_input_project=False,
        feat_channels=256,
        foreground_sampling=dict(
            area_threshold=1024.0, enabled=False, quota=64),
        ignore_void=False,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_queries=100,
        num_stuff_classes=0,
        num_things_classes=1,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        small_object_loss=dict(
            area_threshold=1024.0, enabled=False, max_weight=2.0),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True),
        type='ForegroundQuotaMask2FormerHead'),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=100,
        panoptic_on=False,
        semantic_on=False),
    tiny_matching=dict(area_threshold=64, enabled=True, quota=1024),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler')),
    type='GeometryMatchedMask2Former')
num_classes = 1
num_stuff_classes = 0
num_things_classes = 1
optim_wrapper = dict(
    clip_grad=dict(error_if_nonfinite=True, max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=3e-05,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone':
            dict(decay_mult=1.0, lr_mult=0.1),
            'geometry_fusion':
            dict(lr_mult=10.0),
            'level_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat':
            dict(decay_mult=0.0, lr_mult=1.0),
            'spatial_fusion':
            dict(decay_mult=1.0, lr_mult=10.0),
            'spatial_fusion.stages.0.1':
            dict(decay_mult=0.0, lr_mult=10.0),
            'spatial_fusion.stages.1.1':
            dict(decay_mult=0.0, lr_mult=10.0),
            'spatial_fusion.stages.2.1':
            dict(decay_mult=0.0, lr_mult=10.0),
            'spatial_fusion.stem.0.1':
            dict(decay_mult=0.0, lr_mult=10.0),
            'spatial_fusion.stem.1.1':
            dict(decay_mult=0.0, lr_mult=10.0)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=30000,
        gamma=0.1,
        milestones=[
            18000,
            24000,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=20261013)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img=''),
        data_root='/workspace/data/course_blender_20261013/prepared',
        filter_cfg=None,
        metainfo=dict(classes=('component', )),
        pipeline=[
            dict(
                geometry_index=
                '/workspace/outputs/course_m2f_G_20261027_recovery2/geometry_index.json',
                type='LoadGeometryRGBD'),
            dict(keep_ratio=True, scale=(
                1706,
                1024,
            ), type='GeometryResize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackGeometryRGBD'),
        ],
        test_mode=True,
        type='CourseRGBDCocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/workspace/data/course_blender_20261013/prepared/annotations/instances_val.json',
    classwise=True,
    metric='segm',
    prefix='course',
    proposal_nums=(
        1,
        10,
        100,
    ),
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=5000)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img=''),
        data_root='/workspace/data/course_blender_20261013/prepared',
        filter_cfg=None,
        metainfo=dict(classes=('component', )),
        pipeline=[
            dict(
                geometry_index=
                '/workspace/outputs/course_m2f_G_20261027_recovery2/geometry_index.json',
                type='LoadGeometryRGBD'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                ann_file='annotations/instances_train.json',
                area_threshold=1024,
                data_root='/workspace/data/course_blender_20261013/prepared',
                enabled=True,
                geometry_index=
                '/workspace/outputs/course_m2f_G_20261027_recovery2/geometry_index.json',
                max_instances=100,
                max_paste=3,
                prob=0.5,
                type='GeometryCopyPaste'),
            dict(prob=0.5, type='GeometryFlip'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.8,
                    1.25,
                ),
                resize_type='GeometryResize',
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    1024,
                    1024,
                ),
                crop_type='absolute',
                recompute_bbox=True,
                type='GeometryCrop'),
            dict(
                by_mask=True,
                keep_empty=False,
                min_gt_bbox_wh=(
                    1e-05,
                    1e-05,
                ),
                type='FilterAnnotations'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'course_copy_paste_activity',
                ),
                type='PackGeometryRGBD'),
        ],
        test_mode=False,
        type='CourseRGBDCocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='Resize',
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            1024,
            1024,
        ),
        crop_type='absolute',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(
        by_mask=True,
        min_gt_bbox_wh=(
            1e-05,
            1e-05,
        ),
        type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img=''),
        data_root='/workspace/data/course_blender_20261013/prepared',
        filter_cfg=None,
        metainfo=dict(classes=('component', )),
        pipeline=[
            dict(
                geometry_index=
                '/workspace/outputs/course_m2f_G_20261027_recovery2/geometry_index.json',
                type='LoadGeometryRGBD'),
            dict(keep_ratio=True, scale=(
                1706,
                1024,
            ), type='GeometryResize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackGeometryRGBD'),
        ],
        test_mode=True,
        type='CourseRGBDCocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/workspace/data/course_blender_20261013/prepared/annotations/instances_val.json',
    classwise=True,
    metric='segm',
    prefix='course',
    proposal_nums=(
        1,
        10,
        100,
    ),
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/workspace/outputs/course_m2f_G_20261027_recovery2/formal/runs/geometry_long'
