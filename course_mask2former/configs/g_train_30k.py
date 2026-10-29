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
experiment_provenance = dict(
    class_initialization='mean of 80 foreground COCO rows; preserve no-object',
    implementation_sha256=dict({
        'nyu_v4_plan.py':
        'cd33d50367201b3d26e756aa560cc5990650c473145d3795dba55aa43519e8a6',
        'optimizations/__init__.py':
        'd9cb52c525c68cfd1ddcaf42b64e39ad832764061454719a5580a7c37ce42dba',
        'optimizations/configure.py':
        'd5ddfee1bf6e9cab1e1979ac84dc8f44bccfdd668c44f1870f91e2dad0e30600',
        'optimizations/small_object_crop.py':
        '875dab77c258bda4c07a7f2bedb4a7059b9ece487d52418fcb1cadc5ededad6c',
        'optimizations/weighted_boundary_head.py':
        'b239c8b17658db46929dfdc3fd25757675eae2c0c54027a058eb7e7b75ad8c73',
        'rgbd/__init__.py':
        '5d62e89d95beec4446ae7a8be01e7e37f17f878c1baafb2331fc0fbc0b835fdd',
        'rgbd/advanced_model.py':
        'fbab581711793bdc2ed317cbddb2ba370393c2fb0dcf4c9cf849d7aeaa679d3d',
        'rgbd/course_channel.py':
        '56b16a8565787de2d006895784a74dff60f2f7a0dc559ec86fe7bc0fbe2f0827',
        'rgbd/course_copy_paste.py':
        '6bdddff93f238bb69892eafef29f998930664cc3c4dee70b1b82272dbb7b9515',
        'rgbd/course_data.py':
        'b9b5b4c0ebf357cd13717a7ef91058f586ff08ca837a198ef9790997695148c0',
        'rgbd/course_feature_models.py':
        '749049ba0836b81e4e30fd16a9b6135744021bb8228a01c8386336248da7e5fd',
        'rgbd/course_geometry_data.py':
        'b96ac527d0c5f7c467a47204c28c5f6daf9788fb71ca942fc5de93f67887c113',
        'rgbd/course_geometry_model.py':
        'f441e24992dc9ff54a900f24f06684f5ab0c2fb8f5a6b3aa335e50553567cceb',
        'rgbd/course_head.py':
        'a18f00f448e4b20c5fc466d777fd5bce8a5974f6510e814e144eca6047f692b6',
        'rgbd/course_pilot.py':
        '95993b84c4597935542daad2922febd64ebc73ac078319344441665c79bb82b9',
        'rgbd/course_pointnoise.py':
        '9bfc1e990043699ee6eb45f9f3f86951b819d759be3696423205a8085b6851d1',
        'rgbd/course_tiny_matching.py':
        '36f92376237f2c4b97197b31cd02f4b229475d52c3210068525a5de69e9cb8e7',
        'rgbd/data.py':
        'e2208a7e726a84a471f4edcb61f6114fcf83ba4be03aa105b69895f0c1ffa054',
        'rgbd/foreground_head.py':
        '3f0de7e1f8cc221674a68510ff447916267b5c2fa47d286ae4fd230e690cbb73',
        'rgbd/matching_head.py':
        '2c7a7bbc69731282053a56035d4a606bebcd74b6793b52bf7c8b512a27c5aec1',
        'rgbd/metric.py':
        '867bb5fa763a7568ecdb2641b387c7c8529a084da4b4782ad20de062b1604e9a',
        'rgbd/model.py':
        '180f3f0eff3e140220d0db693ac936b77530b2b11d4544b22a2c5a2f89bea00b',
        'rgbd/multiscale.py':
        '54b7752c24fe01bbc1a9a01092216de6fc71bbd5b06da5c13d28f5f979ee950b',
        'rgbd/small_boundary_head.py':
        '42d43dac3ebc83d50621c6c4c2000acca51b18617937888e9ce1be98e0520d58',
        'rgbd/void_head.py':
        '58fa7b144940eb4bb1bd6595440fec487f6d9e4064dcddb0d8a5384be98ec7c8',
        'run_coco_experiment.py':
        '53e6ad9bae96d94b8cac4cf5b33d4b9c07ea1758891c7ad2162185dff85db43b',
        'run_course_experiment.py':
        '6c906de4f2c471c493d55d7fd8e6f1ba0773e69861cff3b4d70224d483dfb516',
        'run_nyu_experiment.py':
        'cf909d1cdf45bf1d874caff45a364636ea486fff7393c51cc2eaee899188cdb4',
        'run_nyu_v2.py':
        'b3c5f45ad0f6020810cfb3a7f18f6665a352424267da1d9433f1e9c101bebdb1',
        'train_course_directions.py':
        '4285ea1bb63a02b58bd29753b95fab27ecf96e36afed8ae57157c3f957e53f05'
    }),
    protocol='component-official-train-val-standard-coco-mask-maxdets100',
    source_checkpoint_sha256=
    '41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879',
    test_annotations_used=False,
    test_inputs_used=False,
    train_annotations_sha256=
    '1f72e6564579f9257ab1ef54044207eccaeb452affd33f05b19a6ef00105e882',
    val_annotations_sha256=
    '71612ebf1d548eb85e146e97de4d211578d2f87d5273f9b0f5e2d48b1af2ea2d')
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
