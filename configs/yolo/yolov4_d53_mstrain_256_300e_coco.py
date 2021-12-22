checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

class_name = ['0']

# model settings
model = dict(
    type='YOLOV4',
    backbone=dict(
        type='YOLOV4Backbone'
    ),
    neck=dict(
        type='YOLOV4Neck',
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV4Head',
        num_classes=1,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            # base_sizes=[[(116, 90), (156, 198), (373, 326)],
            #             [(30, 61), (62, 45), (59, 119)],
            #             [(10, 13), (16, 30), (33, 23)]],
            # base_sizes=[
            #     [[30, 28], [33, 17], [21, 25]],
            #     [[24, 17], [18, 17], [14, 21]],
            #     [[20, 12], [12, 15], [11, 11]]],
            base_sizes=[
                [[2*30, 2*28], [2*33, 2*17], [2*21, 2*25]],
                [[2*24, 2*17], [2*18, 2*17], [2*14, 2*21]],
                [[2*20, 2*12], [2*12, 2*15], [2*11, 2*11]]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.3,
            neg_iou_thr=0.3,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.4),
        max_per_img=100))
# dataset settings
dataset_type = 'MyCocoDataset'
# data_root = '/Users/kyanchen/Code/mmdetection/data/multi_label'
# data_root = r'M:\Tiny_Ship\20211214_All_P_Slice_Data'
data_root = '/data/kyanchen/det/data/Tiny_P'
img_norm_cfg = dict(mean=[52.27434974492982, 69.82640643452488, 79.01744958336889],
                    std=[2.7533898592345842, 2.634773617140497, 2.172352333590293], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 1.2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.01, 0.05, 0.1),
        min_crop_size=0.7),
    # dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='Resize', img_scale=[(256, 256)], keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='PhotoMetricDistortion',
         brightness_delta=20,
         contrast_range=(0.7, 1.3),
         saturation_range=(0.7, 1.3),
         hue_delta=15
         ),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')
         )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=100,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        ann_file='../data/tiny_ship/tiny_train.json',
        img_prefix=data_root+'/train',
        classes=class_name,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='../data/tiny_ship/tiny_val.json',
        img_prefix=data_root+'/val',
        classes=class_name,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='../data/tiny_ship/tiny_test.json',
        classes=class_name,
        img_prefix=data_root+'/test',
        pipeline=test_pipeline))
# optimizer
# AdamW
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='PolyLrUpdaterHook',
#     warmup='linear',
#     warmup_iters=2000,  # same as burn-in in darknet
#     warmup_ratio=0.1,
#     step=[218, 246])
lr_config = dict(
    policy='Poly', power=0.9, min_lr=0.00001, by_epoch=True,
    warmup='linear', warmup_iters=15, warmup_ratio=0.1, warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(interval=1, metric=['bbox'], mode='eval', areaRng=[0, 20, 200])
test = dict(interval=2, metric=['bbox'], mode='test', areaRng=[0, 20, 200])
