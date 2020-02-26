'''
brief description:
template for chimp
retinanet r50

'''
debug_mode = False
# model settings
# key settings
video = True
snip_size = 7
test_snip_size = 21
use_tcs = False  # if use tcs test snip size must == snip size
tcmode = 'mode2'  # mode 1 sliding window || mode2 snip_size=2n+1 if reduce must be this mode || mode3 any fit
use_train_multiscale = True
use_train_img_aug = True
dataset_repeat_mode_test = True
dataset_repeat_mode_train = False
work_dir = './workdir/chimp/retinanet_r50/tc_mode2_train7_test21'
load_from = './workdir/slurm_v_retinanet_r50_allon_repeat/epoch_13.pth'

#####################################
if True:  # fold this static setting
    input_scale_train = [
        (800, 500), (1088, 608)] if use_train_multiscale else (
        800, 500)
    aug = dict(
        type='Compose',
        #bbox_params={'format': 'pascal_voc', 'min_area': 2, 'min_visibility': 0.5, 'label_fields': ['category_id']},
        transforms=[
            # dict(
            #     height=404,
            #     width=720,
            #     type='RandomSizedBBoxSafeCrop',
            #     p=0
            # ),
            dict(
                p=0,
                max_h_size=32,
                max_w_size=32,
                type='Cutout'
            ),
            dict(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0,
                type='RandomBrightnessContrast'
            ),
            dict(
                p=0,
                quality_lower=80,
                quality_upper=99,
                type='JpegCompression'
            ),
        ],
        p=1.0
    )
    aug_prob = [0.5, 0.5, 0.5]
    aug_p = 0.5
    if not use_train_img_aug:
        aug = None
    multi_mode = 'range' if use_train_multiscale else 'value'
    input_scale_test = (1088, 608)
    tc_mode = {
        'mode1':  # sliding window
        dict(
            repeat_mode=False,
            is_position_encoding=True,
        ),
        'mode2':  # just distinguish references frames and target frames input frame should be 2n+1
        dict(
            repeat_mode=True,
            is_position_encoding=False,
            window_size=None,
        ),
        'mode3':  # no share weight at all
        dict(
            repeat_mode=False,
            is_position_encoding=False,
            window_size=None,
        )
    }
    repeat_mode = tc_mode[tcmode]['repeat_mode']
    is_position_encoding = tc_mode[tcmode]['is_position_encoding']
    test_snip_size = snip_size if use_tcs else test_snip_size

model = dict(
    type='RetinaNet',
    pretrained='modelzoo://resnet50',
    test_reduce=dataset_repeat_mode_test,
    train_reduce=dataset_repeat_mode_train,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        video=video,
        test_reduce=dataset_repeat_mode_test,
        train_reduce=dataset_repeat_mode_train,
        ############# tc ##################
        tc=dict(
            insert_pos='after_add',
            repeat_mode=repeat_mode,
            is_position_encoding=is_position_encoding,
            window_size=5,
            detach=False,
            local_mean=True,
            reduce=False
        ),
        stage_with_tc=(False, False, False, True),

        ############# tcs ##################
        tcs=dict(
            insert_pos='after_add',
            snip_size=snip_size,
        ),
        stage_with_tcs=(
            False,
            False,
            False,
            False) if use_tcs else (
            False,
            False,
            False,
            False),
        ############# tcsm ##################
        tcsm=dict(
            insert_pos='after_add',
        ),
        stage_with_tcsm=(True, True, True, True),
        ############# ct ###################
        ct=dict(
            insert_pos='after_add',
            ratio=1. / 4.,
        ),
        stage_with_ct=(False, True, True, True),
        ############# cp #####################
        cp=dict(
            topk=5,
            mode='max'
        ),
        stage_with_cp=(False, False, False, False),

        ############# dcn ##################
        dcn=dict(
            modulated=False,
            groups=32,
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),

        ############# norm ##################
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0])
)

# training and testing settings

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)

# dataset settings
dataset_type = 'CHIMP_TRAIN'
data_root = '/mnt/storage/scratch/rn18510/chimp_annotation/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root,
        img_prefix=data_root + 'train2017/',
        img_scale=input_scale_train,
        multiscale_mode=multi_mode,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        snip_frame=snip_size,
        how_sparse=1,
        debug=debug_mode,
        repeat_mode=dataset_repeat_mode_train,
        extra_aug=aug,
        aug_prob=aug_prob,
        aug_p=aug_p
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root,
        img_prefix=data_root + 'val2017/',
        img_scale=input_scale_test,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        snip_frame=test_snip_size,
        debug=debug_mode,
        repeat_mode=dataset_repeat_mode_test,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Data/VID/test',
        img_prefix=data_root + 'val2017/',
        img_scale=(1088, 608),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    # gamma=0.02,
    warmup_iters=500,
    warmup_ratio=1.0 / 20,
    step=[8, 12]
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 14
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
