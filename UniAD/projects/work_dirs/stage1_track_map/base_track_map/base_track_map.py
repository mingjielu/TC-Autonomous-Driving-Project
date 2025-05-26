point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesE2EDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesInCeph',
        to_float32=True,
        file_client_args=dict(backend='disk'),
        img_root='data/nuscenes/'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D_E2E',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=True),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0]),
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=False),
    dict(
        type='ObjectRangeFilterTrack',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilterTrack',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds', 'img', 'timestamp',
            'l2g_r_mat', 'l2g_t', 'gt_fut_traj', 'gt_fut_traj_mask',
            'gt_past_traj', 'gt_past_traj_mask', 'gt_sdc_bbox', 'gt_sdc_label',
            'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask', 'gt_lane_labels',
            'gt_lane_bboxes', 'gt_lane_masks', 'gt_segmentation',
            'gt_instance', 'gt_centerness', 'gt_offset', 'gt_flow',
            'gt_backward_flow', 'gt_occ_has_invalid_frame',
            'gt_occ_img_is_valid', 'gt_future_boxes', 'gt_future_labels',
            'sdc_planning', 'sdc_planning_mask', 'command'
        ])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesInCeph',
        to_float32=True,
        file_client_args=dict(backend='disk'),
        img_root='data/nuscenes/'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='LoadAnnotations3D_E2E',
        with_bbox_3d=False,
        with_label_3d=False,
        with_attr_label=False,
        with_future_anns=True,
        with_ins_inds_3d=False,
        ins_inds_add_1=True),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0]),
        ignore_index=255,
        only_vehicle=True,
        filter_invisible=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=[
                    'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'gt_lane_labels',
                    'gt_lane_bboxes', 'gt_lane_masks', 'gt_segmentation',
                    'gt_instance', 'gt_centerness', 'gt_offset', 'gt_flow',
                    'gt_backward_flow', 'gt_occ_has_invalid_frame',
                    'gt_occ_img_is_valid', 'sdc_planning', 'sdc_planning_mask',
                    'command'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='NuScenesE2EDataset',
        data_root='data/nuscenes/',
        ann_file='data/infos/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFilesInCeph',
                to_float32=True,
                file_client_args=dict(backend='disk'),
                img_root='data/nuscenes/'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='LoadAnnotations3D_E2E',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False,
                with_future_anns=True,
                with_ins_inds_3d=True,
                ins_inds_add_1=True),
            dict(
                type='GenerateOccFlowLabels',
                grid_conf=dict(
                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0]),
                ignore_index=255,
                only_vehicle=True,
                filter_invisible=False),
            dict(
                type='ObjectRangeFilterTrack',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilterTrack',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds', 'img',
                    'timestamp', 'l2g_r_mat', 'l2g_t', 'gt_fut_traj',
                    'gt_fut_traj_mask', 'gt_past_traj', 'gt_past_traj_mask',
                    'gt_sdc_bbox', 'gt_sdc_label', 'gt_sdc_fut_traj',
                    'gt_sdc_fut_traj_mask', 'gt_lane_labels', 'gt_lane_bboxes',
                    'gt_lane_masks', 'gt_segmentation', 'gt_instance',
                    'gt_centerness', 'gt_offset', 'gt_flow',
                    'gt_backward_flow', 'gt_occ_has_invalid_frame',
                    'gt_occ_img_is_valid', 'gt_future_boxes',
                    'gt_future_labels', 'sdc_planning', 'sdc_planning_mask',
                    'command'
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk'),
        use_valid_flag=True,
        patch_size=[102.4, 102.4],
        canvas_size=(200, 200),
        bev_size=(200, 200),
        queue_length=5,
        predict_steps=12,
        past_steps=4,
        fut_steps=4,
        use_nonlinear_optimizer=True,
        occ_receptive_field=3,
        occ_n_future=6,
        occ_filter_invalid_sample=False),
    val=dict(
        type='NuScenesE2EDataset',
        data_root='data/nuscenes/',
        ann_file='data/infos/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFilesInCeph',
                to_float32=True,
                file_client_args=dict(backend='disk'),
                img_root='data/nuscenes/'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='LoadAnnotations3D_E2E',
                with_bbox_3d=False,
                with_label_3d=False,
                with_attr_label=False,
                with_future_anns=True,
                with_ins_inds_3d=False,
                ins_inds_add_1=True),
            dict(
                type='GenerateOccFlowLabels',
                grid_conf=dict(
                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0]),
                ignore_index=255,
                only_vehicle=True,
                filter_invisible=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                            'gt_lane_labels', 'gt_lane_bboxes',
                            'gt_lane_masks', 'gt_segmentation', 'gt_instance',
                            'gt_centerness', 'gt_offset', 'gt_flow',
                            'gt_backward_flow', 'gt_occ_has_invalid_frame',
                            'gt_occ_img_is_valid', 'sdc_planning',
                            'sdc_planning_mask', 'command'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk'),
        patch_size=[102.4, 102.4],
        canvas_size=(200, 200),
        bev_size=(200, 200),
        predict_steps=12,
        past_steps=4,
        fut_steps=4,
        use_nonlinear_optimizer=True,
        samples_per_gpu=1,
        eval_mod=['det', 'track', 'map'],
        occ_receptive_field=3,
        occ_n_future=6,
        occ_filter_invalid_sample=False),
    test=dict(
        type='NuScenesE2EDataset',
        data_root='data/nuscenes/',
        ann_file='data/infos/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFilesInCeph',
                to_float32=True,
                file_client_args=dict(backend='disk'),
                img_root='data/nuscenes/'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='LoadAnnotations3D_E2E',
                with_bbox_3d=False,
                with_label_3d=False,
                with_attr_label=False,
                with_future_anns=True,
                with_ins_inds_3d=False,
                ins_inds_add_1=True),
            dict(
                type='GenerateOccFlowLabels',
                grid_conf=dict(
                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0]),
                ignore_index=255,
                only_vehicle=True,
                filter_invisible=False),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                            'gt_lane_labels', 'gt_lane_bboxes',
                            'gt_lane_masks', 'gt_segmentation', 'gt_instance',
                            'gt_centerness', 'gt_offset', 'gt_flow',
                            'gt_backward_flow', 'gt_occ_has_invalid_frame',
                            'gt_occ_img_is_valid', 'sdc_planning',
                            'sdc_planning_mask', 'command'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk'),
        patch_size=[102.4, 102.4],
        canvas_size=(200, 200),
        bev_size=(200, 200),
        predict_steps=12,
        past_steps=4,
        fut_steps=4,
        occ_n_future=6,
        use_nonlinear_optimizer=True,
        eval_mod=['det', 'map', 'track']),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=6,
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFilesInCeph',
            to_float32=True,
            file_client_args=dict(backend='disk'),
            img_root='data/nuscenes/'),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='LoadAnnotations3D_E2E',
            with_bbox_3d=False,
            with_label_3d=False,
            with_attr_label=False,
            with_future_anns=True,
            with_ins_inds_3d=False,
            ins_inds_add_1=True),
        dict(
            type='GenerateOccFlowLabels',
            grid_conf=dict(
                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0]),
            ignore_index=255,
            only_vehicle=True,
            filter_invisible=False),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='CustomCollect3D',
                    keys=[
                        'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
                        'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks',
                        'gt_segmentation', 'gt_instance', 'gt_centerness',
                        'gt_offset', 'gt_flow', 'gt_backward_flow',
                        'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid',
                        'sdc_planning', 'sdc_planning_mask', 'command'
                    ])
            ])
    ],
    planning_evaluation_strategy='uniad')
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './projects/work_dirs/stage1_track_map/base_track_map/'
load_from = 'ckpts/bevformer_r101_dcn_24ep.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = 512
_dim_half_ = 128
canvas_size = (200, 200)
queue_length = 5
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = True
occ_n_future = 4
occ_n_future_plan = 6
occ_n_future_max = 6
planning_steps = 6
use_col_optim = True
planning_evaluation_strategy = 'uniad'
occflow_grid_conf = dict(
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0])
train_gt_iou_threshold = 0.3
model = dict(
    type='UniAD',
    gt_iou_threshold=0.3,
    queue_length=5,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=10,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    freeze_img_backbone=True,
    freeze_img_neck=False,
    freeze_bn=False,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1),
    mem_args=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4),
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=10,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_past_traj_weight=0.0),
    pts_bbox_head=dict(
        type='BEVFormerTrackHead',
        bev_h=200,
        bev_w=200,
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=4,
        fut_steps=4,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=4),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=200,
            col_num_embed=200),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    seg_head=dict(
        type='PansegformerHead',
        bev_h=200,
        bev_w=200,
        canvas_size=(200, 200),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        num_query=300,
        num_classes=4,
        num_things_classes=3,
        num_stuff_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        transformer=dict(
            type='SegDeformableTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=4),
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=4)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0),
        thing_transformer_head=dict(
            type='SegMaskHead', d_model=256, nhead=8, num_decoder_layers=4),
        stuff_transformer_head=dict(
            type='SegMaskHead',
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            self_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
            assigner_with_mask=dict(
                type='HungarianAssigner_multi_info',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(type='DiceCost', weight=2.0)),
            sampler=dict(type='PseudoSampler'),
            sampler_with_mask=dict(type='PseudoSampler_segformer'))),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
info_root = 'data/infos/'
ann_file_train = 'data/infos/nuscenes_infos_temporal_train.pkl'
ann_file_val = 'data/infos/nuscenes_infos_temporal_val.pkl'
ann_file_test = 'data/infos/nuscenes_infos_temporal_val.pkl'
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=6)
find_unused_parameters = True
gpu_ids = range(0, 4)
