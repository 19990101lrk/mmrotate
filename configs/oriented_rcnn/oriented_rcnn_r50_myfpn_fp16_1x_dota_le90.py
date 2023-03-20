_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']

work_dir = "E:/lrk/trail/logs/modify_V1/oriented_rcnn/oriented_rcnn_s50_fpn_fp16_1x_dota_le90_R3_flipNone_v2"

fp16 = dict(loss_scale='dynamic')

# model = dict(
#     neck=dict(
#         type='MyFPN',
#         in_channels=[256, 512, 1024, 2048]),
#
#     roi_head=dict(
#         bbox_head=dict(
#             reg_decoded_bbox=True,
#             loss_bbox=dict(_delete_=True, type='RotatedAngleIoULoss', loss_weight=1.0, balance_factor=1)
#         )
#     )
# )

# ----------------------  主干模块添加 plugins ---------------------------------#
# model = dict(
#     type='OrientedRCNN',
#     backbone=dict(
#         plugins=[
#             dict(
#                 cfg=dict(
#                     type='GeneralizedAttention',
#                     spatial_range=-1,
#                     num_heads=8,
#                     attention_type='0010',
#                     kv_stride=2),
#                 stages=(False, False, True, True),
#                 position='after_conv2')
#         ],
#         dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
#         stage_with_dcn=(False, True, True, True)
#     ),
#     neck=dict(
#         type='MyFPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5),
# )

# ------------------------------- 主干修改为ResNeSt -------------------------------------------- #
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ResNeSt',
        stem_channels=64,
        depth=50,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnest50')),
    # neck=dict(
    #     type='MyFPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     num_outs=5),
    # roi_head=dict(
    #     bbox_head=dict(
    #         reg_decoded_bbox=True,
    #         loss_bbox=dict(
    #             _delete_=True,
    #             type='GDLoss',
    #             loss_type='kld',
    #             fun='log1p',
    #             tau=1.0,
    #             sqrt=False,
    #             loss_weight=8.0)
    #     )
    # )
)

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }
#     )
# )

# roi_head=dict(
#     bbox_head=dict(
#         reg_decoded_bbox=True,
#         # loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0)
#         loss_bbox=dict(
#             type='GDLoss_v1',
#             loss_type='kld',
#             fun='log1p',
#             tau=1,
#             loss_weight=1.0)
#     )
# )
# )


# resume_from= 'E:/lrk/trail/logs/modify_V1/oriented_rcnn/oriented_rcnn_r50_myfpn_fp16_1x_dota_le90_R3_flipNone_2/latest.pth'
