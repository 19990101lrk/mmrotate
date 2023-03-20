_base_ = [
    './rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
]

work_dir = "E:/lrk/trail/logs/modify_V1/rotated_faster_rcnn/rotated_faster_rcnn_r50_myfpn_fp16_1x_dotaship_le90_V2_1"

fp16 = dict(loss_scale='dynamic')

model = dict(
    neck=dict(
        type='MyFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
)

# resume_from = 'E:/lrk/trail/logs/modify_V1/twoDense/rotated_faster_rcnn_r50_myfpn_fp16_1x_dotaship_le90_V2/latest.pth'
