_base_ = [
    './rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
]

work_dir = "E:/lrk/trail/logs/baseline/rotated_faster_rcnn_r50_fpn_fp16_1x_dota_le90_R3_flip"

fp16 = dict(loss_scale='dynamic')


# resume_from = 'E:/lrk/trail/logs/baseline/rotated_faster_rcnn_r50_fpn_fp16_1x_dotaship_le90_R8_flipNone/latest.pth'