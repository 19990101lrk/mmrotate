_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']


work_dir = "E:/lrk/trail/logs/baseline/ext/oriented_rcnn_r50_fpn_fp16_1x_dota_le90_R3_flipNone_v1"

fp16 = dict(loss_scale='dynamic')
