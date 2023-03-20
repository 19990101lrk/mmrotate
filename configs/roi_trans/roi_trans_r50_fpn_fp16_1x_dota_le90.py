_base_ = ['./roi_trans_r50_fpn_1x_dota_le90.py']

work_dir = "E:/lrk/trail/logs/baseline/roi_trans_r50_fpn_fp16_1x_dota_le90_R3"

fp16 = dict(loss_scale='dynamic')
