_base_ = ['./redet_re50_refpn_1x_dota_le90.py']


work_dir = "E:/lrk/trail/logs/baseline/redet_re50_refpn_fp16_1x_dota_le90_R3"

fp16 = dict(loss_scale='dynamic')
