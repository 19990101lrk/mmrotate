# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    #
    parser.add_argument('--img', default='E:/lrk/trail/datasets/DOTA-v1.5/divide_ship/test/images/P0896__800__1200___1200.png', help='Image file')
    # 基础网络
    parser.add_argument('--config', default='E:/lrk/trail/logs/baseline/oriented_rcnn_r50_fpn_fp16_1x_dota_le90_R3/oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py', help='Config file')
    parser.add_argument('--checkpoint', default='E:/lrk/trail/logs/baseline/oriented_rcnn_r50_fpn_fp16_1x_dota_le90_R3/latest.pth', help='Checkpoint file')

    # proposal method
    # parser.add_argument('--config', default='E:/lrk/trail/logs/modify_V1/oriented_rcnn/oriented_rcnn_s50_k1r2_adamW_myfpn_fp16_1x_dota_le90_R3_flipNone_v3/oriented_rcnn_r50_myfpn_fp16_1x_dota_le90.py', help='Config file')
    # parser.add_argument('--checkpoint', default='E:/lrk/trail/logs/modify_V1/oriented_rcnn/oriented_rcnn_s50_k1r2_adamW_myfpn_fp16_1x_dota_le90_R3_flipNone_v3/latest.pth', help='Checkpoint file')



    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
