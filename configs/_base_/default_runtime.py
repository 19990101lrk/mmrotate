# yapf:disable
log_config = dict(
    interval=50,        # 50次迭代打印一次日志
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间
load_from = None

# resume_from从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
resume_from = None

workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
