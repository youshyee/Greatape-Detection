from __future__ import division

import argparse
from mmcv import Config

from mmdet import __version__
import torch.distributed as dist
from mmdet.datasets import get_dataset, build_dataloader
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import torch
from mmcv.runner import load_checkpoint, get_dist_info
from get_eval_pth import get_eval_pth, final_evaluate
from mmdet.core import vid_result2txt
from test import multi_gpu_test, single_gpu_test
import os.path as osp
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--datapath', help='training split file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        type=bool,
        default=False)

    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    cfg.data.train.ann_file = args.datapath
    cfg.data.val.ann_file = args.datapath
    cfg.data.train.type = 'CHIMP_TRAIN'
    cfg.data.val.type = 'CHIMP_TRAIN'
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if 'video' in cfg:
        video_model = cfg.video
    else:
        video_model = False

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg, video=video_model)

    train_dataset = get_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text,
            classes=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
        video=video_model)

    # add final test here
    del model
    del train_dataset
    if args.test == 1:
        cfg.data.val.test_mode = True
        cfg.data.val.min_val = False
        # cfg.data.val.debug=False
        # init distributed env first, since logger depends on the dist info.
        # build the dataloader
        dataset = get_dataset(cfg.data.val)
        data_loader = build_dataloader(dataset,
                                       imgs_per_gpu=1,
                                       workers_per_gpu=cfg.data.workers_per_gpu,
                                       dist=distributed,
                                       shuffle=False,
                                       video=video_model)

        # build the model and load checkpoint
        model = build_detector(
            cfg.model,
            train_cfg=None,
            test_cfg=cfg.test_cfg,
            video=video_model)

        # load 2 or 1 checkpoint here
        checkpoints = get_eval_pth(cfg.work_dir)
        for checkpoint in checkpoints:
            load_checkpoint(
                model,
                osp.join(
                    cfg.work_dir,
                    checkpoint),
                map_location='cpu')
            print('evaluating ', checkpoint)
            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader, args.show)
            else:
                model = MMDistributedDataParallel(model.cuda())
                outputs = multi_gpu_test(
                    model, data_loader, osp.join(
                        cfg.work_dir, 'temp__'))

            rank, _ = get_dist_info()
            dist.barrier()
            if rank == 0:
                ep = checkpoint.split('.')[0].split('_')[-1]
                raw_result = osp.join(
                    cfg.work_dir, 'raw_result_final_ep{}.txt'.format(ep))
                vid_result2txt(outputs, raw_result)
                # save the result
                outputs = [result['result'] for result in outputs]
                final_evaluate(
                    outputs,
                    dataset,
                    None,
                    cfg.work_dir,
                    refresh=True)
            dist.barrier()
        # last thing is to copy config file:args.config into workdir:
        # cfg.work_dir
        if rank == 0:
            config_name = args.config.split('/')[-1]
            shutil.copy2(args.config, osp.join(cfg.work_dir, config_name))


if __name__ == '__main__':
    main()
