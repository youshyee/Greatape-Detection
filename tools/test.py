from __future__ import division

import argparse
from mmcv import Config
import mmcv
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
import os.path as osp
import shutil


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            ids = data['img_meta'][0].data[0][0]['frame_ids']
            result = dict(result=result, ids=ids)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(bytearray(tmpdir.encode()),
                                  dtype=torch.uint8,
                                  device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        'checkpoint',
        help='checkpoint of corresponding config')
    parser.add_argument('datapath', help='training split file path')

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
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
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

    checkpoint = args.checkpoint

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg, video=video_model)
    load_checkpoint(model, checkpoint, map_location='cpu')

    # add final test here

    cfg.data.val.ann_file = args.datapath
    cfg.data.val.type = 'CHIMP_TRAIN'
    cfg.data.val.test_mode = True
    cfg.data.val.min_val = False
    cfg.data.val.improved_annotation = True
    dataset = get_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset,
                                   imgs_per_gpu=1,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False,
                                   video=video_model)

    print('evaluating ', checkpoint)
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show)
    else:
        rank, _ = get_dist_info()
        model = MMDistributedDataParallel(model.cuda(rank), device_ids=[rank])
        outputs = multi_gpu_test(
            model, data_loader, osp.join(
                cfg.work_dir, '../.temp'))

        dist.barrier()
        if rank == 0:
            outputs = [result['result'] for result in outputs]
            final_evaluate(
                outputs,
                dataset,
                None,
                cfg.work_dir,
                refresh=True)
        dist.barrier()


if __name__ == '__main__':
    main()
