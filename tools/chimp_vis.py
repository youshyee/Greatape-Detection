import argparse
import os
import os.path as osp
import shutil
import tempfile

import cv2
import mmcv
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, get_classes, results2json, tensor2imgs
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
from termcolor import colored

from generate_frame_xml import create_frame_xml


# import stackprinter
# stackprinter.set_excepthook(style='darkbg2')
def parse_args():
    parser = argparse.ArgumentParser(
        description='detection on great ape video footage ')
    parser.add_argument(
        '--input', help='the input dir of videos or single video path')
    parser.add_argument(
        '--config', help='the model config file path (default: Retinanet based res50-tcm-scm model)')
    parser.add_argument(
        '--checkpoint', help='checkpoint of the model (default:Retinanet based res50-tcm-scm model)')
    parser.add_argument('--output_dir', default='./output',
                        help='the output video dir')

    parser.add_argument('--tmpdir', default='../.tmps',
                        help='tmp dir for writing some results')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.tmpdir)
    mmcv.mkdir_or_exist(args.output_dir)
    # = './workdir/chimp/retinanet_r50/tc_mode2_train7_test21/epoch_7.pth'
    checkpoint = args.checkpoint
    config = args.config  # 'configs/chimp/chimptest.py'
 

    ####################
    
    # videos = videos2
    videos=mmcv.list_from_file('/home/richard/space/dataspace/PanAfrica/2.txt')
    print(videos)
    videos = [osp.join(args.input, v+'.mp4')for v in videos]
    ####################
    print(colored('Inferring with TCM+SCM ', 'green'))

    cfg = mmcv.Config.fromfile(config)
    if 'video' in cfg:
        video_model = cfg.video
    else:
        video_model = False
        # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.val.test_mode = True
    cfg.data.val.type = 'CHIMP_INFER'
    cfg.data.val.how_sparse = 1
    init_dist(args.launcher, **cfg.dist_params)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg,
                           test_cfg=cfg.test_cfg, video=video_model)
    load_checkpoint(model, checkpoint, map_location='cpu')

    # load the dataset
    datasets = []
    data_loaders = []
    precess_videos = []
    video_metas = []
    print(colored('Processing', 'red'))
    for video in tqdm.tqdm(videos):
        # extract frames for each video and save it in tmp_dir
        video_frames = mmcv.VideoReader(video)
        video_metas.append({'height': video_frames.height,
                            'width': video_frames.width, 'fps': video_frames.fps})
        frame_path = os.path.join(
            args.tmpdir, 'original/', os.path.basename(video))
        mmcv.mkdir_or_exist(frame_path)
        # video_frames.cvt2frames(frame_path)
        cfg.data.val.ann_file = frame_path
        dataset = get_dataset(cfg.data.val)
        data_loader = build_dataloader(dataset,
                                       imgs_per_gpu=1,
                                       workers_per_gpu=cfg.data.workers_per_gpu,
                                       dist=True,
                                       shuffle=False,
                                       video=video_model)
        datasets.append(dataset)
        data_loaders.append(data_loader)
        precess_videos.append(os.path.basename(video))

    model = model.cuda()
    #model = MMDataParallel(model, device_ids=[0])
    model = MMDistributedDataParallel(model.cuda())
    model.eval()

    # begin to infer
    for m in range(len(precess_videos)):
        data_loader = data_loaders[m]
        video_meta = video_metas[m]
        precess_video = precess_videos[m]
        video_basename = precess_video.split('.')[0]
        dataset = datasets[m]
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            show_result(i, data, result, cfg.img_norm_cfg, dataset, outfile_dir=os.path.join(
                args.tmpdir, 'processed/', video_basename), videoname=video_basename, species='chimpanzee')
        # after process each frame, convert them to video stream
        dist.barrier()
        rank, _ = get_dist_info()
        # if rank == 0:
        #     frame2videos(os.path.join(args.tmpdir, 'processed/', video_basename),
        #                  osp.join(args.output_dir, video_basename+'.mp4'), video_meta)
        # dist.barrier()


def show_result(batch_num,
                data,
                result,
                img_norm_cfg,
                dataset=None,
                score_thr=0.6,
                outfile_dir='demo',
                videoname='dumpvideo',
                species='gorilla'
                ):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    img_tensor = data['img'][0]
    size = img_tensor.size(1)
    img_metas = data['img_meta'][0].data[0]*size
    imgs = tensor2imgs(img_tensor[0, ...], **img_norm_cfg)
    rank, _ = get_dist_info()
    if dataset.repeat_mode:
        snip = dataset.snip_frames
        imgs = imgs[snip//2:snip//2+1]
        img_metas = img_metas[snip//2:snip//2+1]
    assert len(imgs) == len(img_metas)
    class_names = ['chimp']
    i = 0
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        img_show = mmcv.imrescale(img_show, 1/img_meta['scale_factor'])
        bboxes = np.vstack(bbox_result[i])
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result[i])
        ]
        labels = np.concatenate(labels)
        #################
        # log the bboxes over than 0.7 together with frame number
        selected_boxes = bboxes[bboxes[:, 4] > 0.7]
        create_frame_xml(videoname, img_meta, selected_boxes,
                         species, './generated_annotationss')
        #################
        # mmcv.imshow_det_bboxes(
        #     img_show,
        #     bboxes,
        #     labels,
        #     thickness=2,
        #     class_names=class_names,
        #     score_thr=score_thr,
        #     show=False,
        #     out_file=osp.join(outfile_dir, '{}_{}_{}.jpeg'.format(batch_num, rank, i)))
        # i += 1


def frame2videos(frame_dir, output_path, video_meta):
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_meta['fps']
    frames = os.listdir(frame_dir)
    frames = sorted(frames, key=lambda x: int(
        x.split('_')[0])*2+int(x.split('_')[1]))
    h = video_meta['height']
    w = video_meta['width']
    # build writer
    writer = cv2.VideoWriter(output_path, video_FourCC, fps, (w, h))

    for i in tqdm.tqdm(range(len(frames))):
        frame = frames[i]
        img = cv2.imread(os.path.join(frame_dir, frame))
        writer.write(img)
    writer.release()


if __name__ == "__main__":
    main()
