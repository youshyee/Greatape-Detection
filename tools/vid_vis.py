import argparse
import os.path as osp
import shutil
import tempfile
import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
from mmdet.core import tensor2imgs, get_classes
import tqdm
# import stackprinter
# stackprinter.set_excepthook(style='darkbg2')
checkpoint = 'workdir/slurm_v_retinanet_r50_fpn_tc5/epoch_15.pth'
config='configs/vid/tc_ct_com/v_retinanet_r50_fpn_tc^5^_1x.py'

cfg = mmcv.Config.fromfile(config)
if 'video' in cfg:
    video_model=cfg.video
else:
    video_model=False
    # set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.data.val.test_mode = True

model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg,video=video_model)
load_checkpoint(model, checkpoint, map_location='cpu')

dataset = get_dataset(cfg.data.val)
data_loader = build_dataloader(dataset,
                                imgs_per_gpu=1,
                                workers_per_gpu=cfg.data.workers_per_gpu,
                                dist=True,
                                shuffle=False,
                                video=video_model)


model = model.cuda()
model = MMDataParallel(model, device_ids=[0])
model.eval()
#print(model)
results = []


def show_result(batch_num,
                data,
                result,
                img_norm_cfg,
                dataset=None,
                score_thr=0.3,
                outfile_dir='demo'):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        size=img_tensor.size(1)
        img_metas = data['img_meta'][0].data[0]*size
        imgs = tensor2imgs(img_tensor[0,...], **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        class_names = list(dataset.id2name.values())
        i=0

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            img_show = mmcv.imrescale(img_show,1/img_meta['scale_factor'])
            bboxes = np.vstack(bbox_result[i])
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result[i])
            ]
            labels = np.concatenate(labels)

            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                show=False,
                out_file=osp.join(outfile_dir,'{}_{}.jpeg'.format(batch_num,i)))
            i+=1


for i, data in tqdm.tqdm(enumerate(data_loader)):
    if i <5:
        print(i)
    else:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        show_result(i,data,result,cfg.img_norm_cfg,dataset)
    if i > 6000:
        break

