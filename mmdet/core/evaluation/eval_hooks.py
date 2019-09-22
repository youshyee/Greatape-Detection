import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset
from ..utils import vid_result2txt
from .coco_utils import results2json, fast_eval_recall
from .mean_ap import eval_map
#from mmdet.datasets import build_dataloader
from mmdet import datasets

from mmcv.runner import load_checkpoint, get_dist_info
import tempfile
import shutil

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
class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch_(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if hasattr(self.dataset,'snip_frames'):
            collate=self.dataset.collate_fn
        else:
            from mmcv.parallel import collate as collate
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            #data_gpu['img'][0]=data_gpu['img'][0].squeeze(0)
            #print('gpu data',data_gpu['img'][0].shape,type(data_gpu['img'][0]))
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result
            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def after_train_epoch(self,runner): # fast version of eval
        if not self.every_n_epochs(runner, self.interval):
            return
        if hasattr(self.dataset,'snip_frames'):
            video=True
            img_per_gpu=1
        else:
            video=False
            img_per_gpu=1
        data_loader = datasets.build_dataloader(self.dataset,
                                   imgs_per_gpu=img_per_gpu,
                                   workers_per_gpu=8,
                                   dist=True,
                                   shuffle=False,
                                   video=video)
        runner.model.eval()
        results = []
        rank=runner.rank
        world_size = runner.world_size
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = runner.model(return_loss=False, rescale=True, **data)
                ## added part
                ids=data['img_meta'][0].data[0][0]['frame_ids']
                result = dict(result=result,ids=ids)
                ##
            results.append(result)

            if rank == 0:
                batch_size = data['img'][0].size(0)
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        dist.barrier()
        results = collect_results(results, len(self.dataset), os.path.join(runner.work_dir,'temp/'))
        if runner.rank == 0:
            # ep=runner.epoch+1
            # raw_result=os.path.join(runner.work_dir,'raw_result_ep{}.txt'.format(ep))
            # vid_result2txt(results,raw_result)
            ##########
            results=[result['result'] for result in results]
            self.evaluate(runner, results)
    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        if hasattr(self.dataset, 'txtfiles'):
            gt_ignore = None
        if hasattr(self.dataset, 'snip_frames'):
            print('vid dataset eval')
            outs=[]
            for result in results:
                outs+=result
            results=outs
            del outs
            for i in range(len(self.dataset)):
                anns,_ = self.dataset.get_ann_info(i)
                bboxes = [ann['bboxes'] for ann in anns]
                labels = [ann['labels'] for ann in anns]
                if self.dataset.repeat_mode:
                    bboxes = bboxes[self.dataset.snip_frames//2:self.dataset.snip_frames//2+1]
                    labels = labels[self.dataset.snip_frames//2:self.dataset.snip_frames//2+1]
                gt_bboxes+=bboxes
                gt_labels+=labels
        else:
            for i in range(len(self.dataset)):
                ann = self.dataset.get_ann_info(i)
                bboxes = ann['bboxes']
                labels = ann['labels']
                if gt_ignore is not None:
                    ignore = np.concatenate([
                        np.zeros(bboxes.shape[0], dtype=np.bool),
                        np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                    ])
                    gt_ignore.append(ignore)
                    bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                    labels = np.concatenate([labels, ann['labels_ignore']])
                gt_bboxes.append(bboxes)
                gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        elif self.dataset.__class__.__name__=='VID':
            ds_name='vid'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        ep=runner.epoch+1
        save_name=os.path.join(runner.work_dir,'ep{}_map_{}.result'.format(ep,int(round(mean_ap*10000))))
        torch.save(eval_results,save_name)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        os.remove(tmp_file)
