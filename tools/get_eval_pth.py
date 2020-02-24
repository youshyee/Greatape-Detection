'''
fininal evaluation file
'''
import os
from mmdet.core import eval_map
import torch
from yxy import path


def get_eval_pth(workdir):
    def get_result_map(x): return int(x.split('.')[0].split('_')[-1])
    def get_result_ep(x): return int(
        x.split('.')[0].split('_')[0].replace('ep', ''))

    def get_pth_ep(x): return int(x.split('.')[0].split('_')[-1])

    all_result_file = [i for i in os.listdir(workdir) if '.result' in i]
    all_pth_file = [i for i in os.listdir(
        workdir) if '.pth' in i and 'latest' not in i]
    to_exe_pth = []

    best_result = sorted(all_result_file, key=get_result_map)[-1]
    latest_result = sorted(all_result_file, key=get_result_ep)[-1]

    latest_pth = sorted(all_pth_file, key=get_pth_ep)[-1]
    best_pth = 'epoch_{}.pth'.format(get_result_ep(best_result))
    assert get_pth_ep(latest_pth) == get_result_ep(latest_result)

    best_map = get_result_map(best_result)
    latest_map = get_result_map(latest_result)

    assert best_map >= latest_map

    to_eval = []
    if best_map - latest_map < 50:
        to_eval.append(latest_pth)
    else:
        to_eval += [latest_pth, best_pth]
    return to_eval


def final_evaluate(results, dataset, ep, work_dir, refresh=True):
    gt_bboxes = []
    gt_labels = []
    gt_ignore = [] if dataset.with_crowd else None
    if hasattr(dataset, 'snip_frames'):
        print('vid dataset eval')
        outs = []
        for result in results:
            outs += result
        results = outs
        del outs
        is_cache = os.path.isfile(
            f'{path.code}/.cache/cache_{type(dataset)}_gt.dict')
        if not refresh and is_cache:
            print('load cache')
            gt = torch.load(
                f'{path.code}/.cache/cache_{type(dataset)}_gt.dict')
            gt_bboxes = gt['gt_bboxes']
            gt_labels = gt['gt_labels']
        else:
            for i in range(len(dataset)):
                anns, _ = dataset.get_ann_info(i)
                bboxes = [ann['bboxes'] for ann in anns]
                labels = [ann['labels'] for ann in anns]
                if dataset.repeat_mode:
                    bboxes = bboxes[dataset.snip_frames //
                                    2:dataset.snip_frames // 2 + 1]
                    labels = labels[dataset.snip_frames //
                                    2:dataset.snip_frames // 2 + 1]
                gt_bboxes += bboxes
                gt_labels += labels
            gt = {'gt_labels': gt_labels, 'gt_bboxes': gt_bboxes}
            torch.save(
                gt, f'{path.code}/.cache/cache_{type(dataset)}_gt.dict')
    # If the dataset is VOC2007, then use 11 points mAP evaluation.
    if hasattr(dataset, 'year') and dataset.year == 2007:
        ds_name = 'voc07'
    elif hasattr(dataset, 'snip_frames'):
        ds_name = 'vid'
    else:
        ds_name = dataset.CLASSES
    mean_ap, eval_results = eval_map(
        results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=ds_name,
        print_summary=True)
    print(mean_ap)
