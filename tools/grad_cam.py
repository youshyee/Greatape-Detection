import argparse
import os.path as osp
import os
import shutil
import tempfile
import numpy as np
import mmcv
import torch
import copy
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
from mmdet.core import tensor2imgs, get_classes

from PIL import Image
import matplotlib.cm as mpl_color_map
import tqdm
import cv2
# import stackprinter
# stackprinter.set_excepthook(style='darkbg2')
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, img,img_meta, return_loss=False, rescale=True, ):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        img=img[0]
        img=img.cuda()
        img.requires_grad=True
        img.register_hook(self.save_gradient)
        img_meta=img_meta
        x=self.model(return_loss=False, rescale=True,img=[img],img_meta=img_meta)
        return  x
def apply_colormap_on_image(org_im, activation_org, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    activation=np.uint8(activation_org * 255)
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = activation_org
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap,heatmap, heatmap_on_image

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work_dir', help='word_dir store epoch')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--std', type=float, default=0.001)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    checkpoint = './workdir/chimp/retinanet_r50/tc_mode2_train7_test21/epoch_7.pth'
    config='configs/chimp/chimptest.py'
    videos_foloer = '/mnt/storage/scratch/rn18510/infer_videos/'
    videos=os.listdir(videos_foloer)
    videos=['2AwqiWB5Ud.mp4','2miYEJ2Ofx.mp4']
    print('tcm evaling')
    inputstd=args.std
    print(inputstd)
    cfg = mmcv.Config.fromfile(config)
    if 'video' in cfg:
        video_model=cfg.video
    else:
        video_model=False
        # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.val.test_mode = True
    cfg.data.val.type='CHIMP_INFER'
    cfg.data.val.snip_frame=3
    cfg.data.val.how_sparse=1

    init_dist(args.launcher, **cfg.dist_params)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg,video=video_model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    extractor =CamExtractor(model)

    datasets=[]
    data_loaders=[]
    precess_videos=[]
    for video in videos:
        cfg.data.val.ann_file = os.path.join(videos_foloer,video)
        dataset = get_dataset(cfg.data.val)
        data_loader = build_dataloader(dataset,
                                        imgs_per_gpu=1,
                                        workers_per_gpu=cfg.data.workers_per_gpu,
                                        dist=True,
                                        shuffle=False,
                                        video=video_model)
        datasets.append(dataset)
        data_loaders.append(data_loader)
        precess_videos.append(video)

    model = model.cuda()
    #model = MMDataParallel(model, device_ids=[0])
    model = MMDistributedDataParallel(model.cuda())
    model.eval()

    for m in range(len(precess_videos)):
        data_loader=data_loaders[m]
        precess_video=precess_videos[m]
        mmcv.mkdir_or_exist(f'./results/example/cam_{inputstd}/{precess_video}/heatmap/')
        mmcv.mkdir_or_exist(f'./results/example/cam_{inputstd}/{precess_video}/no_tran_heatmap/')
        mmcv.mkdir_or_exist(f'./results/example/cam_{inputstd}/{precess_video}/heatmap_on_image/')
        dataset=datasets[m]
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            outs=extractor.forward_pass(return_loss=False, rescale=True, **data)
            #outs = model(return_loss=False, rescale=True, **data)
            classification = outs[0]
            b,a,h,w=outs[0][0].shape
            classification=[out.view(b,a,-1) for out in classification]
            classification = torch.cat(classification,dim=2)
            classification=classification.sigmoid()

            mask=classification>0.3
            one_hot_output=torch.zeros_like(classification)
            one_hot_output[mask]=1
            model.zero_grad()
            classification.backward(gradient=one_hot_output, retain_graph=True)
            weights=extractor.gradients[0].detach().cpu()
            weights=weights.permute(0,2,3,1).numpy()

            img_tensor = data['img'][0]
            size=img_tensor.size(1)
            img_metas = data['img_meta'][0].data[0]*size
            imgs = tensor2imgs(img_tensor[0,...], **cfg.img_norm_cfg)
            for j,cam in enumerate(weights):
                img_meta=img_metas[j]
                img = imgs[j]
                #img=img.permute(1,2,0).numpy()
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                img_show = mmcv.imrescale(img_show,1/img_meta['scale_factor'])
                cv2.imwrite(f'./results/grad_sample/{precess_video}/org/{i}_{j}.jpg',img_show)

                #print(img.shape)
                img=Image.fromarray(np.array(img,dtype=np.uint8))
                #print(cam.shape)
                cam=cam.mean(axis=2)
                # #cam basic knowledge
                # print('min', cam.min())
                # print('mean',cam.mean())
                # print('max',cam.max())
                # print('std',cam.std()) #around 0.012
                std=inputstd
                xtime=500
                cam=abs(cam)-std
                cam=np.where(cam>0,cam,0)
                cam=np.where(cam<std*xtime,cam,std*xtime)

                #apply 3 std ways
                # cam_std=cam.std()
                # cam_mean=cam.mean()

                #cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
                cam=cam/std/xtime
                cam = np.uint8(cam * 255)

                no_tran_heatmap,heatmap ,heatmap_on_image = apply_colormap_on_image(img, cam, 'Blues')
                heatmap_on_image=cv2.cvtColor(np.array(heatmap_on_image),cv2.COLOR_RGBA2RGB)
                no_tran_heatmap=cv2.cvtColor(np.array(no_tran_heatmap),cv2.COLOR_RGBA2RGB)
                heatmap=cv2.cvtColor(np.array(heatmap),cv2.COLOR_RGBA2RGB)

                heatmap_on_image = heatmap_on_image[:h, :w, :]
                heatmap_on_image = mmcv.imrescale(heatmap_on_image,1/img_meta['scale_factor'])

                no_tran_heatmap = no_tran_heatmap[:h, :w, :]
                no_tran_heatmap = mmcv.imrescale(no_tran_heatmap,1/img_meta['scale_factor'])

                heatmap = heatmap[:h, :w, :]
                heatmap = mmcv.imrescale(heatmap,1/img_meta['scale_factor'])

                cv2.imwrite(f'./results/example/cam_{inputstd}/{precess_video}/heatmap/{i}_{j}.jpg',heatmap)
                cv2.imwrite(f'./results/example/cam_{inputstd}/{precess_video}/heatmap_on_image/{i}_{j}.jpg',heatmap_on_image)
                cv2.imwrite(f'./results/example/cam_{inputstd}/{precess_video}/no_tran_heatmap/{i}_{j}.jpg',no_tran_heatmap)

if __name__ == "__main__":
    main()
