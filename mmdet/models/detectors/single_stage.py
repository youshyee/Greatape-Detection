import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result
from mmcv.parallel import DataContainer as DC

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video=False,
                 test_reduce=False,
                 train_reduce=False
                 ):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.video=video
        self.test_reduce=test_reduce
        self.train_reduce=train_reduce

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        #print(img.shape)
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # print('###################')
        # print('meta')
        # print('type',type(img_metas),len(img_metas))
        # print('type',type(img_metas[0]),len(img_metas[0]))
        # print('type',type(img_metas[0][0]),len(img_metas[0][0]))

        if self.video:
            if not self.train_reduce:
                gt_bboxes=gt_bboxes[0]
                gt_labels=gt_labels[0]
                img_metas=img_metas*len(gt_bboxes)
            else:
                snip_size=len(gt_bboxes)
                gt_bboxes=gt_bboxes[0][snip_size//2:snip_size//2+1]
                gt_labels=gt_labels[0][snip_size//2:snip_size//2+1]
                img_meta=img_meta
        # print('gt_bbx')
        # print(len(gt_bboxes))
        # print(len(gt_bboxes[0]))
        # print(len(gt_bboxes[0][0]))
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False,graph_mode=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if graph_mode:
            return outs
        if self.video:
            if not self.test_reduce:
                bbox_inputs = outs + (img_meta*len(x[0]), self.test_cfg, rescale)
            else:
                bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        else:
            bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        if self.video:
            return bbox_results
        else:
            return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
