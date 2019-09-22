import os.path as osp
import xml.etree.ElementTree as ET
import os
import mmcv
import numpy as np
import torch
from .custom import CustomDataset
import tqdm


class DET(CustomDataset):
    CLASSES = ('n02510455',
                'n02342885',
                'n02355227',
                'n02084071',
                'n02402425',
                'n02691156',
                'n04468005',
                'n02324045',
                'n02509815',
                'n02958343',
                'n02924116',
                'n02484322',
                'n04530566',
                'n02062744',
                'n01503061',
                'n02503517',
                'n01674464',
                'n02131653',
                'n02129165',
                'n01662784',
                'n03790512',
                'n02834778',
                'n02118333',
                'n02419796',
                'n02121808',
                'n01726692',
                'n02411705',
                'n02129604',
                'n02391049',
                'n02374451')
    id2name = {'n02510455': 'giant_panda',
                'n02342885': 'hamster',
                'n02355227': 'squirrel',
                'n02084071': 'dog',
                'n02402425': 'cattle',
                'n02691156': 'airplane',
                'n04468005': 'train',
                'n02324045': 'rabbit',
                'n02509815': 'lesser_panda',
                'n02958343': 'car',
                'n02924116': 'bus',
                'n02484322': 'monkey',
                'n04530566': 'vessel',
                'n02062744': 'whale',
                'n01503061': 'bird',
                'n02503517': 'elephant',
                'n01674464': 'lizard',
                'n02131653': 'bear',
                'n02129165': 'lion',
                'n01662784': 'turtle',
                'n03790512': 'motorcycle',
                'n02834778': 'bicycle',
                'n02118333': 'fox',
                'n02419796': 'antelope',
                'n02121808': 'cat',
                'n01726692': 'snake',
                'n02411705': 'sheep',
                'n02129604': 'tiger',
                'n02391049': 'zebra',
                'n02374451': 'horse'}
    txtfiles=[2, 4, 20, 24, 26, 33, 37, 39, 58, 59, 64, 70, 74, 84, 92, 103, 105, 113, 114, 141, 144, 155, 159, 166, 182, 185, 188, 197, 198, 200]
    def __init__(self, val_mode,load_cache=False,debug=False,**kwargs):
        self.val_mode = val_mode
        self.load_cache=load_cache
        self.debug=debug
        super(DET, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
    def load_annotations(self, ann_file):
        p='/mnt/storage/scratch/rn18510/mmdetection/cache/det_val.list' if self.val_mode else '/mnt/storage/scratch/rn18510/mmdetection/cache/det_train.list'
        if self.load_cache and os.path.exists(p):
            print('loading cache file {}'.format(p))
            states=torch.load(p)
            img_infos=states[0]
            print(states[1])
        else:
            img_infos = []
            if not self.val_mode:
                train_val_files=['train_{}.txt'.format(i) for i in self.txtfiles]
            else:
                train_val_files=['val.txt']
            mode = 'val' if self.val_mode else 'train'
            damaged=0
            other_classes=0
            noimgs=0
            nolabel=0
            for each_file in train_val_files:
                each_file = osp.join(ann_file,each_file)
                img_ids = mmcv.list_from_file(each_file)
                print('processing_file:',each_file)
                for img_id in tqdm.tqdm(img_ids):
                    img_id = img_id.split()[0]
                    filename = osp.join('Data/DET/{}'.format(mode),'{}.JPEG'.format(img_id))
                    if not os.path.isfile(osp.join(self.img_prefix,filename)):
                        noimgs+=1
                        continue
                    xml_path = osp.join(self.img_prefix, 'Annotations/DET/{}'.format(mode),
                                        '{}.xml'.format(img_id))
                    if not os.path.isfile(xml_path):
                        nolabel+=1
                        continue
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    objs=root.findall('object')
                    if len(objs)==0:
                        damaged+=1
                        continue
                    if self.val_mode:
                        for obj in objs:
                            name = obj.find('name').text
                            if name in self.CLASSES:
                                pass
                            else:
                                break
                        else:
                            other_classes+=1
                            continue
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                    img_infos.append(
                        dict(id=img_id, filename=filename,xml_path=xml_path ,width=width, height=height))
            damage_info={'damaged imgs':damaged,'other_classes':other_classes,'nolabel':nolabel,'noimgs':noimgs}
            print('damaged imgs: ',damaged,'other_classes:',other_classes,'nolabel',nolabel,'noimgs',noimgs)
            torch.save([img_infos,damage_info],p)
        if self.debug:
            img_infos=img_infos[:1024]
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = self.img_infos[idx]['xml_path']
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            difficult = False if name in self.CLASSES else True
            if not difficult:
                label = self.cat2label[name]
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            if difficult:
                bboxes_ignore.append(bbox)
                #labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            #labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            #labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            #labels_ignore=labels_ignore.astype(np.int64)
            )
        return ann
