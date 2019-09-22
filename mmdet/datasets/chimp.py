import os
import os.path as osp
import xml.etree.ElementTree as ET
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
import tqdm
import collections
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import json
class Bbox(object):
    def __init__(self,x1,x2,y1,y2,tag,frame):
        if x1 is not None:
            self.x1=round(x1,2)
            self.x2=round(x2,2)
            self.y1=round(y1,2)
            self.y2=round(y2,2)
        else:
            self.x1=x1
            self.x2=x2
            self.y1=y1
            self.y2=y2
        self.tag=tag
        self.frame=frame
        assert isinstance(frame,int) and frame>=0
    def get_box(self):
        return (self.x1,self.y1,self.x2,self.y2)
    def get_frame(self):
        return self.frame
    def get_tag(self):
        return self.tag
    def __repr__(self):
        if self.is_none():
            return 'frame:{},tag:{}'.format(self.frame,self.tag)
        else:
            return 'frame:{},tag:{},box:{}'.format(self.frame,self.tag,self.get_box())
    def __str__(self):
        if self.is_none():
            return 'frame:{},tag:{}'.format(self.frame,self.tag)
        else:
            return 'frame:{},tag:{},box:{}'.format(self.frame,self.tag,self.get_box())
    def is_none(self):
        def isnone(c):
            if c is None:
                return True
            else:
                return False
        return isnone(self.x1) or isnone(self.x2) or isnone(self.y1) or isnone(self.y2) or isnone(self.tag)
class CHIMP(Dataset):

    CLASSES = ['Chimp']

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 multiscale_mode='value',
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=False,
                 with_label=True,
                 with_semantic_seg=False,
                 seg_prefix=None,
                 seg_scale_factor=1,
                 extra_aug=None,
                 aug_prob=[],
                 aug_p=0,
                 resize_keep_ratio=True,
                 test_mode=False,
                 min_val=False,
                 min_seed=1011729,
                 snip_frame=8,
                 how_sparse=3,
                 debug=False,
                 repeat_mode=False):
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.snip_frames=snip_frame
        self.how_sparse=how_sparse
        self.test_mode = test_mode
        self.min_val=min_val
        self.repeat_mode=repeat_mode
        self.ann_file=ann_file
        if repeat_mode:
            assert self.snip_frames>2 and self.snip_frames%2==1, 'snip frame should be odd number and larger than 2'
        self.debug=debug
        if self.test_mode:
            self.how_sparse=1
        else:
            self.min_val=False
        ############################################
        self.img_prefix = img_prefix
        self.img_infos,self.box_infos = self.load_annotations(ann_file)

        if self.min_val:
            np.random.seed(min_seed)
            np.random.shuffle(self.img_infos)
            self.img_infos=self.img_infos[::1]
        if self.debug:
            np.random.shuffle(self.img_infos)
            self.img_infos=self.img_infos[:16] # for testing

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = multiscale_mode
        assert multiscale_mode in ['value', 'range']

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = seg_scale_factor
        # in test mode or not

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()
        # if use extra augmentation
        self.aug_p=aug_p
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
            self.aug_prob=aug_prob
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio
    def __len__(self):
        return len(self.img_infos)
    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)
    def all_bbox_to_tubes(self,all_bbox):
        tube_names=set(i.get_tag() for i in all_bbox)
        tubes={}
        for tube_name in tube_names: 
            tubes[tube_name]=[]
        for bbox in all_bbox:
            tubes[bbox.get_tag()].append(bbox)
        return tubes
    def get_cands(self,l):
        p=0
        f=0
        l=sorted(l)
        a=[]
        for i,j in enumerate(l[:-1]):
            if j+1==l[i+1]:
                continue
            else:
                p=j
                f=l[i+1]
                if (f-p)<20 and (f-p)>0:
                    a.append((p,f))
        return a
    def get_bbox(self,f,tubes):
        for i in tubes:
            if i.get_frame()==f:
                return i
    def interplote(self,bbox1,bbox2):
        assert bbox2.get_frame()-bbox1.get_frame()>1
        assert bbox2.get_tag()==bbox1.get_tag()
        f1=bbox1.get_frame()
        f2=bbox2.get_frame()
        inter_num=f2-f1-1
        assert inter_num>0
        box1=bbox1.get_box()
        box2=bbox2.get_box()
        diff=[None,None,None,None]
        each_delta=[None,None,None,None]
        for i in range(4):
            diff[i]=box2[i]-box1[i]
            each_delta[i]=diff[i]/(1+inter_num)
        inter_boxes=[]
        for j in range(inter_num):
            box=[None,None,None,None]
            for i in range(4):
                box[i]=box1[i]+each_delta[i]*(j+1)
            f=f1+j+1
            bbox=Bbox(x1=box[0],y1=box[1],x2=box[2],y2=box[3],tag=bbox1.get_tag(),frame=f)
            inter_boxes.append(bbox)
        return inter_boxes
    def tubes2bboxes(self,tubes):
        out={}
        for tube in tubes:
            if tube is not None:
                for bbox in tubes[tube]:
                    f=bbox.get_frame()
                    if f not in out:
                        out[f]=[]
                        out[f].append(bbox)
                    else:
                        out[f].append(bbox)
        return out
    def is_continuous(self,l):
        l=sorted(l)
        return len(l)==(max(l)-min(l)+1)
    def process_imgid(self,frame_id,basename,shape):
        filename =basename+'_frame_{}'.format(frame_id)+'.jpg'
        return dict(id=frame_id, filename=filename, video_prefix=basename,width=shape[1],height=shape[0])

    def get_snippet(self, basename, len_of_video, num_snippets,f_num,shape):
        '''Group snippets.
            Returns:
          grouped_snippet_frame: (list) [[cand1,cand2,...],.....] cand:[filename]*#snip_frames
          grouped_snippet_label: (list) [[cand1,cand2,...],....] cand:[filename]*#num_snip_frames
        '''
        def index2meta(cand,basename,f_num,shape):
            video_data_cand=[]
            for each in cand:
                frame_id=f_num[each]
                filename =basename+'_frame_{}'.format(frame_id)+'.jpg'
                video_data_cand.append(dict(id=frame_id, filename=filename, video_prefix=basename,width=shape[1],height=shape[0]))
            return video_data_cand

        frames=[i for i in range(len_of_video)]
        grouped_snippet_frame=[]
        for i in range(num_snippets):
            cands=[]
            for j in range(self.how_sparse):
                cand=frames[j+i*self.snip_frames*self.how_sparse:j+(i+1)*self.how_sparse*self.snip_frames:self.how_sparse]
                if len(cand)!=self.snip_frames and i!=0:
                    diff=self.snip_frames-len(cand)
                    cand=frames[j+i*self.snip_frames*self.how_sparse-self.how_sparse*diff:j+(i+1)*self.how_sparse*self.snip_frames-self.how_sparse*diff:self.how_sparse]
                if len(cand)!=self.snip_frames:
                    diff=self.snip_frames-len(cand)
                    cand=cand+[frames[-1]]*diff
                cand=index2meta(cand,basename,f_num,shape)
                cands.append(cand)
            grouped_snippet_frame.append(cands)
        return grouped_snippet_frame
    def load_annotations(self, ann_file):
        sort_=lambda x:int(x.split('.')[0])
        sort_bboxes=lambda b :sorted(b,key=lambda x:x.get_frame())
        snip_cand_infos = []
        box_infos={}
        videos= os.listdir(ann_file)
        videos=[i for i in videos if 'output' in i]
        videos=sorted(videos)
        num_videos=len(videos)
        #split train and val with random seed
        a=np.arange(len(videos))
        np.random.seed(10)
        np.random.shuffle(a)
        np.random.seed()
        val_index=a[:int(0.1*num_videos)]
        train_index=a[int(0.1*num_videos):]
        train_videos=[videos[i] for i in train_index]
        val_videos=[videos[i] for i in val_index]
        if self.test_mode:
            target_videos=val_videos
        else:
            target_videos=train_videos

        damage_videos=0
        print('Loading to memory ...')
        for video in tqdm.tqdm(target_videos): # vidoe in formate 'somethin_out'
            video_prefix=os.path.join(ann_file,video)
            basename=video.split('_')[0]
            annot_path=os.path.join(ann_file,basename+'.mp4.json')
            assert os.path.isfile(annot_path),'error annot path : '+annot_path
            assert 'JPEGImages' in os.listdir(video_prefix),'no jpeg folder : {}'.format(basename)
            frames= os.listdir(os.path.join(video_prefix,'JPEGImages'))
            frames=sorted(frames,key=lambda x:int(x.split('_')[-1].split('.')[0])) # start with frame 1
            frames_num=[int(i.split('_')[-1].split('.')[0]) for i in frames]
            # interpolation
            with open(annot_path) as file:
                data=json.load(file)
            all_bboxes=[]
            for f in data['frames']:
                list_bboxes=data['frames'][f]
                if not len(list_bboxes)==0: # if this frame is not empty
                    for each_box in list_bboxes:
                        tag=each_box['tags'][0]
                        x1=each_box['box']['x1']
                        x2=each_box['box']['x2']
                        y1=each_box['box']['y1']
                        y2=each_box['box']['y2']
                        orig_h=each_box['height']
                        orig_w=each_box['width']
                        bbox=Bbox(x1,x2,y1,y2,tag,int(f))
                        all_bboxes.append(bbox)
                else:
                    bbox=Bbox(x1=None,x2=None,y1=None,y2=None,tag=None,frame=int(f))
                    all_bboxes.append(bbox)
            tubes=self.all_bbox_to_tubes(all_bboxes)
            for tube in tubes:
                if tube is None:
                    continue
                bboxes=tubes[tube]
                fs=sorted([i.get_frame() for i in bboxes])
                pairs=self.get_cands(fs)
                for pair in pairs:
                    bbox1=self.get_bbox(pair[0],bboxes)
                    bbox2=self.get_bbox(pair[1],bboxes)
                    tubes[tube]+=self.interplote(bbox1,bbox2)
                    tubes[tube]=sort_bboxes(tubes[tube])
            dic_bboxes=self.tubes2bboxes(tubes)
            box_infos[f'{basename}']=dic_bboxes
            # if not self.is_continuous(list(dic_bboxes.keys())):
            #     print('not continus:', list(dic_bboxes.keys())) # make sure it is continuous
            f_num=sorted(list(dic_bboxes.keys()))
            len_video=len(f_num)

            if not self.repeat_mode:
                num_snippets, remain =divmod(len_video,self.snip_frames*self.how_sparse)
                if remain/(self.snip_frames*self.how_sparse)>0.4:
                    num_snippets+=1
                # exclude the no one
                if num_snippets==0:
                    damage_videos+=1
                    continue
                cand_snippets=self.get_snippet(basename,len_video,num_snippets,f_num,(orig_h,orig_w))
            else:
                num_seg, remain =divmod(len_video,self.how_sparse)
                if remain/self.how_sparse>0.6:
                    num_seg+=1
                    f_num=f_num+[f_num[-1]]*(self.how_sparse-remain) # make img_ids can be divided by how_sparse
                if num_seg==0:
                    damage_videos+=1
                    continue
                if len(f_num)<self.how_sparse*self.snip_frames:
                    damage_videos+=1
                    continue
                grouped=[]
                for i in range(num_seg):
                    togroup=f_num[i*self.how_sparse:(i+1)*self.how_sparse]
                    togroup=[self.process_imgid(i,basename,(orig_h,orig_w)) for i in togroup]
                    grouped.append(togroup)
                if num_seg < self.snip_frames:
                    damage_videos+=1
                    continue
                assert num_seg>=self.snip_frames,'num seg not enough got {}'.format(num_seg)
                cand_snippets=[]
                radius=(self.snip_frames-1)//2
                for i in range(num_seg):
                    region = min(i,num_seg-i-1)
                    if region < radius: #head and tail frame need special
                        head = True if i< num_seg-i-1 else False
                        diff=radius-region
                        if head:
                            cand_snippet=grouped[diff:0:-1]+grouped[:i+radius+1]
                        else:
                            cand_snippet=grouped[i-radius:]+grouped[-2:-2-diff:-1]
                        assert len(cand_snippet) == self.snip_frames, 'cand_snip in head tail special region fail'
                    else:
                        cand_snippet=grouped[i-radius:i+radius+1]
                        assert len(cand_snippet) == self.snip_frames, '333cand_snip in head tail' 
                    cand_snippet=list(zip(*cand_snippet))
                    assert len(cand_snippet[0]) == self.snip_frames, '222cand_snip in head tail special region fail {} vs {}'.format(len(cand_snippet[0]),self.snip_frames)
                    cand_snippets.append(cand_snippet)

            snip_cand_infos+=cand_snippets
        print('damage videos',damage_videos)
        return snip_cand_infos,box_infos
    def get_ann_info(self, idx):
        assert len(self.img_infos[0]) == self.how_sparse
        rand_num=np.random.randint(0,len(self.img_infos[0]))
        #print(rand_num,self.img_infos[idx][rand_num][0]['video_prefix']+':'+self.img_infos[idx][rand_num][0]['id'])
        anns=[]
        for i in range(self.snip_frames):
            img_id = self.img_infos[idx][rand_num][i]['id']
            img_name= self.img_infos[idx][rand_num][i]['filename']
            basename=self.img_infos[idx][rand_num][i]['video_prefix']
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            list_of_boxes=self.box_infos[basename][img_id]
            for box in list_of_boxes:
                if box.is_none():
                    continue
                else:
                    xmin,ymin,xmax,ymax=box.get_box()
                    bbox=[xmin,ymin,xmax,ymax]
                    bboxes.append(bbox)
                    labels.append(1)
            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ))
            else:
                bboxes = np.array(bboxes, ndmin=2) - 1
                labels = np.array(labels)
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0, ))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
                labels_ignore = np.array(labels_ignore)
            ann = dict(
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64),
                bboxes_ignore=bboxes_ignore.astype(np.float32),
                labels_ignore=labels_ignore.astype(np.int64))
            anns.append(ann)
        return anns,rand_num
    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = list(range(len(self.img_infos)))
        return valid_inds
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            img_info=img_info[0][0]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    def prepare_train_img(self, idx):
        anns,rand_num = self.get_ann_info(idx)
        #print('rand_num',rand_num)
        img_info = self.img_infos[idx][rand_num]
        # load image
        imgs=[]
        #infos=[]
        for each_img_info in img_info:
            img = mmcv.imread(osp.join(self.ann_file,each_img_info['video_prefix']+'_output','JPEGImages' ,each_img_info['filename']))
            imgs.append(img)
            #infos.append({'video_id':each_img_info['video_prefix'],'frame_id':each_img_info['filename']})
        # load proposals if necessary
        orig_h,orig_w,_=imgs[0].shape
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None
        gt_bboxes = []
        gt_labels = []
        for ann in anns:
            gt_bboxes.append(ann['bboxes'])
            gt_labels.append(ann['labels'])

        # skip the image if there is no valid gt bbox
        if any([len(i)==0 for i in gt_bboxes]):
            return None

        # extra augmentation
        np.random.seed()
        if self.extra_aug is not None and np.random.rand()<self.aug_p:
            aug_status=[True if np.random.rand() < aug_pro else False for aug_pro in self.aug_prob]
            seeds=[np.random.randint(1e16) for aug_pro in self.aug_prob]
            for i in range(len(imgs)):
                g_b=gt_bboxes[i].clip(0.)
                g_l=gt_labels[i]
                assert len(g_b)==len(g_l)
                aug_input={'image':imgs[i],'bboxes':g_b.tolist(),'category_id':g_l.tolist(),'each_force_apply':aug_status,'seeds':seeds}
                aug_out = self.extra_aug(aug_input)
                imgs[i]=aug_out['image'].copy()

                auged_box=aug_out['bboxes']
                auged_label=aug_out['category_id']

                auged_box=np.array(auged_box).astype(np.float32)
                auged_label=np.array(auged_label).astype(np.int64)

                gt_bboxes[i]=auged_box.copy()
                gt_labels[i]=auged_label.copy()
        
        if any([len(i)==0 for i in gt_bboxes]):
            return None
        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        imgs_=[]
        gt_bboxes_=[]
        for i,img in enumerate(imgs):
            img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
            img = img.copy()
            if self.proposals is not None:
                proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                                flip)
                proposals = np.hstack(
                    [proposals, scores]) if scores is not None else proposals
            gt_bbox = self.bbox_transform(gt_bboxes[i], img_shape, scale_factor,
                                            flip)
            gt_bbox=gt_bbox.copy()
            if self.with_crowd:
                gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                    scale_factor, flip)
            if self.with_mask:
                gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                            scale_factor, flip)

            imgs_.append(img)
            gt_bboxes_.append(gt_bbox)
        _, img_shape, pad_shape, scale_factor = self.img_transform(
                imgs[0], img_scale, flip, keep_ratio=self.resize_keep_ratio)
        ori_shape = (orig_h, orig_w, 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            videoframe=os.path.basename(img_info[0]['video_prefix'])+'+'+str([img_info[i]['id'] for i in range(len(imgs))]))

        data = dict(
            img=DC(to_tensor(imgs_), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC([to_tensor(gt_bbox) for gt_bbox in gt_bboxes_]))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC([to_tensor(gt_label) for gt_label in gt_labels])
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)

        return data
    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx][0]
        def prepare_single(imgs, scale, flip, orig_shape,proposal=None):
            _imgs=[]
            for img in imgs:
                _img, img_shape, pad_shape, scale_factor = self.img_transform(
                    img, scale, flip, keep_ratio=self.resize_keep_ratio)
                _imgs.append(_img)
            frame_ids = [int(i['id']) for i in img_info]
            if self.repeat_mode:
                frame_ids = frame_ids[len(imgs)//2:len(imgs)//2+1]
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            _img_meta = dict(
                ori_shape=(orig_shape[0], orig_shape[1], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
                #meta about val frame id
                frame_ids=frame_ids
                )
            return to_tensor(_imgs), _img_meta, _proposal
        imgs=[]
        for each_img_info in img_info:
            img = mmcv.imread(osp.join(self.ann_file,each_img_info['video_prefix']+'_output','JPEGImages', each_img_info['filename']))
            imgs.append(img)
        orig_h,orig_w,_=imgs[0].shape
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None
        images=[]
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _imgs, _img_meta, _proposal = prepare_single(
                imgs, scale, False, (orig_h,orig_w),proposal)
            images.append(_imgs)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True,(orig_h,orig_w),proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=images, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
    def collate_fn(self, batch, samples_per_gpu=1):
        """Puts each data field into a tensor/DataContainer with outer dimension
        batch size.
        Extend default_collate to add support for
        :type:`~mmcv.parallel.DataContainer`. There are 3 cases.
        1. cpu_only = True, e.g., meta data
        2. cpu_only = False, stack = True, e.g., images tensors
        3. cpu_only = False, stack = False, e.g., gt bboxes
        """

        if not isinstance(batch, collections.Sequence):
            raise TypeError("{} is not supported.".format(batch.dtype))

        if isinstance(batch[0], DC):
            assert len(batch) % samples_per_gpu == 0
            stacked = []
            if batch[0].cpu_only:
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append(
                        [sample.data for sample in batch[i:i + samples_per_gpu]])
                return DC(
                    stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
            elif batch[0].stack:
                for i in range(0, len(batch), samples_per_gpu):
                    assert isinstance(batch[i].data, torch.Tensor)
                    # TODO: handle tensors other than 3d
                    assert batch[i].dim() == 4
                    s,c, h, w = batch[i].size()
                    for sample in batch[i:i + samples_per_gpu]:
                        assert s == sample.size(0)
                        h = max(h, sample.size(2))
                        w = max(w, sample.size(3))
                    padded_samples = [
                        F.pad(
                            sample.data,
                            (0, w - sample.size(3), 0, h - sample.size(2)),
                            value=sample.padding_value)
                        for sample in batch[i:i + samples_per_gpu]
                    ]
                    stacked.append(default_collate(padded_samples))
            else:
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append(
                        [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DC(stacked, batch[0].stack, batch[0].padding_value)
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [self.collate_fn(samples, samples_per_gpu) for samples in transposed]
        elif isinstance(batch[0], collections.Mapping):
            return {
                key: self.collate_fn([d[key] for d in batch], samples_per_gpu)
                for key in batch[0]
            }
        else:
            return default_collate(batch)


