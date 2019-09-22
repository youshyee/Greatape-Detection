from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret

def vid_result2txt(results,savepath):
    id_maps={0: 12,
            1: 13,
            2: 23,
            3: 8,
            4: 7,
            5: 0,
            6: 25,
            7: 19,
            8: 20,
            9: 6,
            10: 5,
            11: 17,
            12: 27,
            13: 28,
            14: 4,
            15: 10,
            16: 16,
            17: 2,
            18: 15,
            19: 26,
            20: 18,
            21: 3,
            22: 11,
            23: 1,
            24: 9,
            25: 22,
            26: 21,
            27: 24,
            28: 29,
            29: 14}
    with open(savepath,'w+') as outfile:
        for each_snip in results:
            result=each_snip['result']
            ids=each_snip['ids']
            assert len(ids) == len(result)
            for idx,each_frame in enumerate(result):
                assert len(each_frame)==30
                frame_id=ids[idx]
                for i,each_class in enumerate(each_frame):
                    #class transform
                    clss=id_maps[i]
                    if each_class.size==0:
                        continue
                    else:
                        for box in each_class:
                            txt='{} {} {} {} {} {} {}'.format(frame_id,clss,box[-1],box[0],box[1],box[2],box[3])
                            print(txt,file=outfile)


