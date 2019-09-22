import cv2
import os
import mmcv
import numpy as np
import tqdm
result_dir= './results/'
video_FourCC    = cv2.VideoWriter_fourcc(*'mp4v')
fps=24
comparisons=os.listdir(result_dir)
assert 'temp' in comparisons
assert 'tcm_scm' in comparisons
comparisons=['tcm_scm','temp']
path1=result_dir+comparisons[0]
path2=result_dir+comparisons[1]
videos=os.listdir(path1)
mmcv.mkdir_or_exist(result_dir+'comparisons')
for video in videos:
    if video in os.listdir(result_dir+comparisons[1]):
        frames1=os.listdir(os.path.join(path1,video))
        frames1=sorted(frames1,key=lambda x:int(x.split('_')[0])*2+int(x.split('_')[1]))

        frames2=os.listdir(os.path.join(path2,video))
        frames2=sorted(frames1,key=lambda x:int(x.split('_')[0])*2+int(x.split('_')[1]))
        assert len(frames1)==len(frames2)
        img=cv2.imread(os.path.join(path1,video,frames1[0]))
        h,w,_=img.shape
        output_path=result_dir+'comparisons/'+video
        #build writer
        writer=cv2.VideoWriter(output_path, video_FourCC, fps, (w,h*2))

        for i in tqdm.tqdm(range(len(frames1))):
            frame1=frames1[i]
            frame2=frames2[i]
            img1=cv2.imread(os.path.join(path1,video,frame1))
            cv2.putText(img1,'with ours',(30,30), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255),2,cv2.LINE_AA)
            img2=cv2.imread(os.path.join(path2,video,frame2))
            cv2.putText(img2,'w/t ours',(30,30), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255),2,cv2.LINE_AA)
            newimg=np.vstack([img1,img2])
            writer.write(newimg)
        writer.release()
    else:
        print(f'can not find {video}')


