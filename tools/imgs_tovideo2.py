import cv2
import os
import mmcv
import numpy as np
import tqdm
result_dir= './results/more/tcm_predict/'
video_FourCC    = cv2.VideoWriter_fourcc(*'mp4v')
fps=24
outpath='./results/more/videos/'
# videos=os.listdir(path1)
mmcv.mkdir_or_exist(outpath)
videos=os.listdir(result_dir)
for video in videos:
    frames=os.listdir(os.path.join(result_dir,video))
    frames=sorted(frames,key=lambda x:int(x.split('_')[0])*2+int(x.split('_')[1]))
    img=cv2.imread(os.path.join(result_dir,video,frames[0]))
    h,w,_=img.shape
    output_path=outpath+video # later fix this
    #build writer
    writer=cv2.VideoWriter(output_path, video_FourCC, fps, (w,h))

    for i in tqdm.tqdm(range(len(frames))):
        frame=frames[i]
        img=cv2.imread(os.path.join(result_dir,video,frame))
        writer.write(img)
    writer.release()

