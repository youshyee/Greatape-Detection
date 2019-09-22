'''
given a wordking dir 
calculate the result for each epoch saving and save it as txt file
'''
import os
import mmcv
import argparse
import os.path as osp
import shutil
import tempfile

import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
import subprocess

files=os.listdir('/mnt/storage/home/rn18510/')
folders=[f for f in files if 'slurm_mm' in f]
all_shfiles=[]
for folder in folders:
  root=os.path.join('/mnt/storage/home/rn18510/',folder)
  shfiles = os.listdir(root)
  shfiles = [os.path.join(root,i) for i in shfiles if '.sh' in i]
  all_shfiles+=shfiles

for shfile in all_shfiles:
  list_sh=mmcv.list_from_file(shfile)
  for line in list_sh:
    if 'WORK_DIR=' in line:
      workdir=line
    if 'CONFIG=' in line:
      config=line
  
  workdir=workdir.replace('WORK_DIR=','')
  config=config.replace('CONFIG=','')
  if os.path.exists(workdir):
    pass
  else:
    print('not exe')
    continue
  print(workdir)
  print(config)
  all_result_file=[i for i in os.listdir(workdir) if '.result' in i]
  all_pth_file=[i for i in os.listdir(workdir) if '.pth' in i and 'latest' not in i]
  to_exe_pth=[]
  if len(all_result_file)>11 and len(all_pth_file)>11:
    #find the epoch
    best=sorted(all_result_file,key = lambda x:int(x.split('.')[0].split('_')[-1]))[-1]
    latest=sorted(all_pth_file,key = lambda x : int(x.split('.')[0].split('_')[-1]))[-1]
    ep=int(best.split('.')[0].split('_')[0].replace('ep',''))
    ep='epoch_{}.pth'.format(ep)
    if ep ==latest:
      pass
    else:
      to_exe_pth.append(ep)
    to_exe_pth.append(latest)
  else:
    to_exe_pth+=all_pth_file
  all_txt_file=[i for i in os.listdir(workdir) if '.txt' in i]
  txt_eps=[i.split('.')[0].split('_')[-1].replace('ep','') for i in all_txt_file]
  to_exe_eps=[i.split('.')[0].split('_')[-1] for i in to_exe_pth]
  to_exe_eps=list(set(to_exe_eps)-set(txt_eps))
  to_exe_pth = ['epoch_{}.pth'.format(ep) for ep in to_exe_eps]

  #filter already has .txt file epoch
  for exe_pth in to_exe_pth:
    print('runing',config,workdir,exe_pth)
    subprocess.run(['sh','tools/dist_test.sh','{}'.format(config),'{}'.format(os.path.join(workdir,exe_pth)),'2','--work_dir','{}'.format(workdir)])


  #exe
