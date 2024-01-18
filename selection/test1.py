#! /usr/bin/env python3

from skimage import io
from torchvision.ops import nms
import os
import glob
import cv2
import torch
import matplotlib.pyplot as plt
# import selcpu
import selcuda
from maskcreate import maskcreate
import tqdm
import time
from torchvision import transforms

# torch.ops.load_library("build/libselection_cpu_cmake.so")
# print(torch.ops.sel_cpu.selection_cpu)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
dir = "../visualizer"
# cate = [path+"/"+x for x in os.listdir(path) if os.path.isdir(path+"/"+x)]
# for i in glob.glob(path+"/*.jpg"):
#   print(i)
#   break
timetotal=0
size = 21
delta = 110
ratio = torch.tensor([0.5,0.7,0.9],dtype=torch.float32,device=device)
type = "contr"
mask = maskcreate(size,type)[0]
num = maskcreate(size,type)[1]
toPIL = transforms.ToPILImage()
# generate = (x for x in range(50))
time0 = time.perf_counter()
for i in range(50):
  path = os.path.join(dir,"image"+str(i)+".jpg")
  img = torch.load(path)[0]#.type(torch.cuda.FloatTensor).type(torch.uint8) [480, 640, 3]
  # img=torch.from_numpy(io.imread(path))
  # print(img.shape) #torch.uint8,type(img),img.dtype
  # plt.figure(0)
  # plt.subplot(2,3,2)
  # plt.imshow(img)
  # print("img_shape:",img.shape)
  img_gpu = img.to(device).type(torch.cuda.FloatTensor)
  # print("imggpu_shape:",img_gpu.shape,type(img_gpu))
  time1 = time.perf_counter()
  testout = selcuda.selection(img_gpu,mask.to(device),num,delta,ratio).cpu().type(torch.uint8)
  time2 = time.perf_counter()
  timeper = time2-time1
  timetotal+=timeper
  print("time cost for pic"+str(i)+":",round(timeper,3)) #0.373

  # for j in range(3):
    # print("img:",img[422,422:512,j],img[422,422:512,j].shape)
    # print("testout:",testout.shape,testout[0,422,422:512,j])
    # print("diff("+str(j)+"):",max(img[422,422:512,j])-min(img[422,422:512,j]),max(testout[0,422,422:512,j])-min(testout[0,422,422:512,j]))

  for i in range(3):
    # print("testshape:",testout[i].shape)
    output = toPIL(testout[i].permute(2,0,1))
    plt.figure(i+1)
    # plt.subplot(2,3,4+i)
    plt.imshow(output)
  plt.show()
print("time cost total:",round(timetotal,3),"\n","time cost per pic average:",round(timetotal/50,3))



















