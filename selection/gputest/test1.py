#! /usr/bin/env python
from ctypes.wintypes import POINT
import numpy as np
from torchvision.ops import nms
import os
import cv2
import torch
import matplotlib.pyplot as plt
# import selcpu
import selcuda
import sys
sys.path.append("..")
from maskcreate import maskcreate
import cupy as cp
from ctypes import *
torch.set_printoptions(profile="full")

# torch.ops.load_library("build/libselection_cpu_cmake.so")
# print(torch.ops.sel_cpu.selection_cpu)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

path = os.path.join("../image1.jpg")
img = torch.load(path)[0]#.type(torch.cuda.FloatTensor).type(torch.uint8) [480, 640, 3]
print(img.shape,type(img),img.dtype) #torch.uint8
# width = img.shape[1]
# height = torch.from_numpy(img.shap[0]).type(torch.cuda.FloatTensor)
# print(width,type(width)) #640 <class 'int'>
plt.figure(0)
# plt.subplot(2,3,2)
plt.imshow(img)
# plt.show()
'''========================cpu experiment======================'''
# img = selcpu.selection_cpu(img)
# print(img[0].shape,type(img[0]),img[0].dtype,img[1],img[1].dtype)
# plt.imshow(img[0])
# plt.show()
'''======================gpu experiment satage1=================='''
# img_gpu = img.to(device).type(torch.cuda.FloatTensor)
# output = selcuda.selection(img_gpu).cpu().type(torch.uint8)
# print(output.shape,type(output),output.is_cuda,output.dtype)
# print("img:",img[100:110,100,0])
# print("output:",output[100:110,100,0])
# plt.imshow(output)
# plt.show()

'''=====================gpu experiment statage2 mask=============='''
# mask = maskcreate(11,"star")
# num = maskcreate(11,"cross")[1]
# print("mask:",mask.shape,type(mask),mask[:,:,0])
# img_gpu = img.to(device).type(torch.cuda.FloatTensor)
# maskout = selcuda.selection(img_gpu,mask.to(device),20,0.5).cpu()
# print("maskoutput:",maskout.shape,maskout[1001,:,:,2])

'''===============gpu experiment stage3&4 image transform========'''
size = 25
type = "contr"
mask = maskcreate(size,type)[0]
num = maskcreate(size,type)[1]
# print("mask:",mask[0].shape,type(mask),mask)
img_gpu = img.to(device).type(torch.cuda.FloatTensor)
delta = 110
ratio = torch.tensor([0.5,0.7,0.9],dtype=torch.float32,device=device)
testout = selcuda.selection(img_gpu,mask.to(device),num,delta,ratio).cpu().type(torch.uint8)
#========patchout check:
# print("img:",img[170:181,0:11,2],img[170:181,0:11,2].shape)
# print("testout:",testout.shape,testout[0,108800,:,:,2])
#========output check:h422:512 w299:351
for j in range(3):
  print("img:",img[422,422:512,j],img[422,422:512,j].shape)
  print("testout:",testout.shape,testout[0,422,422:512,j])
  print("diff("+str(j)+"):",max(img[422,422:512,j])-min(img[422,422:512,j]),max(testout[0,422,422:512,j])-min(testout[0,422,422:512,j]))
for i in range(3):
  plt.figure(i+1)
  # plt.subplot(2,3,4+i)
  plt.imshow(testout[i])

plt.show()


















