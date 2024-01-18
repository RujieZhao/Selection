#! /usr/bin/env python

from torchvision.ops import nms
import os
import cv2
import torch
import matplotlib.pyplot as plt
import selcpu
# import selcuda
print(selcpu.selection_cpu(torch.rand(3, 3,6)))

# torch.ops.load_library("build/libselection_cpu_cmake.so")
# print(torch.ops.sel_cpu.selection_cpu)
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:1" if use_cuda else "cpu")
#
# path = os.path.join("./image1.jpg")
# img = torch.load(path)[0]#.type(torch.cuda.FloatTensor).type(torch.uint8)
# print(img.shape,type(img),img.dtype) #torch.uint8
# test = selcpu.selection_cpu(img)
# print(len(test),test[0].shape,type(test[0]),test[0])

# img = selcpu.selection_cpu(img)
# print(img[0].shape,type(img[0]),img[0].dtype,img[1],img[1].dtype)
# plt.imshow(img[0])
# plt.show()
# img_gpu = img.to(device).type(torch.cuda.ByteTensor)
# img1 = img.permute(2,0,1).contiguous().cpu()
# img1 = torch.unsqueeze(img1,0)
# print(img1.shape,type(img1),img1.dtype)
# output = selcuda.selection(img)[0].cpu()
# output = torch.unsqueeze(output,0)
# print(output.shape,type(output))





# cv2.imshow(img)
# cv2.waitKey()
# boxes = torch.randint(3, 5, (3,4))
# boxes=boxes.type(torch.DoubleTensor)
# print(boxes.shape,type(boxes),boxes.dtype)
# #
# score = torch.rand(3)
# print(score)
# #
# iou_threshold=torch.tensor(0.7)
# #
# NMS = torch.ops.torchvision.nms(boxes,score,iou_threshold)
# print(NMS)








