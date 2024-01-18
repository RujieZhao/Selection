import torch
import numpy as np
from . import selecpycuda,selecbox
from detectron2.structures.boxes import Boxes
import time
# torch.set_printoptions(precision=6)
# @torch.no_grad()
def IS2box(mask,delta,checkboard,img_height,img_width,mask_th=0.5,patch=7,pylayer=2,num_ignore=1,amp=0.2,num_th=4) -> Boxes:
	"""
	this is a transmition from IS network to box prediction
	mask: mask output [img_h,img_w]
	delta: delta output [img_h,img_w,2]
	mask_th: Set it to 0.5. (if in train it is 0.7, if it's eval, it is 0.05. They are consistent with faster rcnn.)
	cardboard: Although polygon in annatation is represented by xyxyxy(width,height) format, I use (height,width) as my delta target. when I create checkboard, I follow the same format of delta.
	pylayer: the number of pyramid layers
	num_ignore: how many locations will be neglected in second pyramid layer
	amp: the limitation of the neglected value
	num_th: the minimum number of polygons
	return: instance boxes coordinate (XYXY)
	"""
	# time00 = time.perf_counter()
	# print("mask property:", mask.device, mask.shape)  # torch.Size([427, 640])
	assert mask.is_cuda and delta.is_cuda,"mask and delta should be cuda parameters."
	assert delta.shape == checkboard.shape, "checkboard shape is not match mask and delta!"

	#1.create cloud map
	# detach the required_grad
	cloudmap = torch.zeros((img_height,img_width)) #resemble to a counter
	valid_mask = (mask > mask_th).cpu()
	# print("valid_mask:",valid_mask.shape,checkboard.shape)#torch.Size([640, 1000]) torch.Size([640, 1000, 2])
	valid_indices = checkboard[valid_mask] #(H,W)
	# print("valid_indices:",valid_indices.device,valid_indices.shape) #cuda:0 torch.Size([192029, 2]),valid_indices
	valid_delta = delta[valid_mask].cpu()
	# print("valid_delta_shape:",valid_delta.shape,valid_delta[0:5]) #torch.Size([192029, 2])
	assert valid_indices.shape==valid_delta.shape
	# print("type test:",valid_indices.dtype,valid_delta.dtype,valid_indices.device,valid_delta.device) #torch.int64 torch.float32 cuda:0 cuda:0
	cloud_indices = torch.round(valid_indices-valid_delta).type(torch.ShortTensor)
	# print("cloud_indices:",cloud_indices.device,cloud_indices.shape,type(cloud_indices),cloud_indices.dtype)
	# In cuda kernal, I will start patch filter at 16th(coor[15]) pix due to coco small size is 32. But here, the result after abtractive delta should be full of image size.
	torch.clamp_(cloud_indices[...,0],min=0,max=(img_height-1))
	torch.clamp_(cloud_indices[...,1],min=0,max=(img_width-1))
	# print("cloud_indices_type:",cloud_indices.type(),cloud_indices.device,cloud_indices)
	# time0 = time.perf_counter()
	# print("torchtimetest1:", time0 - time00)
	for i in range(cloud_indices.shape[0]):
		cloudmap[cloud_indices[i,0],cloud_indices[i,1]]+=1
	# print("cloudmap:",cloudmap[465:476,741:751])

	# time1 = time.perf_counter()
	# print("for loop time:",time1 - time0) #0.0008
	cloudmap,finalcenter,finalcloud = selecpycuda.selecpy(cloudmap.to(mask.device),patch,pylayer,num_ignore,amp,num_th)
	# time2 = time.perf_counter()
	# print("pyramid time:",time2-time1) #0.0017
	# equlist = testmap.view([345, 470])
	# print("testmap:",equlist.shape,equlist[95:105,433:443]),finalcenter[103:114,452:461]
	# print("cuda cloudmap:",cloudmap.shape,cloudmap[465:476,741:751])
	# print("finalcenter:",finalcenter.shape,finalcenter[465:476,741:751])
	# print("finalcloud",finalcloud.shape,finalcloud[465:476,741:751],finalcloud.shape,type(finalcloud))

	"""find original coors for boxes creator"""
	# print("img_size:",img_height,img_width)
	finalcloud = finalcloud.view(img_height, img_width, 1).expand(img_height, img_width, 5).contiguous()
	# print("new finalcloud:",finalcloud[214:216,320],finalcloud.shape)
	# print("finalcloud_shape:", finalcloud.shape,finalcloud.get_device(),finalcloud.dtype)  # torch.Size([640, 1000, 5]) 0
	# total_ind = torch.arange(finalcenter.numel(),dtype=torch.float32,device=finalcenter.device)
	# print("finalcenter:",finalcenter.shape,finalcenter.numel()) # torch.Size([640, 1000]) 640000
	total_ind = torch.arange(finalcenter.numel(),dtype=torch.float32,device=finalcenter.device)
	finalcenterlist = finalcenter.flatten(0)
	assert (total_ind.shape==finalcenterlist.shape) & (finalcenterlist.shape[0]==img_height*img_width)
	# print(finalcenterlist.get_device(),finalcenterlist,finalcenterlist.dtype)
	finalcenter_valid = finalcenterlist>0
	finalcenterlist = finalcenterlist[finalcenter_valid]
	total_ind = total_ind[finalcenter_valid]
	# print("finalcenterlist:",finalcenterlist.shape,finalcenterlist,finalcenterlist.dtype)
	finalcenterlist = torch.stack((finalcenterlist,total_ind),dim=0).transpose(1,0).contiguous()
	# print("finalcenterlist:",finalcenterlist.shape,finalcenterlist.dtype,finalcenterlist.device,finalcenterlist[:,1]%img_width,(finalcenterlist[:,1]/img_width).type(torch.IntTensor)) #torch.Size([63442, 2]) cuda:0
	box_output = Boxes(selecbox.boxcreator(patch,finalcloud,finalcenterlist,cloud_indices.type(torch.FloatTensor).to(finalcenterlist.device),valid_indices.type(torch.FloatTensor).to(finalcenterlist.device)))
	# print("selecbox_corrs:", box_output.shape,box_output)
	# time3 = time.perf_counter()
	# print("boxcreate time:",time3-time2) #0.0016
	# print("gpu time:",time3-time0) #0.004
	# print("total time:",time3-time00) #0.333
	# del valid_mask,valid_delta,finalcenterlist,finalcloud,cloudmap,finalcenter
	return box_output

# if __name__=="__main__":
# 	mask = torch.rand(640,800)
# 	delta = torch.rand(640, 800, 2)
# 	time0 = time.perf_counter()
# 	result = IS2box(mask, delta)
# 	time1 = time.perf_counter()
# 	print(time1-time0)
































