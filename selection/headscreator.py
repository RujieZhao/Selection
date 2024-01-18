import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from detectron2.config import configurable
from detectron2.structures import Boxes
from detectron2.layers import Conv2d,ConvTranspose2d
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
import fvcore.nn.weight_init as weight_init
# from mmcv.runner import auto_fp16
from detectron2.utils.registry import Registry
from collections import OrderedDict
from operator import itemgetter
__all__=["build_IShead","build_FLhead"]


IShead_REGISTRY = Registry("IS_HEAD")
IShead_REGISTRY.__doc__ ="""
return comprises center coordinate delta and mask. The size are [N,1,img_height,img_width] and [N,img_height,img_width,2] respectly.
"""

def build_IShead(cfg,input_shape:ShapeSpec):
	dim = []
	for i in [*input_shape.values()]:
		dim.append(i.channels) #[256, 256, 256, 256, 256]
		# print("test:",test)

	# dim = cfg.MODEL.IShead.dim
	layer = cfg.MODEL.IShead.layer
	start_level = cfg.MODEL.IShead.start_level
	name = cfg.MODEL.IShead.NAME
	return IShead_REGISTRY.get(name)(dim=dim,layer=layer,start_level=start_level)

@IShead_REGISTRY.register()
class IShead(nn.Module):
	"""
	Simple conv head, refer to FPN
	Aim to make IS mask.
	It follows 2 steps. First, resize the channel and h,w to output size([N,1,imgh,imgw]). Second, use cnn to create mask head and delta head.
	"""
	def __init__(self,dim=[96,192,384,768],layer=3,start_level=-1,):
		super(IShead, self).__init__()
		inch = int(dim[0]/16)
		# outch = int(inch*2)
		# self.act = nn.ELU() #nn.Sigmoid()
		self.neckact = nn.PReLU()
		self.norm = nn.InstanceNorm2d(inch)
		if layer == 1:
			# self.maskconv = self._get_rpn_conv(in_channels=inch,out_channels=outch,kernal=7,stride=1,padding=3)
			# self.deltaconv = self._get_rpn_conv(in_channels=inch,out_channels=outch,kernal=7,stride=1,padding=3)
			self.conv = self._get_rpn_conv(in_channels=inch,out_channels=inch,kernal=7,stride=1,padding=3,norm=self.norm)
		else:
			self.conv = nn.Sequential()
			# self.maskconv = nn.Sequential()
			# self.deltaconv = nn.Sequential()
			for i in range(layer):
				k = 3 if i == (layer-1) else 7
				p = 3 if k==7 else 1
				deltaconv = self._get_rpn_conv(inch,inch,kernal=k,stride=1,padding=p,norm=self.norm)
				# self.deltaconv.add_module(f"ISdeltaconv{i}",deltaconv)
				# maskconv = self._get_rpn_conv(inch,outch,kernal=k,stride=1,padding=p)
				# self.maskconv.add_module(f"ISmaskconv{i}",maskconv)
				self.conv.add_module(f"ISconv{i}",deltaconv)
				# inch = outch

		# self.last_mask_layer = nn.Conv2d(inch,1,1,1)
		# self.last_delta_layer = nn.Conv2d(inch,2,1,1)

		self.last_mask_layer = nn.Sequential(
			Conv2d(inch, inch, 3, 1, 1, norm=self.norm, activation=self.neckact),
			Conv2d(inch, inch, 3, 1, 1, norm=self.norm, activation=self.neckact),
			Conv2d(inch, inch, 3, 1, 1, norm=self.norm, activation=self.neckact),
			Conv2d(inch, 1, 1, 1))
		self.last_delta_layer = nn.Sequential(
			Conv2d(inch, inch, 3, 1, 1, norm=self.norm, activation=self.neckact),
			Conv2d(inch, inch, 3, 1, 1, norm=self.norm, activation=self.neckact),
			Conv2d(inch, inch, 3, 1, 1, norm=self.norm, activation=self.neckact),
			Conv2d(inch, 2, 1, 1))

		for l in self.conv:
			if isinstance(l,nn.Conv2d):
				# weight_init.c2_msra_fill(l)
				nn.init.normal_(l.weight, std=0.001)
				nn.init.constant_(l.bias, 0)

	def _get_rpn_conv(self, in_channels, out_channels,*,kernal=3,stride=1,padding=1,norm=None):
		return Conv2d(
			in_channels,
			out_channels,
			kernel_size=kernal,
			stride=stride,
			padding=padding,
			activation=None, #nn.ReLU()nn.ELU()nn.Tanh()
			norm=norm
		)

	# @auto_fp16()
	def forward(self,x,img_height,img_width):
		"""
		The shape of x should be [[3,96,h,w],[3,192,h/2,w/2],[3,384,h/4,w/4],[3,768,h/8,w/8]] or p2 [4, 256, 176, 288] p3 [4, 256, 88, 144] p4 [4, 256, 44, 72]  p5 [4, 256, 22, 36] p6 [4, 256, 11, 18]
		"""
		# print("input feature:",x[0].shape) #[4, 256, 176, 288]
		# assert len(x) == len(self.dim)
		N = x[0].shape[0]
		height = x[0].shape[2]
		width = x[0].shape[3]
		# for i in range (len(self.dim[:self.start_level])):
		# 	lateral = self.lateral[i](x[self.start_level-i])
		# 	upsample = self.upsample[i](lateral)
		# 	if not x[self.start_level - i - 1].shape == upsample.shape:
		# 		upsample = F.interpolate(upsample,x[self.start_level-1-i].shape[2:],mode="nearest")
		# 	assert x[self.start_level - i - 1].shape == upsample.shape
		# 	x[self.start_level-1-i] += upsample
		output = x[0].view(N,-1,height*4,width*4) #[4, 16, 704, 1152]
		# mask_output = self.last_mask_layer(output)
		# output = self.head(output) #[N,1,h,w]
		if not output.shape[2]*output.shape[3] == img_height*img_width:
			size = [img_height,img_width]
			#output size is [3,6,img_h,img_w]
			output = F.interpolate(output,size=size,mode="nearest")
		# print(output.shape) #[4, 1, 704, 1146]
		output = self.neckact(self.norm(output))
		# build mask net
		# mask_output = F.relu(output) #torch.Size([3, 1, 853, 640])
		# mask_output = self.maskconv(output) #torch.Size([3, 1, 853, 640])
		# mask_output = F.relu(self.norm2(mask_output))
		mask_output = self.last_mask_layer(output) #[4, 1, 704, 1146]
		# print("mask_output:",mask_output.shape)

		# build delta output net
		# delta_output = self.deltaconv(output)
		# delta_output = F.relu(self.norm2(delta_output))
		delta_output = self.last_delta_layer(output)
		delt_output = torch.permute(delta_output,(0,2,3,1)) #[4, 704, 1146, 2]
		# print("delt_output:",delt_output.shape)
		#build delt net
		# for i,layer in enumerate(self.delt):
		# 	delt_output = F.relu(layer(delt_output)) if i< len(self.delt)-1 else layer(delt_output)
		#mask and delta do not have act func and softmax on last layer
		output = {}
		output["mask"] = mask_output
		# output["delta"] = self.act(delt_output)
		output["delta"] = delt_output
		# print("mask_out_check:",mask_output.shape,mask_output.device,mask_output.type(),mask_output.is_contiguous())
		# print(output["mask"].shape,output["delta"].shape)
		return output

FLhead_REGISTRY = Registry("FL_HEAD")
FLhead_REGISTRY.__doc__ ="""
By default, resnet is used as our FL backbone and we will use "res4" as our FL feature. The return's shape is [bs,3] that indicates which focal length index will be used to meet the par of gt_box.
"""
@FLhead_REGISTRY.register()
class FLhead(nn.Module):

	def __init__(self,cfg,scale_name,input_shape,num_class,pooler_resolution,FL_num=4):
		"""
		Args:
			FL_num: the number of focal length;
		Return:
			output['FL'] is the probability for each FL with the shape of [Bs,3].
			output['class'] is the probaliblity for classifcation with the shape of [BS,N,num_classes]. Here the N represents the pred boxes num from IS mode. In this part, the ROIPooler function from detectron2 has been used.
		"""
		super().__init__()
		# print("flhead_inputchannel:",input_shape[scale_name[-1]].channels) #256
		in_channel =input_shape[scale_name[-1]].channels
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.linear = nn.Sequential(
			nn.Linear(in_channel,int(in_channel/2)),
			nn.Linear(int(in_channel/2),FL_num),
		)
		self.act = nn.ReLU() #nn.GELU()
		self.softmax = nn.Softmax(dim=-1)
		self.norm = nn.InstanceNorm2d(1024)
		# for param in self.linear:
		# 	nn.init.kaiming_normal_(param.weight,mode="fan_out",nonlinearity="relu")
		# 	nn.init.constant_(param.bias, 0)

		self.box_pooler = ROIPooler(
			output_size=pooler_resolution,
			scales=tuple(1.0 / input_shape[k].stride for k in scale_name),
			sampling_ratio=0,
			pooler_type="ROIAlignV2"
		)
		self.box_head = build_box_head(cfg, ShapeSpec(channels=in_channel, height=pooler_resolution, width=pooler_resolution))
		# self.cls_norm = nn.LayerNorm(1024)#nn.BatchNorm2d(1)
		# self.cls_avgpool = nn.AdaptiveAvgPool2d(1)
		# print("FRCNN head channels:",self.box_head.output_shape.channels) #1024
		# print("num_class:",num_class) #80
		self.cls_head = nn.Linear(self.box_head.output_shape.channels, num_class+1)
		# nn.init.kaiming_normal_(self.cls_head.weight,mode="fan_out",nonlinearity="relu")
		# nn.init.normal_(self.cls_head.weight,std=0.01)
		# nn.init.constant_(self.cls_head.bias,0)

		for layer in self.modules():
			if isinstance(layer,nn.Linear):
				nn.init.normal_(layer.weight,std=0.01)
				nn.init.constant_(layer.bias, 0)

	#TODO args is a temperiary name. I will reset it after pred_box being created.
	def forward(self,x,predbox: List[Boxes]= None,FL_ind = None,visual=False):
		"""
		predbox: It has to be a list containing all boxes.
		x: It is a dict or list collection from FL backbone layers.Usually the len is 4.
		predbox: It is a list from IS_boxcreator moddule. In training, it has been leached out with gt_box, so there is no negative box and class imbalance issue along with it. In inference, it is purely outputs list as predicted boxes.

		return: Total 2 outputs for FL heads. First is to decide the FL levels with the shape of [bs,3]. In this branch, we use last layer of feature map("res5") as input.
		Second is to decide the class corresponding to related box from IS mode and the output shape should be [bs,N,num_class]. The methods to pick the input feature is adopted from MaskRcnn.
		Here N is the total predicted boxs in fixed order, which can be generated from multiple mode or one-hot mode. In addition, num_class is not including background.
		"""
		print("FLhead_training:",self.training)
		if self.training:
			output={}
			for i in range(len(x)):
				# print("feature_check:",type(x),x[i].shape)
				x[i] = self.act(self.norm(x[i]))
			# x = self.act(self.norm(x))
			FL_feature,cl_feature = self.feature_split(x,FL_ind)
			# print("FL_feature:",FL_feature.shape)#[1, 256, 13, 21]
			# print("cl_feature:",type(cl_feature),len(cl_feature),cl_feature[-1].shape)
			if not visual:
				# Part1: FL output
				output["FL_level"] =self.lv_generate(FL_feature)
				# print("fllv dim:",output["FL_level"].dim())#[1,3]

				# Part2: class output
				output["classes"] = self.cls_creator(cl_feature,predbox)
				return output
			else:
				return self.cls_creator(cl_feature,predbox)
		else:
			x = self.feature_split(x,outputtype="lv")
			x = self.act(self.norm(x))
			return self.softmax(self.lv_generate(x))

	def feature_split(self,feature,FL_ind=None,outputtype:str=None):
		if type(feature) is list:
			if outputtype == "cl":
				cl_feature = list(map(lambda x:x[FL_ind].unsqueeze(0),feature))
				return cl_feature
			elif outputtype == "lv":
				FL_feature = feature[-1][0].unsqueeze(0)
				return FL_feature
			else:
				cl_feature = list(map(lambda x: x[FL_ind].unsqueeze(0), feature))
				FL_feature = feature[-1][0].unsqueeze(0)
		elif type(feature) is dict:
			if outputtype == "cl":
				cl_feature = list(map(lambda x:x[FL_ind].unsqueeze(0),[*feature.values()]))
				return cl_feature
			elif outputtype == "lv":
				FL_feature = [*feature.values()][-1][0].unsqueeze(0)
				return FL_feature
			else:
				cl_feature = list(map(lambda x: x[FL_ind].unsqueeze(0), [*feature.values()]))
				FL_feature = [*feature.values()][-1][0].unsqueeze(0)
			# cl_feature = list(map(itemgetter(1),feature.items()))
		else:
			raise TypeError("input feature must be either a dict or a list type.")
		return FL_feature,cl_feature

	def lv_generate(self,FL_feature):
		FL_level = self.avgpool(FL_feature)
		FL_level = torch.flatten(FL_level, 1)
		FL_level = self.linear(FL_level)
		#if hot is ture below line should be commented
		# FL_level = FL_level.sigmoid()
		return FL_level

	def cls_creator(self,cl_feature,predbox):
		cls_feature = self.box_pooler(cl_feature,predbox)
		# print("cls_feature:",cls_feature.shape) #torch.Size([14391, 256, 7, 7])
		cls_feature = self.box_head(cls_feature)
		# print("cls_feature:",cls_feature.shape)#[14391,1024]
		if cls_feature.dim()>2:
			cls_feature = torch.flatten(cls_feature,start_dim=1)
		# print("cls_feature:", cls_feature.shape)#[14391,1024]
		cls_head = self.cls_head(cls_feature)
		del cls_feature
		return cls_head

	def inference(self,feature,predbox: List[Boxes]= None):
		feature = self.feature_split(feature,FL_ind=0,outputtype="cl")
		for i in range(len(feature)):
			# print("feature_check:",type(feature),feature[i].shape)
			feature[i] = self.act(self.norm(feature[i]))
		return self.softmax(self.cls_creator(feature,predbox))

		# FL_feature,_ = self.feature_split(feature)
		# FL_level = self.lv_generate(FL_feature).squeeze(0)
		# print(FL_level)
		# lv_ind = torch.where(FL_level>0.5)[0]
		# if one_hot and len(lv_ind)>1:
		# 	lv_ind_val = lv_ind[torch.argmax(FL_level[lv_ind])]
		# 	lv_ind = lv_ind[lv_ind_val]
		# 	img_selec = img_selec[lv_ind].unsqueeze(0)
		# elif len(lv_ind)==0:
		# 	raise ValueError("level is empty in Inference.")
		# img_selec = img_selec[lv_ind]
		# print(img_selec.shape)
		# del lv_ind
		# return img_selec

	def predcls(self,feature,predbox:list[Boxes]):
		_,cls_feature = self.feature_split(feature)
		pred_cls = self.cls_creator(cls_feature,predbox)
		pred_cls_score = F.softmax(pred_cls,dim=-1)[:,:-1]
		# print("pred_cls_check:",pred_cls_score.shape)
		pred_cls = torch.argmax(pred_cls_score,dim=-1)
		return pred_cls,pred_cls_score

def build_FLhead(cfg,input_shape=None):
	# print("test_keys:",[*input_shape])
	FL_num = cfg.MODEL.FLhead.FL_NUM
	name = cfg.MODEL.FLhead.NAME
	num_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES
	pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
	# if cfg.MODEL.BACKBONE.FL_EN or cfg.MODEL.SELECBACKBONE.ENABLED:
	# print("backbone enable:",cfg.MODEL.SELECBACKBONE.ENABLED)
	if not cfg.MODEL.SELECBACKBONE.ENABLED:
		scalesname = [*input_shape]
		# print("sclaesname:",scalesname) #['p2', 'p3', 'p4', 'p5', 'p6']
	else:
		scalesname = [0,1,2,3]
	# print("scalesname:", scalesname)
	# print("inputshape:", input_shape)
	return FLhead_REGISTRY.get(name)(cfg,scalesname,input_shape,num_class,pooler_resolution,FL_num=FL_num)

# class_head_REGISTRY = Registry("classification")
# class_head_REGISTRY.__doc__ ='''For classification head, we inherit maskrcnn and DETR. In addition, the shared feature map will be obtained from FL backbone.'''

# @class_head_REGISTRY.register()
# class classification:


# if __name__ == "__main__":
# 	device = "cuda:0" if torch.cuda.is_available() else "CPU"
# 	a = torch.rand((3,96, 214, 160)).to(device)
# 	b = torch.rand((3, 192, 107, 80)).to(device)
# 	c = torch.rand((3,384, 54, 40)).to(device)
# 	d = torch.rand((3,768, 27, 20)).to(device)
# 	e = [a,b,c,d]
# 	print(len(e))
# 	net = IShead(start_level=-1).to(device)
# 	print(net)
# 	output = net(e,img_height=853,img_width=640)
# 	for keys in output.keys():
# 		print(output[keys].shape)






























