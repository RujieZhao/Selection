import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.layers import cross_entropy
from detectron2.structures import Instances
from fvcore.nn import giou_loss, smooth_l1_loss
from .util import giou_generate

__all__=["selec_criterion"]

class selec_criterion(nn.Module):
	"""
	get 3 parts of loss.
	Loss 1: Instance Segmentation
	Loss 2: focal length among 1,2,3
	Loss 3: box classification
	"""
	@configurable()
	def __init__(self,weight_dict:dict,num_class,selec_onehot,GIOU_LOSS,BOX_LOSS,vis_period):
		super().__init__()
		self.weight_dict = weight_dict
		self.selec_onehot = selec_onehot
		self.GIOU_LOSS = GIOU_LOSS
		self.BOX_LOSS = BOX_LOSS
		self.num_classes = num_class
		self.vis_period = vis_period
		empty_weight = torch.ones(self.num_classes+1)
		self.eos_coef = empty_weight[-1]
		self.register_buffer("eimpty_weight",empty_weight)

	@classmethod
	def from_config(cls,cfg):
		weight_dict = {
			"mask":  cfg.MODEL.SELECTION.MASK_WEIGHT,
			"delta": cfg.MODEL.SELECTION.DELTA_WEIGHT,
			"lv":    cfg.MODEL.SELECTION.LV_WEIGHT,
			"cls":   cfg.MODEL.SELECTION.CLS_WEIGHT,
			"giou":  cfg.MODEL.SELECTION.GIOU_WEIGHT}
		ret = {
			"weight_dict": weight_dict,
			"num_class": cfg.MODEL.SELECTION.NUM_CLASS,
			"selec_onehot": cfg.MODEL.SELECTION.ONEHOT,
			"GIOU_LOSS": cfg.MODEL.SELECTION.GIOU_LOSS,
			"BOX_LOSS": cfg.MODEL.SELECTION.BOX_LOSS,
			"vis_period": cfg.VIS_PERIOD}
		return ret

	def forward(self,num_lv,selec_pred,selec_gt,loss_stage=1,frozen=False,loss_delta=None):
		"""
		selec_pred: mask,delta,lv,cls,box
		selec_gt: mask,delta,lv,cls,gt_box,gt_iou
		loss_stage: training stage 1 and 2
		retur: total loss
		"""
		# time00=time.perf_counter()
		#all of pred and gt should be in cuda
		loss={}
		val_ind = torch.tensor(0)
		if loss_stage==0:
			assert len(selec_pred)==len(selec_gt)==2,"input len are not correct in stage 1."
			pred_mask, pred_delta = selec_pred
			gt_mask, gt_delta = selec_gt

			# t0=time.perf_counter()
			loss_delta = self.loss_delta(pred_delta, gt_delta) #val_ind,
			val_delta_ind = torch.argmin(loss_delta)
			print("val_delta_ind:",val_delta_ind)
			# loss.update(loss_delta)
			# t1 = time.perf_counter()
			# print("loss time:",t1-t0)
			# pred_mask = pred_mask[val_ind]
			# gt_mask = gt_mask[val_ind]
			loss_mask = self.loss_mask(pred_mask,gt_mask,vis_period=self.vis_period)
			# val_mask_ind = torch.argmin(loss_mask)
			# print("val_mask_ind:",val_mask_ind)
			# loss_sum = loss_delta+loss_mask
			# val_sum_ind = torch.argmin(loss_sum)
			# print("loss_sum:",loss_sum,val_sum_ind)
			val_ind = torch.stack([val_delta_ind,val_delta_ind])
			loss["loss_mask"] = loss_mask[val_ind[0]] * self.weight_dict["mask"]
			loss["loss_delta"] = loss_delta[val_ind[1]] * self.weight_dict["delta"]
			return val_ind, loss
		elif loss_stage is None:
			assert len(selec_pred) == 2 and len(selec_gt) == 2, "input len are not correct in stage 2."
			pred_mask, pred_delta = selec_pred #, pred_lv, pred_cls, pre_box, iou
			gt_mask, gt_delta= selec_gt #, gt_lv, gt_cls, gt_box
			loss_delta = self.loss_delta(pred_delta, gt_delta)
			val_ind = torch.argmin(loss_delta)
			return val_ind, loss_delta
		elif loss_stage ==1:
			pred_mask, pred_delta,pred_lv,pred_cls = selec_pred
			gt_mask, gt_delta,gt_lv,gt_cls = selec_gt
			assert len(selec_pred) == len(selec_gt) == 4, "input len are not correct in stage 1."
			loss_lv,loss_cls = self.loss_lv_cls(pred_lv,gt_lv,pred_cls,gt_cls)
			loss["loss_lv"] = loss_lv*self.weight_dict["lv"]
			loss["loss_cls"] = loss_cls*self.weight_dict["cls"]
			# if ture only lv and cls loss will be calculated. otherwise the total loss will add mask and delta.
			if not frozen:
				assert loss_delta is not None, "loss_delta should not be None under frozen."
				loss_mask = self.loss_mask(pred_mask, gt_mask, vis_period=self.vis_period)
				print("stage1_lossmask:",loss_mask.shape,loss_delta.shape,loss_delta)
				loss["loss_mask"] = loss_mask[val_ind] * self.weight_dict["mask"]
				loss["loss_delta"] = loss_delta * self.weight_dict["delta"]
			# loss_mask = self.loss_mask(pred_mask, gt_mask, vis_period=self.vis_period)
			# gt_lv = gt_lv.to(dtype=torch.int64)
			# loss.update(self.loss_mask(num_lv,pred_mask,gt_mask,vis_period=self.vis_period))
			# loss.update(self.loss_delta(num_lv,pred_delta,gt_delta))
			# time0 = time.perf_counter()
			# loss.update(self.loss_lv_cls(pred_lv,gt_lv,pred_cls,gt_cls))
			# time1 = time.perf_counter()
			# print("lvcls loss time:",time1-time0)
			# if self.GIOU_LOSS:
			# 	loss.update(self.loss_giou(iou,pre_box,gt_box))
			# 	time2 = time.perf_counter()
			# 	print("giou loss:",time2-time1)
			return loss
		else:
			raise ValueError("loss_stage is not a valid number.")
			# return val_ind,loss

	@torch.jit.unused
	def loss_mask(self,inputs:torch.Tensor,targets:torch.Tensor,alpha:float=0.25,gamma:float=2,vis_period:int=0):
		# print("mask shape:",inputs.shape,targets.shape) #[806784]

		prob = inputs.sigmoid()
		assert prob.shape == targets.shape
		weight = torch.empty(targets.shape).to(targets)
		# print("weight:",weight.shape) #torch.Size([4, 806784])
		pos_num = targets[0].sum()
		total_num = targets[0].numel()
		weight[targets==1] = (total_num-pos_num)/total_num
		weight[targets==0] = (pos_num/total_num)
		# weight[targets == 0] = torch.clamp((pos_num / total_num), min=0.001)
		loss = F.binary_cross_entropy_with_logits(inputs, targets,weight=weight, reduction="none")
		p_t = prob*targets+(1-prob)*(1-targets)
		loss = (loss*((1-p_t)**gamma)).sum(1)
		# if alpha >= 0:
		# 	alpha_t = alpha*targets+(1-alpha)*(1-targets)
		# 	loss = alpha_t*loss


		# Attemp to apply smooth_l1 for mask loss calculation. And considering imbalance issue, add ratio parameter is optional.
		# input_pro = inputs.sigmoid()
		# loss = smooth_l1_loss(inputs, targets,beta=0,reduction="none")
		print("mask loss:",loss.shape,loss) #torch.Size([4, 545920])
		# loss = (loss*weight*((1-p_t)**gamma)).sum(1)

		print("mask total length&pos num:",total_num,pos_num)
		test = (prob[:,targets[0]==1]>0.5).sum(1)
		# print("test:",test.shape)
		print("correct mask num:",test,end="  ")
		print("total pred:", (prob>0.5).sum(1))
		new_rate = 0.85
		test_high = (prob[:, targets[0] == 1] > new_rate).sum(1)
		print("new correct mask num:", test_high, end="  ")
		print("new total pred:", (prob > new_rate).sum(1))


		#TODO: write the log to tensorboard.
		# if gt_mask.dtype == torch.bool:
		# 	targets_bool = gt_mask[0]
		# else:
		# 	targets_bool = gt_mask[0] > 0.5
		# num_pos = targets_bool.sum().item()
		# for id_lv in range(num_lv):
		# 	mask_incorrect = prob[id_lv] != targets_bool
		# 	mask_acc = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
		# 	# false_pos is the wrong bit in pred is true.
		# 	false_pos = (mask_incorrect & ~targets_bool).sum().item() / max(targets_bool.numel() - num_pos, 1.0)
		# 	false_neg = (mask_incorrect & targets_bool).sum().item() / max(num_pos, 1.0)
		# 	storage = get_event_storage()
		# 	storage.put_scalar("selection/accuracy_lv" + str(id_lv), mask_acc)
		# 	storage.put_scalar("selection/false_pos" + str(id_lv), false_pos)
		# 	storage.put_scalar("selection/false_neg" + str(id_lv), false_neg)
		# 	#Maybe I need make some change for showing the picture more nicely.
		# 	if vis_period > 0 and storage.iter % vis_period == 0:
		# 		vis_masks = torch.cat([prob[id_lv], gt_mask[id_lv]], dim=1)
		# 		vis_masks = torch.stack([vis_masks] * 3, dim=0)
		# 		name = "Left: mask prediction;   Right: mask GT lv" + str(id_lv)
		# 		storage.put_image(name, vis_masks)
		# t2 = time.perf_counter()
		# print("post mask loss:",t2-t1,t2-self.t)
		# return {"loss_mask":(loss.sum(1))*self.weight_dict["mask"]}
		return loss

	@torch.jit.unused
	def loss_delta(self,pred_delta,gt_delta):
		'''
		gt_delta is corresponding to [height,width].
		Hence, image_size should match delta(height,width).
		'''
		# print("size_check:",pred_delta.shape,gt_delta.shape)#([4, 290, 2]) torch.Size([4, 290, 2])
		assert pred_delta.shape==gt_delta.shape,"delta size is not matched with gt!"
		# loss_delta = F.l1_loss(pred_delta, gt_delta, reduction="none")
		loss_delta = smooth_l1_loss(pred_delta,gt_delta,beta=1e-2,reduction="none") #2e-3
		loss_delta = loss_delta.sum(dim=(-1)).mean(-1) #4
		# ind = torch.argmin(loss_delta)
		# print("ind:",ind)
		# ind = torch.tensor([0])
		print("loss delta shape:",loss_delta) #,ind
		# loss={"loss_delta":loss_delta[ind]*self.weight_dict["delta"]}
		return loss_delta

	@torch.jit.unused
	def loss_lv_cls(self,pred_lv,gt_lv,pred_cls:torch.Tensor,cls_target:torch.Tensor):
		"""
		when lv is not one hot, lv_loss calculation will become a multi-label classification.
		So binary cross entropy is used to obtain lv_loss and normal cross entropy is used for cls_loss.
		"""
		# print("devicecheck:",pred_lv.device,gt_lv.device,pred_lv.is_cuda)
		assert pred_lv.is_cuda and gt_lv.is_cuda, "pred and gt are not cuda."
		pred_lv = pred_lv.squeeze(0)
		if self.selec_onehot:
			#one-to-one
			print("selec_onehot:",pred_lv,gt_lv,cls_target,pred_cls.shape,cls_target.shape)
			# there is no sigmoid for lv head output.
			loss_lv = cross_entropy(pred_lv,gt_lv,reduction="mean")
			loss_cls = cross_entropy(pred_cls,cls_target,reduction="mean")
		# gtlv = torch.zeros(pred_lv.shape,device=pred_lv.device)
		# gtlv[gt_lv]=1
		# num_class=cls_target.numel()
		# pred_classes = pred_cls.argmax(dim=1)
		# bg_class_ind = pred_cls.shape[1]-1
		# fg_inds = (cls_target>=0)&(cls_target<bg_class_ind) #len=cls_target
		# num_fg = fg_inds.nonzero().numel()
		# fg_gt_classes = cls_target[fg_inds]
		# fg_pred_classes = pred_classes[fg_inds]
		#
		# num_false_negative = (fg_pred_classes==bg_class_ind).nonzero().numel()
		# num_accurate = (pred_classes==cls_target).nonzero().numel()
		# fg_num_accurate = (fg_pred_classes==fg_gt_classes).nonzero().numel()
		#
		# storage = get_event_storage()
		# storage.put_scalar(f"cls_accuracy",num_accurate/num_class)
		# if num_fg > 0:
		# 	storage.put_scalar(f"fg_cls_accuracy",fg_num_accurate/num_fg)
		# 	storage.put_scalar(f"false_negative",num_false_negative/num_fg)
		# # print("cross_entropy:",F.cross_entropy(pred_cls,cls_target,reduction="mean")*self.weight_dict["cls"])
		# loss = {"loss_cls": cross_entropy(pred_cls,cls_target)*self.weight_dict["cls"]}
		# loss["loss_lv"] =F.binary_cross_entropy_with_logits(pred_lv,gtlv)*self.weight_dict["lv"]
			return loss_lv,loss_cls

	# @torch.jit.unused
	# def loss_giou(self,iou:torch.Tensor,predbox,gtbox):
	# 	loss={"loss_iou": torch.mean(1 - giou_generate(iou,predbox,gtbox))*self.weight_dict["giou"]}
	# 	return loss






































