import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
from detectron2.layers import cat
from .backbone import build_selection_backbone,build_att_backbone
from .headscreator import build_IShead,build_FLhead
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY,build_backbone
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
from .tools import IS_Visualizer,pyramid_Visualizer
from .targetgenerator import target_FL_generate
from .lossmaker import selec_criterion
from .selecpostprocess import sele_posprocess
from .util import pixel_perception,tonumpy,totensor
from . import selcuda
# from .maskcreate import maskcreate

__all__=["selection",]

@META_ARCH_REGISTRY.register()
class selection(nn.Module):
	"""
	create a overall model for selection which will contain following moduel:
	1.instance segmentation backbone and Focal Length backbone build;
	2.forword feedback fpn backbone;
	3.filting heads for 3 candidates;
	4.optional: small objects detection heads;
	"""
	@configurable
	def __init__(
			self,
			*,
			alpha,metadata,mask_th,num_class,
			input_format: Optional[str] = None,
			predata_en,atten,
			selec_onehot,num_FL,
			selecsensor,backbone_selec,
			pixel_mean: Tuple[float],
			pixel_std: Tuple[float],
			IShead: nn.Module,
			FLhead: nn.Module,
			FL_target,selec_loss,
			frozen,vis_period):
		super().__init__()
		self.alpha = alpha
		self.metadata = metadata
		self.mask_th = mask_th
		self.num_class = num_class
		self.input_format = input_format
		self.predata_en = predata_en
		self.atten = atten
		self.selec_onehot = selec_onehot
		self.num_FL = num_FL
		self.selecsensor = selecsensor
		self.IShead = IShead
		self.FLhead = FLhead
		self.FLtarget = FL_target
		self.backbone_selec = backbone_selec
		self.frozen = frozen
		self.vis_period = vis_period
		# if backbone_IS:
		# 	logger = logging.getLogger(__name__)
		# 	logger.info(f"initialize Instance backbone weights from{backbone_IS_pretrain}.")
		# 	self.backbone_IS.init_weights(backbone_IS_pretrain)
		self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
		self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
		self.selec_loss = selec_loss
		# print("DEVICE_TEST:", self.pixel_mean.device) #cpu
		if frozen:
			for mode in [self.IShead,self.backbone_selec]: #,self.IShead,self.backbone_selec]
				# print(mode)
				for (n,m) in mode.named_children():
					# print("name:",n)
					# if n in ["norm","last_delta_layer"]:
					# 	print(n)
					for param in m.parameters(recurse=True):
						param.requires_grad = False
						# print(param.shape)
						# print(param.requires_grad)
			# for layer in m.modules():
			# 	print("GRAD_TEST:")
			# 	for param in layer.parameters():
			# 		print(param.requires_grad)
			# 		param.requires_grad = False

	@classmethod
	def from_config(cls,cfg):
		# print(cfg.MODEL.BACKBONE.FREEZE_AT)
		# backbone_selection = build_att_backbone(cfg)
		backbone_selection = build_backbone(cfg)
		metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
		IShead = build_IShead(cfg,backbone_selection.output_shape())
		FL_target = target_FL_generate(cfg)
		FLhead = build_FLhead(cfg,backbone_selection.output_shape())
		selec_loss = selec_criterion(cfg)
		selecsensor = pixel_perception(cfg)
		return {
			"alpha":cfg.MODEL.SELECTION.ALPHA,
			"metadata":metadata,
			"mask_th":cfg.MODEL.PY.MASK_TH,
			"num_class":cfg.MODEL.SELECTION.NUM_CLASS,
			"input_format":cfg.INPUT.FORMAT,
			"predata_en": cfg.INPUT.PREDATASET.EN,
			"atten": cfg.MODEL.SELECBACKBONE.ENABLED,
			"selec_onehot":cfg.MODEL.SELECTION.ONEHOT,
			"num_FL": cfg.MODEL.GPU.FL,
			"selecsensor":selecsensor,
			"backbone_selec":backbone_selection,
			# "backbone_IS_pretrain": backbone_IS_pretrain,
			"pixel_mean": cfg.MODEL.PIXEL_MEAN,
			"pixel_std": cfg.MODEL.PIXEL_STD,
			"IShead": IShead,
			"FLhead": FLhead,
			"FL_target":FL_target,
			"selec_loss":selec_loss,
			"frozen":cfg.MODEL.SELECTION.STAGEFROZEN,
			"vis_period":cfg.VIS_PERIOD
		}

	@property
	def device(self):
		return self.pixel_mean.device

	def visualize_training(self,data,pred,val_ind,image_size_xy=None,finalbox=None,finalcls=None):
		# plot one same input pics twice meanwhile, invoking pyramid_Visualizer for pred outputs and recalling
		# detectron2 visualizer for gt annotation.
		# First pic has gt annotaion, second pic patches with pred boxes.
		# The demostration can be printed by tensorboard command or by opencv immediately.
		# This function refers to rcnn.py detectron2.
		vis_pred = dict()
		if not self.predata_en:
			img = data[0]["image"]
		else:
			img = data[0]["image"][0].permute(1, 2, 0)
		# print("original_img:",img.shape) #torch.Size([640, 853, 3])
		img = convert_image_to_rgb(img, self.input_format)
		if self.training:
			# storage = get_event_storage()
			# max_vis_prop = 20

			# print("testshape:",pred["mask"].shape) #[4, 1, 704, 1146]
			# print(val_ind)
			# In visualizer, parameters type should be set to numpy array which will be imposed  to do some calculations and draw pics.
			vis_pred["mask"] = tonumpy((pred["mask"][val_ind[0]].squeeze()).sigmoid())
			# print("mask check:",vis_pred["mask"].shape,vis_pred["mask"].dtype,type(vis_pred["mask"]))
			#int64 <class'numpy.ndarray'> (704, 1146)
			vis_pred["delta"] = tonumpy(pred["delta"][val_ind[1]]) #(704, 1146, 2)
			gt_ann = data[0]["selecIS_annotation"] #<class 'torch.Tensor'> ([704, 1146]) torch.Size([704, 1146, 2])
			target_fields = data[0]["instances"].get_fields()
			labels = [self.metadata.thing_classes[i] for i in target_fields["gt_classes"]]
			v_gt = IS_Visualizer(gt_ann,img,self.metadata,test=False)
			v_gt = v_gt.overlay_instances(labels=labels,boxes=target_fields.get("gt_boxes",None),masks=target_fields.get("gt_masks",None))
			anno_img = v_gt.get_image()
			# print("anno_img_shape:",anno_img.shape) #(704, 1146, 3)
		elif not self.training:
			vis_pred["mask"] = tonumpy((pred["mask"].squeeze()).sigmoid())
			vis_pred["delta"] = tonumpy(pred["delta"].squeeze())  # (704, 1146, 2)
		v_pred = pyramid_Visualizer(vis_pred,img,self.metadata,test=False,img_size=image_size_xy,mask_th=self.mask_th)
		v_pred = v_pred.overlay_instances(boxes=tonumpy(v_pred.predict_box))
		prop_img = v_pred.get_image()

		if finalbox is not None or finalcls is not None:
			assert isinstance(finalbox,np.ndarray) and isinstance(finalcls,np.ndarray), "predbox and predcls are not numpy."
			finalpredlabel = [self.metadata.thing_classes[i] for i in finalcls]
			vis_finalpred = Visualizer(img,None)
			vis_finalpred = vis_finalpred.overlay_instances(boxes=finalbox,labels=finalpredlabel)
			final_img = vis_finalpred.get_image()
			if self.training:
				vis_img = np.concatenate((anno_img,prop_img,final_img),axis=1)
			else:
				vis_img = np.concatenate((prop_img,final_img),axis=1)
		else:
			vis_img = np.concatenate((anno_img,prop_img),axis=1)
		# vis_img = prop_img
		# print("vis_img:",vis_img.shape)
		#.transpose(2,0,1) <class 'numpy.ndarray'> (704, 2292, 3) (704, 2292, 3)
		# vis_name = "Left:GT bounding boxes; Right: Predicted Proposals"
		# storage.put_image(vis_name,vis_img.transpose(2,0,1))
		# toPIL = transforms.ToPILImage()
		# vis_img = toPIL(vis_img)
		# plt.imshow(vis_img)
		# plt.show()
		imgid = str(data[0]["image_id"]) #436559 <class 'str'>
		# print("vis_image shape:",vis_img.shape)#(704, 2292, 3)
		cv2.imshow(imgid,vis_img[:,:,::-1])
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def forward(self,data:List[Dict[str, torch.Tensor]], train_stage: Optional[int]= 1):
		# print(data[0]["instances"].get_fields().keys()) #['file_name', 'height', 'width', 'image_id', 'image', 'instances', 'selecIS_annotation'] ['gt_boxes', 'gt_classes', 'gt_masks'] 393 640
		time00 = time.perf_counter()

		# if self.frozen:
			# for (name,mode) in self.named_children():
				# print("name",name)
				# if name == "backbone_selec":
				# 	for (n,m) in mode.named_children():
						# print("BBname:",n)
						# if n in ["norm","last_delta_layer"]:
						# for param in m.parameters():
						# 	print(param.requires_grad)
		if self.training:
			images,height,width,image_size_xy,gtmask,gtdelta,gtmaskbool = self.preprocess_image(data) #[4, 3, 704, 1152]
			time01 = time.perf_counter()
			print("prophase:", time01 - time00)


			# print("input shape:",images.tensor.shape)
			feature_IS = self.backbone_selec(images.tensor) #[4, 96, 176, 287] [4, 192, 88, 144] [4, 384, 44, 72] [4, 768, 22, 36]
			# print("BBfeature:",type(feature_IS)) #<class 'dict'>
			# for i in range(len(feature_IS)):
			# 	print([*feature_IS.keys()][i],[*feature_IS.values()][i].shape,[*feature_IS.values()][i].device)
			#p2 [4, 256, 176, 288] p3 [4, 256, 88, 144] p4 [4, 256, 44, 72]  p5 [4, 256, 22, 36] p6 [4, 256, 11, 18]
			if not self.atten:
				feature_IS = [*feature_IS.values()]
				ISheadout = self.IShead(feature_IS, height, width) #0.001360288995783776 torch.Size([4, 1, 704, 1146]) torch.Size([4, 704, 1146, 2]) cuda:0
			else:
				ISheadout = self.IShead(feature_IS, height,width)  # 0.001360288995783776 torch.Size([4, 1, 704, 1146]) torch.Size([4, 704, 1146, 2]) cuda:0
			# print("ISheadout:",ISheadout.keys(),ISheadout["mask"].shape) 3dict_keys(['mask', 'delta']) torch.Size([4, 1, 640, 850])
			time02 = time.perf_counter()
			print("metaphase:", time02 - time01)
			# # print("ishead:",type(ISheadout))
			predmask = ISheadout["mask"].squeeze(1).flatten(1) #[4, 806784]) cuda:0
			preddelta = ISheadout["delta"].flatten(1, 2) #[3227136, 2] [4, 806784, 2]

			preddelta = preddelta[gtmaskbool] #[870, 2]) [1160, 2] cuda:0 torch.cuda.FloatTensor
			preddelta = preddelta.view(self.num_FL+1,-1,2) ##[4, 290, 2]
			loss = dict()
			predlist = list()
			target = list()
			predlist[0:2] = [predmask, preddelta]
			target[0:2] = [gtmask, gtdelta]
			if train_stage == 0:
				val_ind,loss = self.selec_loss(self.num_FL+1, predlist, target, loss_stage=train_stage)
				# print("val_ind:",val_ind.shape,val_ind)
				time03 = time.perf_counter()
				print("anaphase:",time03-time02)
				# print("image_size_xy:",image_size_xy) # tensor([352., 573.], device='cuda:0')

				'''testing the pred value range'''
				valid_preddel = preddelta[val_ind[1]]
				# print("shape test:",valid_preddel.shape,preddelta.shape)
				real_valid_reddel = valid_preddel*image_size_xy
				preddelta_max = torch.max(valid_preddel)
				preddelta_min = torch.min(valid_preddel)
				print("preddelta_max&min:", preddelta_max, preddelta_min) #,valid_preddel[0:5],valid_preddel[0:5]*image_size_xy
				valid_gtdelta = gtdelta[0]
				real_valid_gtdelta = valid_gtdelta*image_size_xy

				real_diff_delta = real_valid_reddel - real_valid_gtdelta
				print("real_diff:", real_diff_delta.shape, real_diff_delta.max(), real_diff_delta.min())

			if train_stage==1:
				if "instances" in data[0]:
					gt_instances = [x["instances"].to(self.device) for x in data]
				else:
					gt_instances = None
				# print("gt_instance:",gt_instances[0].get_fields().keys()) #['gt_boxes', 'gt_classes', 'gt_masks']
				val_ind,loss_delta = self.selec_loss(self.num_FL + 1, predlist, target, loss_stage=None)
				print("loss_delta:",loss_delta.shape)
				# print("val_ind:",val_ind) #tensor(0, device='cuda:0')
				# time0 = time.perf_counter()

				predbox,cls_target = self.FLtarget(self.training,image_size_xy,ISheadout,val_ind,gt_instances,self.device) #list
				# print("iou:",iou.shape,iou.device) # torch.Size([437]) cuda:0
				# print("gt_boxes:",gt_boxes.shape) # [437, 4]
				# print(predbox.device,lv_target.device,cls_target.device)
				# time1 = time.perf_counter()
				# print("FLtarget:",time1-time0)

				FLheadout = self.FLhead(feature_IS,predbox=[predbox],FL_ind=val_ind)
				print("FL_level:",FLheadout["FL_level"].shape)
				print("classes:",FLheadout["classes"].shape) #[22, 81]

				predlist[2:4] = [FLheadout["FL_level"],FLheadout["classes"]]
				target[2:4] = [val_ind,cls_target]
				assert len(predlist)==len(target)==4,"predlist and target len is not equaled to 4."
				loss = self.selec_loss(self.num_FL+1, predlist, target, loss_stage=train_stage,frozen = self.frozen,loss_delta=loss_delta[val_ind])
				print("loss:",loss)
			'''	
				predlist[0:7] = [predmask, preddelta,FLheadout["FL_level"],FLheadout["classes"],predbox.tensor,iou]
				target[0:6] = [gtmask, gtdelta,lv_target,cls_target,gt_boxes]
			time04 = time.perf_counter()
			loss = self.selec_loss(self.num_FL,predlist,target,loss_stage=train_stage)
			time05 = time.perf_counter()
			print("loss part:",time05-time04) #0.25 0.000526 0.002385
			'''
			if self.vis_period>0:
				finalpredbox, finalpredcls = [None, None]
				storage = get_event_storage()
				if (storage.iter+1) % self.vis_period==0:
					if train_stage == 1:
						val_ind = predlist[2]
						finalpredbox,finalpredcls,val_ind = self.finalpredprocess(val_ind,image_size_xy,feature_IS,ISheadout,visual=True)
						val_ind = torch.stack([val_ind, val_ind])
					print("val_ind:",val_ind,val_ind.shape)
					self.visualize_training(data,ISheadout,val_ind,image_size_xy=image_size_xy,finalbox=finalpredbox,finalcls=finalpredcls)
			return loss
		if not self.training:
			return self.inference(data) #, height, width, image_size_xy)

	def inference(self,data:List[Dict[str, torch.Tensor]]): #,height,width,image_size_xy):
		assert not self.training,"Module is not in inference mode!"
		# print("Inference_data:", data[0]["image"].shape, data[0]["image"].device)
		img_orig, img_selec,height,width,image_size_xy = self.preprocess_image(data)
		print("preprocess dtype check:", img_orig.shape,img_orig.device, img_orig.dtype, img_selec.device,img_selec.type(),img_selec.shape)  # cuda:0
		origimg_bbfeature = self.backbone_selec(img_orig)
		# print("origimg_bbfeature:",type(origimg_bbfeature))
		pred_lv = self.FLhead(origimg_bbfeature)
		print("inference_pred_lv:",pred_lv.shape,pred_lv.device,pred_lv)
		pred_lv = torch.argmax(pred_lv)
		print(pred_lv,pred_lv.shape)
		img_selec = img_selec[pred_lv].unsqueeze(0)
		print("img_selec:",img_selec.shape,img_selec.device)
		imgselec_bbfeature = self.backbone_selec(img_selec)
		# for i in range(len(imgselec_bbfeature)):
		# 	print([*imgselec_bbfeature.keys()][i],[*imgselec_bbfeature.values()][i].shape,[*imgselec_bbfeature.values()][i].device)
		if not self.atten:
			imgselec_IS = [*imgselec_bbfeature.values()]
			ISheadout = self.IShead(imgselec_IS , height, width)
		else:
			ISheadout = self.IShead(imgselec_bbfeature, height, width)
		predbox = self.FLtarget(self.training,image_size_xy,ISheadout,device=self.device)
		print("predbox:", predbox.tensor.shape, predbox.tensor.device)
		predscores = self.FLhead.inference(imgselec_bbfeature,[predbox])
		print("predcls:",predscores.shape)
		predcls = torch.argmax(predscores,dim=1)
		print("final cls:",predcls.shape,predcls)
		index = (torch.arange(0, len(predcls)),predcls)
		predscores = predscores[index]
		print("scores:",predscores,type(predscores),predscores.shape,predscores.device)
		ind  = torch.where(predcls != self.num_class)[0]
		print(ind,ind.shape)
		predcls = predcls[ind]
		predscores = predscores[ind]
		predbox.tensor = predbox.tensor[ind]
		ori_h = data[0].get("height")
		ori_w = data[0].get("width")
		print(ori_h,ori_w,height,width)
		result = sele_posprocess(predbox, predcls, ori_h, ori_w, [float(height), float(width)],predscores)
		'''
		if self.premodes==1:
			feature_FL = self.backbone_IS(img_orig.tensor)
		else:
			feature_FL = self.backbone_FL(img_orig.tensor)
		# for keys, value in feature_FL.items():
		# 	print("FLfeaturecheck:",keys, value.shape)
		selecimg = self.FLhead.inference(feature_FL,img_selec.tensor,self.selec_onehot)
		# print("selecimg check:",selecimg.shape,selecimg.device)
		if self.premodes==0:
			feature_IS = self.backbone_FL(selecimg)
		else:
			feature_IS = self.backbone_IS(selecimg)
		# for i in range(len(feature_IS)):
			# print("feature_IS:",feature_IS[i].shape)
		# print("size:",height,width)
		ISheadout = self.IShead(feature_IS,height,width)
		# print(ISheadout["mask"].shape, ISheadout["delta"].shape, ISheadout["delta"].device)
		# print("image_size_xy:",image_size_xy,image_size_xy.dtype)
		predbox = self.FLtarget(self.training,image_size_xy,ISheadout,device=self.device)
		# print("predbox_check:",predbox.tensor.shape)
		pred_cls,scores = self.FLhead.predcls(feature_FL,[predbox])
		# print("classes_check:",pred_cls.shape)
		ori_h = data[0].get("height",img_selec.tensor.shape[0])
		ori_w = data[0].get("width",img_selec.tensor.shape[1])
		# print(ori_h,ori_w,img_selec.tensor.shape)
		result = sele_posprocess(predbox,pred_cls,scores,ori_h,ori_w,[float(height),float(width)])
		# print("result check:", len(result),type(result[0]),result[0]["instances"].get_fields().keys())
		'''
		if self.vis_period > 0:
			predbox = tonumpy(predbox.tensor)
			predcls = tonumpy(predcls)
			# ind = np.where(predcls!=self.num_class)
			# predcls = predcls[ind]
			# predbox = predbox[ind]
			print("final_box:",predbox.shape)
			pred_lv = torch.stack([pred_lv,pred_lv])
			self.visualize_training(data,ISheadout,pred_lv,image_size_xy=image_size_xy,finalbox=predbox,finalcls=predcls)
		return result

	@torch.no_grad()
	@torch.jit.unused
	def preprocess_image(self, data):
		"""
		Normalize, pad and batch the input images. Imagelist will add a dim to image's first position.
		"""
		assert len(data) == 1, "the batchsize is not 1."
		# print("data_id:",data[0]["image_id"])
		height,width = data[0]["input_shape"]
		# print("image size:",height,width)
		image_size_xy = torch.tensor([height, width], dtype=torch.float32, device=self.device)/self.alpha

		if not self.predata_en:
			images = [x["image"].to(self.device).type(torch.cuda.FloatTensor) for x in data] #[ 704, 1146,3]
			# for ind in images:
			# 	print("BGRRBGcheck:", ind.shape, ind[:, 100:120, 100])
			selecimg = self.selecsensor(images[0])#[3, 3, 704, 1146] 0.000132
			images = [images[0].permute(2,0,1).contiguous()]+list(torch.unbind(selecimg,dim=0))
		else:
			# print("predata check")
			images = [x.to(self.device).type(torch.cuda.FloatTensor) for x in data[0]["image"]]
		# for ind in images:
		# 	for a in range(3):
		# 		print("BGRRBGcheck:",ind.shape)
		# 		print(ind[a,100:120,100])
		# 	print("====================================")
		images = [(x - self.pixel_mean) / self.pixel_std for x in images]
		# print("orig image:",images[0].shape)#[3, 704, 1146]
		if not self.atten:
			images=ImageList.from_tensors(images,32)
		else:
			images=ImageList.from_tensors(images)
		# print("after image:",images.tensor.shape) #[4, 3, 704, 1152]
		if not self.training:
			return images.tensor[0].unsqueeze(0),images.tensor,height,width,image_size_xy
		if "selecIS_annotation" in data[0]:
			gt_selec = [x["selecIS_annotation"].to(self.device) for x in data]
			gt_selec_mask = [gt_selec[0].selecgt_mask] #[704, 1146]
			'''mask pred value range monitor'''
			# gtmaskmax = torch.max(gt_selec_mask[0])
			# gtmaxkmin = torch.min(gt_selec_mask[0])
			# print("gtmask max&min:",gtmaskmax,gtmaxkmin)
			gt_selec_delta = [gt_selec[0].selecgt_delta] #[806784, 2]
			gtmaskbool = gt_selec_mask[0].flatten()>0.5
			# print("size check:",gt_selec_delta[0].flatten(0,1).shape,type(gt_selec_delta[0])) #[806784, 2]) <class 'torch.Tensor'>
			gtdelta = gt_selec_delta[0].flatten(0,1)[gtmaskbool]
			# print("orignal gtdelta:",gtdelta[0:2])
			gtdelta = (gtdelta/image_size_xy).unsqueeze(0).expand(self.num_FL + 1, -1, -1) #[4, 290, 2]
			# test_gt_max = torch.max(gtdelta)
			# test_gt_min = torch.min(gtdelta)
			# print("test_gt_max&min:", test_gt_max, test_gt_min)
			gtmaskbool = gtmaskbool.unsqueeze(0).expand(self.num_FL + 1, -1).contiguous()#[4, 806784]
			gtmask = gt_selec_mask[0].flatten().unsqueeze(0).expand(self.num_FL + 1, -1).contiguous() #[4, 806784]
		else:
			raise Exception("gt_selec_mask,gt_selec_delta are empty!")
		return images,height,width,image_size_xy,gtmask,gtdelta,gtmaskbool

	@torch.jit.unused
	def finalpredprocess(self,predlv,image_size_xy,feature_IS,ISheadout,visual=False):
		# this module is for visualization
		print(predlv.shape) #([1, 4])
		finallv = torch.argmax(predlv)
		predbox = self.FLtarget(False,image_size_xy,ISheadout,val_ind=finallv,device=self.device)
		print("predbox:",predbox.tensor.shape,predbox.tensor.device)
		predcls = self.FLhead(feature_IS,predbox=[predbox],FL_ind=finallv,visual=visual)
		predbox = tonumpy(predbox.tensor)
		predcls = tonumpy(predcls)
		assert predbox.shape[0]==predcls.shape[0],"shape of predbox and predcls are not matched."
		finalpredcls=np.argmax(predcls,axis=1)
		ind = np.where(finalpredcls!=self.num_class)
		finalpredcls = finalpredcls[ind]
		finalpredbox = predbox[ind]
		print("finalpredbox:",finalpredbox.shape)
		return finalpredbox,finalpredcls,finallv


























