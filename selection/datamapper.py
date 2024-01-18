#! /usr/bin/env python3

import copy
import torch
from detectron2.data import detection_utils as utils
import numpy as np
from detectron2.data import DatasetMapper
from detectron2.config import configurable
import detectron2.data.transforms as T
from detectron2.data.transforms.augmentation import _get_aug_input_args
from detectron2.structures import Instances
from .targetgenerator import transform_instance_annotations,target_IS_generate
# from .util import pixel_perception
# from . import selcuda

__all__ = ["selecmapper"]

class selecmapper(DatasetMapper):

	@configurable
	def __init__(self,*,predata_dir,predata_en,**kwargs):
		super().__init__(**kwargs)
		self.predata_dir = predata_dir
		# print(self.predata_dir)
		self.predata_en = predata_en
		# self.selecsensor = selecsensor
	@classmethod
	def from_config(cls,cfg, is_train: bool = True):
		ret = super().from_config(cfg,is_train)
		# selecsensor = pixel_perception(cfg)
		ret["predata_dir"] = cfg.INPUT.PREDATASET.DIR
		ret["predata_en"] = cfg.INPUT.PREDATASET.EN
		# ret["selecsensor"] = selecsensor
		return ret

	def __call__(self, dataset_dict):
		"""
		just rewrite the original version of class datamapper, mainly set the path for the prepared data set and add an 'EN' value to control the requirement of the image augmentation operation. And for the rest of the code, it will keep as the same as the original version.
		"""
		# print("my mapper")
		dataset_dict = copy.deepcopy(dataset_dict)
		# print("img_name:",dataset_dict["file_name"],dataset_dict["height"],dataset_dict["width"])
		#/mnt/ssd1/rujie/pytorch/dataset/coco/train2017/000000451949.jpg dict_keys(['file_name', 'height', 'width', 'image_id', 'annotations']) 393 640
		if self.predata_en:
			image_orig = utils.read_image(dataset_dict["file_name"], format=self.image_format)
			utils.check_image_size(dataset_dict, image_orig)
			aug_input = T.AugInput(image_orig)
			# args = _get_aug_input_args(aug_input,)
			transforms = self.augmentations(aug_input)
			new_image_orig = aug_input.image
			new_image_orig_shape = new_image_orig.shape[:2]
			# print("new_image_orig:",new_image_orig.shape,new_image_orig_shape) #(640, 850, 3) (640, 850)
			image_shape = tuple()
			pre_image = []
			img_name = dataset_dict["file_name"].split("/")[-1].split(".")[0]
			# print("img_name:",img_name) #/mnt/ssd1/rujie/pytorch/dataset/coco/train2017/000000451949.jpg
			img_dir = self.predata_dir+ img_name+"/"
			# print("img_dir:",img_dir)
			# print("mapper_filename:", dataset_dict["file_name"].split("/")[-1].split(".")[0])#000000481791
			if "sem_seg_file_name" in dataset_dict:
				raise "sem_seg_file_name should not show in instance objection."

			for i in range(4):
				image_path = img_dir+img_name+"#"+str(i)+".npy"   #".jpg"
				# print("image_path:",image_path)
				image = np.load(image_path)
				# print("datamapper:",image.shape)
				'''
				#Do not need to extral augmentation processing for selecuda
				image = utils.read_image(image_path,format=self.image_format)
				# print("image_format",self.image_format) #BGR
				# if i==0:
				# 	for a in range(3):
				# 		print("imagesizeTrue:",image.shape,type(image),image[100:120,100,a]) #(480, 640, 3) <class 'numpy.ndarray'>
				# else:
				# 	print("selecudashape:",image.shape,image[100:120,100,a])
				utils.check_image_size(dataset_dict, image)
				aug_input = T.AugInput(image)
				# print("augmentation:",self.augmentations)
				transforms = self.augmentations(aug_input)
				# print("transorms:",transforms)
				image = aug_input.image#numpy.adarray
				# print("newimagesize:",image.shape) # (640, 853, 3)
				pre_image.append(torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))))
				'''
				# aug_input = T.AugInput(image)
				# transforms = self.augmentations(aug_input)
				# print("transorms:",transforms)
				pre_image.append(torch.from_numpy(np.array(image,copy=True)))
				if i ==0:
					image_shape = image.shape[1:3]
					# print("image_shape_check:",image.shape,image_shape,type(image_shape)) #(800, 1303) <class 'tuple'>
					assert new_image_orig_shape==image_shape, "Prepared data shape is not matched with augmented raw data shape."
					dataset_dict["input_shape"] = torch.as_tensor(image_shape)
			dataset_dict["image"]=pre_image
		else:
			image = utils.read_image(dataset_dict["file_name"],format=self.image_format)
			# print(self.image_format) #BGR
			# for a in range(3):
			# 	print("imagesizeFalse:",image.shape,image[100:120,100,a]) #(393, 640, 3)
			utils.check_image_size(dataset_dict,image)
			aug_input = T.AugInput(image)
			transforms = self.augmentations(aug_input)
			image = aug_input.image
			image_shape = image.shape[:2]
			# dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1)))
			# print(type(image)) #<class 'numpy.ndarray'>
			dataset_dict["image"] = torch.from_numpy(np.array(image,copy=True))
			# dataset_dict["image"] = torch.as_tensor(image,dtype = torch.float32)
			# print(dataset_dict["image"].dtype,dataset_dict["image"].shape) #torch.uint8 torch.Size([704, 1146, 3])
			dataset_dict["input_shape"] = torch.as_tensor(image_shape)
		if not self.is_train:
			dataset_dict.pop("annotations",None)
			dataset_dict.pop("sem_seg_file_name",None)
		if "annotations" in dataset_dict:
			# print("start_transform_annotation",i)
			self._transform_annotations(dataset_dict, transforms, image_shape)
		# for i in range(3):
		# 	print(dataset_dict["image"][100:120,100,i])
		return dataset_dict
		# else:
		# 	return self.inference_data(dataset_dict)

	def _transform_annotations(self, dataset_dict, transforms, image_shape):
		for anno in dataset_dict["annotations"]:
			# print("anno:",anno)
			if not self.use_instance_mask:
				anno.pop("segmentation", None)
			if not self.use_keypoint:
				anno.pop("keypoints", None)

		annos = [
			transform_instance_annotations(
				obj, transforms, image_shape
			)
			for obj in dataset_dict.pop("annotations")
			if obj.get("iscrowd", 0) == 0
		]
		# selecIS_annotaion = {}
		# annos.append(selec_annotaion)
		# print("newlen:", len(annos))
		# print(annos[0],len(annos))
		# for i in annos:
			# print(len(i["segmentation"]))
			# print(i.keys()) #dict_keys(['iscrowd', 'bbox', 'category_id', 'segmentation', 'bbox_mode'])

		mask_target,delta_target = target_IS_generate(annos,image_shape)
		# print("mask_target:",mask_target.shape,"\n","delta_target:",delta_target.shape,delta_target.device) #torch.size[800, 1303, 2] cpu
		# selecIS_annotaion["selecgt_mask"] = mask_target
		# selecIS_annotaion["selecgt_delta"] = delta_target
		instances = utils.annotations_to_instances(
			annos, image_shape, mask_format=self.instance_mask_format
		)
		selecIS_annotation = Instances(image_shape)
		selecIS_annotation.selecgt_mask = mask_target
		selecIS_annotation.selecgt_delta = delta_target
		if self.recompute_boxes:
			# print("true")
			instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
		# print("count2")
		dataset_dict["instances"] = utils.filter_empty_instances(instances)
		# print("count3")
		# print("instances:",instances.seleIS_mask)
		dataset_dict["selecIS_annotation"] = selecIS_annotation

	# def inference_data(self,dataset_dict):
	# 	dataset_dict.pop("annotations", None)
	# 	dataset_dict.pop("sem_seg_file_name", None)
	# 	print("inference_orignal_dataset:",dataset_dict["height"],dataset_dict["width"])#426 640
	# 	image = utils.read_image(dataset_dict["file_name"],format=self.image_format) #BGR
	# 	utils.check_image_size(dataset_dict,image)
	#
	# 	print("inference dataset check:",dataset_dict.keys())
	# 	return dataset_dict


# if __name__=="__main__":
# 	dataloader = build_detection_train_loader(cfg, mapper=mapper)






































