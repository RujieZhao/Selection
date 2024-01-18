import argparse
import sys
sys.path.append("..")
import cv2
import torch
import numpy as np
import os
import time
from typing import cast, IO
from itertools import chain
from PIL import Image
from selection import selcuda,maskcreate
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from torchvision import transforms
from config import add_selection_config
import matplotlib.pyplot as plt
from selection.util import pixel_perception
from selection import selecmapper

# dir_path = "/mnt/ssd2/rujie/coco_visualizer"
# dir_path = "/mnt/ssd1/rujie/pytorch/dataset/coco/train2017/000000581929.jpg"
# dir_path = "../visualizer/image0.jpg"
# dir_path = "/mnt/ssd2/rujie/predataset/test/image0.jpg"

# data = Image.open(dir_path)
# print("information:",data.size,type(data))
# data = np.asarray(data)
# data = torch.from_numpy(data)
# print("data:",type(data))

# data = torch.load(dir_path)

# img = utils.read_image(dir_path,"RGB")
# print("img:",img.shape,type(img))

def setup(args):
	cfg = get_cfg()
	add_selection_config(cfg)
	if args.config_file:
		cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.DATALOADER.NUM_WORKERS = 0
	cfg.freeze()
	return cfg

def parse_args(in_args=None):
	parser = argparse.ArgumentParser(description="prepare data")
	parser.add_argument(
		"--source",
		choices=["annotation", "dataloader"],
		required=True,
		help="visualize the annotations or the data loader (with pre-processing)",
	)
	parser.add_argument("--config-file", metavar="FILE")
	parser.add_argument("--output-dir",default="/mnt/ssd2/rujie/predataset/coco/")
	parser.add_argument("--show", action="store_true", help="show output in a window")
	parser.add_argument("opts",default=None,nargs=argparse.REMAINDER)
	return parser.parse_args(in_args)

'''NEED TO SET INPUT.PREDATASET.EN TO FALSE and targetgenerator.py test = Flase bc we need tensor type to proceed!!!'''
if __name__ == "__main__":
	time00=time.perf_counter()
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	args = parse_args()
	logger = setup_logger()
	logger.info("Arguments:"+str(args))
	cfg = setup(args)
	t0 = time.perf_counter()
	selesensor = pixel_perception(cfg)
	# mc,num = maskcreate(cfg.MODEL.GPU.MA,cfg.MODEL.GPU.MT)
	# mask = mc #.to(device)
	# delta = cfg.MODEL.GPU.DEL
	# ratio = cfg.MODEL.GPU.RAT
	# ratio = torch.tensor(list(cfg.MODEL.GPU.RAT),dtype=torch.float32,device=device)
	t1 = time.perf_counter()
	# print("maskcreate time:",t1-t0) #0.0008717229356989264
	dirname = args.output_dir +"coco_2017_trainpre_"+str(cfg.MODEL.GPU.MA)+"_"+str(cfg.MODEL.GPU.DEL) + "min640/" #"coco_2017_trainpre_"
	# print("dirname:", dirname, "cocotest" cfg.INPUT.PREDATASET.EN,cfg.INPUT.PREDATASET.DIR)

	# os.makedirs(dirname,exist_ok=True)
	metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
	# print(len(metadata.get("thing_colors")))
	toPIL = transforms.ToPILImage()
	def output(orig_img,selec_out,fname):
		filepath = fname.split(".")
		# print(type(filepath),filepath) #<class 'list'> ['000000226258', 'jpg']
		filedir = os.path.join(dirname,filepath[0])
		print("filedir:",filedir)
		os.makedirs(filedir,exist_ok=True)

		orig_img = orig_img.permute(2, 0, 1)
		print("orig_img:",orig_img.shape,orig_img.device,orig_img.dtype) #[3,H,W]
		# print("selec_out:",selec_out.dtype)
		if args.show:
			for i in range(4):
				if i ==0:
					orig_img = toPIL(orig_img)
					plt.figure(filepath[0] + str(i))
					plt.imshow(orig_img)
				else:
					selec_img = toPIL(selec_out[i - 1])
					plt.figure(filepath[0] + str(i))
					plt.imshow(selec_img)
			plt.show()
		else:
			# for c in selec_out:
			# 	for b in range(3):
			# 		print("selec_out:",c.shape,c[b,100:120,100])
			# 	print("=============================== ")

			selecimg = []
			selec_out = selec_out.numpy()
			# print(selec_out.shape,type(selec_out),selec_out.dtype)
			for i in range(3):
				# print(i)
				selecimg.append(selec_out[i])
				# print("selecimg:",selecimg[i].shape)
			imgsave = [orig_img.numpy()]+selecimg
			# print(len(imgsave),imgsave[0].shape,type(imgsave[0]))

			for i in range(4):
				# 	# with open(filedir+"/"+filepath[0]+"#"+str(i)+".jpg","wb") as f:
				# 	# 	torch.save(imgsave[i],cast(IO[bytes],f))
				np.save(filedir+"/"+filepath[0]+"#"+str(i),imgsave[i])
		'''
		for i in range(4):
			if i == 0:
				# if args.source == "annotation":
				orig_imgPIL = orig_img.permute(2, 0, 1)
				# print("orig_img_save:", orig_img.shape, type(orig_img), orig_img.dtype,orig_img[1,100:120,100])  # torch.Size([3, 704, 1146])
				orig_imgPIL = toPIL(orig_imgPIL)
				if args.show:
					plt.figure(filepath[0]+str(i))
					plt.imshow(orig_imgPIL)
				else:
					orig_imgPIL.save(filedir+"/"+filepath[0]+"#"+str(i)+".jpg")
					# with open("/mnt/ssd1/rujie/pytorch/templete/000000411108_torchsave.jpg","wb") as f:
					#     torch.save(img_ori_load,cast(IO[bytes],f))
			else:
				selec_img = toPIL(selec_out[i - 1])
				# print("selec_img:",selec_out[i-1].shape,selec_out[i-1][1,100:120,100])
				# print("selec_img:",selec_out.shape,type(selec_out),selec_out.dtype) #torch.Size([3, 3, 480, 640]) <class 'torch.Tensor'>
				# selec_img = toPIL(torch.flip(selec_out[i-1].permute(2,0,1),dims=[0]))
				if args.show:
					# print("selec_img:",selec_img.shape)
					plt.figure(filepath[0]+str(i))
					plt.imshow(selec_img)
				else:
					# selec_img=selec_out[i - 1].permute(1,2,0)
					# selec_img = toPIL(selec_img)
					selec_img.save(filedir+"/"+filepath[0]+"#"+str(i)+".jpg")
		if args.show:
			plt.show()
		'''
	mapper = selecmapper(cfg, True)
	if args.source == "annotation":
		# print("start annotation")
		#I will keep all input pics same fixed size(640), bs randomly and blindly scaling pics is not the right way to sense objects.
		dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
		# print(dicts[0].keys()) #['file_name', 'height', 'width', 'image_id', 'annotations']
		for i,dic in enumerate(dicts):
			# if dic["image_id"]==411108:
			'''
			time0 = time.perf_counter()
			orig_img = utils.read_image(dic["file_name"],"BGR") #BGR "RGB"
			print("heigh&weightt:",dic["height"],dic["width"])
			for a in range(3):
				print("orig_img_shape:",type(orig_img),orig_img.shape,orig_img[100:120,100,a]) #torch.Size([480, 640, 3])
			utils.check_image_size(dic, orig_img)		
			orig_img = torch.from_numpy(orig_img.copy())
			orig_img_gpu = orig_img.to(device).type(torch.cuda.FloatTensor)
			time00 = time.perf_counter()
			selec_out = selesensor(orig_img_gpu)
			time1 = time.perf_counter()
			# print("selcuda time cost:",time1-time00,selec_out.shape,selec_out.dtype,selec_out.device) #0.0001490359427407384 torch.Size([3, 3, 426, 640]) torch.float32 cuda:0
			selec_out = selec_out.cpu().type(torch.uint8).contiguous() #.permute(0,2,3,1)
			# print("selec_out_shape:",selec_out.shape)
			time2 = time.perf_counter()
			# print("selcuda type convert:",time2-time1) #0.364
			# print("selec_out:",type(selec_out),len(selec_out),selec_out.shape,selec_out.is_contiguous()) #<class 'torch.Tensor'> 3 torch.Size([3, 480, 640, 3]) True
			output(orig_img,selec_out,os.path.basename(dic["file_name"]))
			'''
			datasetdict = mapper(dic)
			# print("newpredata:", datasetdict.keys(), datasetdict["image"].shape)  # dict_keys(['file_name', 'height', 'width', 'image_id', 'image', 'input_shape', 'instances', 'selecIS_annotation']) torch.Size([640, 850, 3])
			orig_img = datasetdict["image"].to(device).type(torch.cuda.FloatTensor)
			# print(img.device,img.dtype,img.type()) #cuda:0 torch.float32 torch.cuda.FloatTensor
			# for i in range(3):
			# 	print(orig_img[100:120,100,i])
			time0 = time.perf_counter()
			selec_out = selesensor(orig_img).type(torch.uint8).cpu().detach().contiguous()
			orig_img = orig_img.type(torch.uint8).cpu().detach().contiguous()
			time1 = time.perf_counter()
			print("selcuda time cost:", time1 - time0,selec_out.shape,selec_out.dtype)
			output(orig_img, selec_out, os.path.basename(dic["file_name"]))

			# if i == 1:
			# 	break
	if args.source == "dataloader":

		train_data_loader = build_detection_train_loader(cfg,mapper=mapper)
		# print("predata:",cfg.INPUT.PREDATASET.EN)
		for ind,batch in enumerate(train_data_loader):
			for per_image in batch:
				if not cfg.INPUT.PREDATASET.EN:
					# print(type(per_image["image"]),per_image["image"].dtype,per_image["image"].device) #<class 'torch.Tensor'> torch.uint8 cpu
					img = per_image["image"]
					orig_img = img[:, :, [2, 1, 0]].contiguous() # "BGR" TO "RBG"
					# print("orig_img_shape:",orig_img.shape,type(orig_img),orig_img.dtype) #[704, 1146,3]
					orig_img_gpu = orig_img.to(device).type(torch.cuda.FloatTensor)

					time0 = time.perf_counter()
					selec_out = selesensor(orig_img_gpu)
					time1 = time.perf_counter()
					print("selcuda time cost:",time1 - time0) #7.727699994575232e-05
					print("selec_out:",selec_out.shape)

					selec_out = selec_out.cpu().type(torch.uint8).contiguous()
					# selec_out = orig_img_gpu.cpu().type(torch.uint8)
					# print("selec_out:", type(selec_out), len(selec_out), selec_out.shape, selec_out.is_contiguous()) #[3, 3, 640, 959]
					time2 = time.perf_counter()
					print("selcuda type convert:", time2 - time1)
					output(orig_img, selec_out, os.path.basename(per_image["file_name"]))
				else:
					orig_img = per_image["image"][0].permute(1,2,0)[:, :, [2, 1, 0]]
					for i in range(1,4):
						per_image["image"][i] = per_image["image"][i][[2, 1, 0],:,:]
						# print("selec_out_shape:",per_image["image"][i].shape,per_image["image"][i][100:120,100,1])
					selec_out = torch.stack(per_image["image"][1:])
					# print(selec_out.shape)
					output(orig_img, selec_out, os.path.basename(per_image["file_name"]))
			if ind == 0:
				break
# _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
# print("root:",_root,os.getenv("datasets"))












