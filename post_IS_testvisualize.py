#This file is to test the IS2box module

import argparse
import os
from itertools import chain
import cv2
import numpy as np
import torch
import tqdm
from skimage import io
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from selection import selecmapper
from config import add_selection_config
from torchvision import transforms
from selection.tools import pyramid_Visualizer
# from selection import datamapper
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
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)

'''1.targetgenerator.py test set to True line65; 2. INPUT.PREDATASET.EN sets to False; 3. Set True for 
Pyramid_Visualizer(line 89)'''
if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # print("metadata:",type(metadata),len(metadata.thing_colors))#80

    def output(vis, fname):
        if args.show:
            # print(type(vis.get_image()),vis.get_image().shape,vis.get_image()[:, :, ::-1].shape) #<class 'numpy.ndarray'>(704, 1146, 3) (704, 1146, 3)
            cv2.imshow(fname, vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 1.0
    if args.source == "dataloader":
        mapper = selecmapper(cfg)
        train_data_loader = build_detection_train_loader(cfg,mapper = mapper)
        for ind,batch in enumerate(train_data_loader):
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img_name = per_image["file_name"].split("/")[-1].split(".")[0]
                # print("img_name:", img_name)
                # Orignal dataloader image shape is [3,H,W].My datamap output is [H,W,3]. Here I use my version bc
                # IS2box cuda core.
                # img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                img = per_image["image"].cpu().detach().numpy() #[ 704, 1146,    3]
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)#[ 704, 1146,    3]
                # img_size = torch.tensor(img.shape)
                ISmask_target = per_image["selecIS_annotation"]
                # print(type(ISmask_target["selecgt_mask"]),ISmask_target["selecgt_delta"].shape,ISmask_target["selecgt_delta"].dtype)#<class 'numpy.ndarray'> (640, 925, 2) int64
                visualizer = pyramid_Visualizer(ISmask_target, img, metadata=metadata, scale=scale,test=True)
                target_fields = per_image["instances"].get_fields()
                print("gt_box:",target_fields.get("gt_boxes", None).tensor.shape,target_fields.get("gt_boxes", None))
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]

                vis = visualizer.overlay_instances(
                    boxes=visualizer.predict_box.cpu().numpy())# labels=labels,
                    # boxes=target_fields.get("gt_boxes", None), masks=target_fields.get("gt_masks", None),
                    # keypoints=target_fields.get("gt_keypoints", None),
                output(vis, str(per_image["image_id"]) + ".jpg")

            if ind==0:
                break
    else:
        # print("dataset:",type(cfg.DATASETS.TRAIN))#<class 'tuple'>
        # for i in cfg.DATASETS.TRAIN:
        #     print("i:",i,type(i))#coco_2017_train<class 'str'>
            # dataset = DatasetCatalog.get(i)
            # print("dataset:",type(dataset),len(dataset),dataset[0])#<class 'list'> 118287
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        # print("dicts:",len(dicts),type(dicts),dicts[0])
        # print(cfg.MODEL.KEYPOINT_ON)#false
        # if cfg.MODEL.KEYPOINT_ON:
        #     dicts = filter_images_with_few_keypoints(dicts, 1)
        # for dic in tqdm.tqdm(dicts):
        for i,dic in enumerate(dicts):
            # print("dic_item:",dic.items())
            # print("dic_keys:",dic.keys()) #dict_keys(['file_name', 'height', 'width', 'image_id', 'annotations'])
            # print("dic_file_name:",dic["file_name"])
            img = utils.read_image(dic["file_name"], "RGB")

            # print("img:", type(img)) #numpy
            # img = torch.from_numpy(img)
            # print("img_shape:",img.shape) #(480, 640, 3)
            # height = img.shape[0]
            # weight = img.shape[1]
            # img = img.view(3,height,weight)
            # print("img_shape:",img.shape)
            # toPIL = transforms.ToPILImage()
            # img = toPIL(img)
            # print("img_type:",type(img))
            # img.save(dirname+"/image"+str(i)+".jpg")
            # print(weight,height)
            # np.save(img,dirname+"/image"+str(i)+".jpg")
            # torch.save(img,dirname+"/image"+str(i)+".jpg")
            # break

            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))
            # if i==50:
            break





























