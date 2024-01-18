

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
from selection.util import pre_data,IS2box
from selection import target_FL_generate

"""
This testing file is to verify Focal level target generator mode.
The main function of this mode is to calucuate the iou rate for gt box and pred box of each FL. And use the threshold vlue to obtain the valid iou rate and then assign the rate to relative FL.
After collecting all FL iou, compare their corresponding gtbox iou, and find out the FL indice with most matached predboxes.
"""

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
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    # dirname = args.output_dir
    # os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # print("metadata:",type(metadata),len(metadata.thing_colors))#80
    scale = 1.0
    mapper = selecmapper(cfg)
    train_data_loader = build_detection_train_loader(cfg,mapper = mapper)
    for ind,batch in enumerate(train_data_loader):
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            img_name = per_image["file_name"].split("/")[-1].split(".")[0]
            print("img_name:", img_name)
            img = per_image["image"][0].permute(1, 2, 0).cpu().detach().numpy()
            print("image_shape:",img.shape)
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
            target_fields =  per_image["instances"].get_fields()

            ISmask_target = per_image["selecIS_annotation"]
            # print(type(ISmask_target["selecgt_mask"]),ISmask_target["selecgt_delta"].shape,ISmask_target["selecgt_delta"].dtype)#<class 'numpy.ndarray'> (640, 925, 2)
            imagesize = per_image["input_shape"]
            mask, delta, checkboard, img_height, img_width = pre_data(imagesize,ISmask_target)
            predict_box = IS2box(mask, delta, checkboard, img_height, img_width)
            pred_box_len = [predict_box.tensor.shape[0]]
            gt_boxes = target_fields.get("gt_boxes", None).to(predict_box.device)
            # (['gt_boxes', 'gt_classes', 'gt_masks']
            print("gt and pred box type and shape:",gt_boxes.tensor.shape,predict_box.tensor.shape,gt_boxes.device,predict_box.device)
            FL_target = target_FL_generate(cfg)
            iou_matrix,indx,labels = FL_target.iou_generator(gt_boxes,predict_box)
            print(type(iou_matrix),iou_matrix.shape,iou_matrix,indx.shape,indx,labels.shape,labels)#[N,M]
            lv_target = FL_target.level_target_generator(pred_box_len,iou_matrix,indx,labels)
            print("lv_target:",lv_target)
        if ind==0:
            break










































