

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
from torchvision import transforms
# from selection import datamapper
# from config import add_selection_config
def setup(args):
    cfg = get_cfg()
    # add_selection_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH=1
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


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    print("metadata:",type(metadata),len(metadata.thing_colors))#80

    def output(vis, fname):
        if args.show:
            # print(fname)
            print(type(vis.get_image()),vis.get_image().shape)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 1.0

    minbatch = list()
    if args.source == "dataloader":
        # mapper = datamapper.selecmapper(cfg),mapper = mapper
        train_data_loader = build_detection_train_loader(cfg)
        for ind,batch in enumerate(train_data_loader):
            for per_image in batch:
                # print("per_image:",per_image.keys())
                # ['file_name', 'height', 'width', 'image_id', 'image', 'instances']
                # print("per_image name:",per_image["file_name"],per_image["image_id"],type(per_image["image_id"])) #<class 'int'>

                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy() #(704, 1146, 3)
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT) #"BGR"

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                # print("per_image:",str(per_image["image_id"])) # 573286
                # print("instance:",per_image["instances"])
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
                minbatch.append(per_image["image_id"])
            if ind == 0:
                print(minbatch) #<class 'int'>
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











