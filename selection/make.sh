#!/usr/bin/env bash

#python3 test_visualizer.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output-dir /mnt/ssd2/rujie/predataset/test/ --show # annotation

#python3 test_visualizer.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --output-dir /mnt/ssd2/rujie/predataset/test/ --show # annotation

python3 setup.py build_ext --inplace

#python3 test1.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml












