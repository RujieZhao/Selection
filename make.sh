#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 ./train_net.py --config-file /ssd1/chaolu225/C++/selection_v1/config/selection_gpu.yaml --resume --num-gpus 2 SOLVER.IMS_PER_BATCH 2 MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl MODEL.SELECTION.STAGEFROZEN False OUTPUT_DIR ./output_3kminbatch # --resume wholedata 3kminbatch  SOLVER.BASE_LR 2e-4



#CUDA_VISIBLE_DEVICES=1 ./train_net.py --config-file /ssd1/chaolu225/C++/selection_v1/config/selection_gpu.yaml  --resume --num-gpus 1 SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS  detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl   # --resume

#CUDA_VISIBLE_DEVICES=1 ./train_net.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --resume --num-gpus 1 SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS  detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output_300minbatch # --resume
#CUDA_VISIBLE_DEVICES=2,3 ./train_net.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --resume --num-gpus 2 SOLVER.IMS_PER_BATCH 2 MODEL.WEIGHTS  detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output_wholedata # --resume 300minbatch

#CUDA_VISIBLE_DEVICES=2 ./train_net.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --resume --num-gpus 1 SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output # --resume _300minbatch

 # --resume MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output_300minbatch

#CUDA_VISIBLE_DEVICES=1 python3 train_net.py --config-file /ssd1/chaolu225/C++/selection_v1/config/selection_gpu.yaml --resume --eval-only

#Something wrong with this file.
#python3 test_targetgenerator.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml

#python3 test_IStarget_visualizer.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --output-dir /mnt/ssd2/rujie/predataset/test/ --show #annotation
#/mnt/ssd1/rujie/pytorch/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

#python3 test_visualizer.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output-dir /mnt/ssd2/rujie/selec_minbatch/300/batch1 --show #dataloader annotation --show

#python3 post_IS_testvisualize.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --output-dir /mnt/ssd2/rujie/predataset/test/ --show #dataloader

#python3 preparedata.py --source annotation --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --output-dir /mnt/ssd2/rujie/predataset/coco/  #annotation --show dataloader INPUT.PREDATASET.DIR
# /mnt/ssd2/rujie/predataset/coco/coco_2017_trainpre_21_110/ test



