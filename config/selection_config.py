# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN
# from detectron2.config import get_cfg
def add_selection_config(cfg):
	cfg.VIS_PERIOD = 0
	cfg.MODEL.MASK_ON = True

	cfg.MODEL.SELECTION = CN()
	cfg.MODEL.SELECTION.STAGEFROZEN = False
	cfg.MODEL.SELECTION.ALPHA =2.
	cfg.MODEL.SELECTION.NUM_CLASS = 80
	#If it is True, only one FL level will be used.
	cfg.MODEL.SELECTION.ONEHOT = True
	cfg.MODEL.SELECTION.IOU_THRESHOLDS= [0.05]
	cfg.MODEL.SELECTION.IOU_LABELS = [0, 1]
	cfg.MODEL.SELECTION.MASK_WEIGHT =5.0
	cfg.MODEL.SELECTION.DELTA_WEIGHT = 5.0
	cfg.MODEL.SELECTION.LV_WEIGHT = 1.0
	cfg.MODEL.SELECTION.GIOU_WEIGHT = 2.0
	# cfg.MODEL.SELECTION.BOX_WEIGHT = 2.0
	cfg.MODEL.SELECTION.CLS_WEIGHT =1.0
	cfg.MODEL.SELECTION.GIOU_LOSS = True
	cfg.MODEL.SELECTION.BOX_LOSS = True

	cfg.MODEL.GPU = CN()
	cfg.MODEL.GPU.FL = 3
	cfg.MODEL.GPU.MT = "contr"
	cfg.MODEL.GPU.MA = 21
	cfg.MODEL.GPU.DEL = 110
	cfg.MODEL.GPU.RAT = (0.5,0.7,0.9)

	cfg.MODEL.PY = CN()
	cfg.MODEL.PY.MASK_TH = 0.5
	cfg.MODEL.PY.PATCH=7
	cfg.MODEL.PY.PYLAYER=2
	cfg.MODEL.PY.NUM_IGNORE=1
	cfg.MODEL.PY.AMP = 0.2
	cfg.MODEL.PY.NUM_TH = 4

	cfg.INPUT.RESIZE= True
	cfg.INPUT.CROP.ENABLED=False
	cfg.INPUT.PREDATASET=CN({"EN":True})
	#the predataset directory needs to be constructed manually, it can be writen here or at arg.opts
	cfg.INPUT.PREDATASET.DIR="/mnt/ssd2/rujie/predataset/coco/coco_2017_trainpre_21_110/"
	cfg.INPUT.AUGINPUT = CN({"EN":True})

	cfg.MODEL.BACKBONE.FL_EN = True
	cfg.MODEL.BACKBONE.IS_EN = True
	cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
	cfg.MODEL.BACKBONE.FREEZE_AT = 0
	# cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY = 32
	cfg.MODEL.RESNETS.OUT_FEATURES=["res2", "res3", "res4", "res5"]
	cfg.MODEL.BACKBONE.PRETRAIN=None

	cfg.MODEL.SELECBACKBONE=CN({"ENABLED": True})
	cfg.MODEL.SELECBACKBONE.NAME="build_is_backbone"
	cfg.MODEL.BACKBONE.PRETRAIN_IMAGE_SIZE = 224
	cfg.MODEL.BACKBONE.PATCH_EN = True
	cfg.MODEL.BACKBONE.PATCH_SIZE = 4
	cfg.MODEL.BACKBONE.IN_CHAN = 3
	cfg.MODEL.BACKBONE.EMBED_DIM = 96
	cfg.MODEL.BACKBONE.NUM_CLASSES = 80
	cfg.MODEL.BACKBONE.DEPTHS = [2, 2, 6, 2]
	cfg.MODEL.BACKBONE.NUM_HEAD = [3, 6, 12, 24]
	cfg.MODEL.BACKBONE.WINDOW_SIZE = 7
	cfg.MODEL.BACKBONE.MLP_RATIO = 4
	cfg.MODEL.BACKBONE.QKV_BIAS = True
	cfg.MODEL.BACKBONE.QK_SCALE = None
	cfg.MODEL.BACKBONE.DROP_RATE = 0.
	cfg.MODEL.BACKBONE.ATTN_DROP_RATE=0.
	cfg.MODEL.BACKBONE.DROP_PATH_RATE = 0.1
	cfg.MODEL.BACKBONE.APE = False
	cfg.MODEL.BACKBONE.PATCH_NORM = True
	cfg.MODEL.BACKBONE.OUT_INDICES = (0,1,2,3)
	cfg.MODEL.BACKBONE.FROZEN_STAGES = -1
	cfg.MODEL.BACKBONE.USE_CHECKPOINT = False

	cfg.MODEL.IShead = CN()
	cfg.MODEL.IShead.NAME = "IShead"
	cfg.MODEL.IShead.dim = [96,192,384,768]
	cfg.MODEL.IShead.layer = 3
	cfg.MODEL.IShead.start_level = -1

	cfg.MODEL.FLhead = CN()
	cfg.MODEL.FLhead.NAME = "FLhead"
	cfg.MODEL.FLhead.FL_NUM = 4
	cfg.MODEL.FLhead.GIOU = False












