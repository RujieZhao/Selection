#!/usr/bin/env python3

# import sys
# sys.path.append("/mnt/ssd1/rujie/pytorch/ImageDetection/maskrcnn/detectron2")
# import copy
# import detectron2
import logging
import os
from collections import OrderedDict
import torch
from contextlib import ExitStack, contextmanager
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch,SimpleTrainer
from detectron2.data import MetadataCatalog,build_detection_train_loader,build_detection_test_loader
from detectron2.evaluation import (
	CityscapesInstanceEvaluator,
	CityscapesSemSegEvaluator,
	COCOEvaluator,
	COCOPanopticEvaluator,
	DatasetEvaluators,
	LVISEvaluator,
	PascalVOCDetectionEvaluator,
	SemSegEvaluator,
	inference_on_dataset,
	print_csv_format,
	verify_results,
)
from detectron2.utils.logger import setup_logger
from detectron2.modeling import GeneralizedRCNNWithTTA
from config import add_selection_config
from selection import selecmapper
import time

print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.")


class Trainer(DefaultTrainer):
    def run_step(self):
        assert self._trainer.model.training, "model was changed to eval model"
        # print("run_step test",__name__,self._trainer._data_loader_iter) #run_step test __main__ True
        start  = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # print("second_data:",data[0]["file_name"])
        # print("data_test:",len(data[0]),data[0].keys(),len(data[0]["image"]),data[0]["width"]) #7 dict_keys(['file_name', 'height', 'width', 'image_id', 'image', 'instances', 'selecIS_annotation']) 4 640
        data_time = time.perf_counter() - start
        print("dataload time:",data_time)

        # train_stage = 1
        # if self.iter<3000000:
        #     train_stage = 0
        # else:
        train_stage = 0 # 0 1
        train_time0 = time.perf_counter()
        loss_dict = self._trainer.model(data,train_stage=train_stage)
        # print("loss_dict gpu check:",type(loss_dict),loss_dict) #<class 'dict'>
        train_time1 = time.perf_counter()
        print("trian_time:",train_time1-train_time0)
        # print("loss value check:",loss_dict)

        if isinstance(loss_dict,torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        # print("loss_gpu:",losses)
        self._trainer.optimizer.zero_grad()
        losses.backward() #0.1177
        self._trainer._write_metrics(loss_dict,data_time) #0.50453
        print(self._trainer.optimizer.state_dict()['param_groups'][0]["lr"])
        self._trainer.optimizer.step()
        end = time.perf_counter()
        print("total time:", end - start, end - train_time1)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (torch.cuda.device_count() > comm.get_rank()), \
                "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (torch.cuda.device_count() > comm.get_rank()), \
                "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type))
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = selecmapper(cfg,True)
        return build_detection_train_loader(cfg,mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = selecmapper(cfg,False)
        return build_detection_test_loader(cfg,dataset_name,mapper)


def setup(args):
    cfg = get_cfg()
    add_selection_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    default_setup(cfg,args)
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="selection")
    return cfg


def main(args):
    cfg = setup(args)
    # print("Command Line Args:", args)
    if args.eval_only:
        print("eval True")
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        # 	res.update(Trainer.test_with_TTA(cfg, model))
        # if comm.is_main_process():
        # 	verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )





















































