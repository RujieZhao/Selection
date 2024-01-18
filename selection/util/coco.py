import json
import time
import contextlib
import io
import numpy as np
import os
from fvcore.common.timer import Timer
import logging
from pycocotools.coco import COCO
from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from collections import defaultdict

inaccurate_id = [480733,]

_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS","datasets"))
json_file = os.path.join(_root,"coco/annotations/instances_train2017.json")
# print(_root,json_file)
time0 = time.perf_counter()
with open(json_file,"r") as js:
	dataset = json.load(js)
	print(len(dataset["images"]))
	print(len(dataset["annotations"]))

	for ind,img in enumerate(dataset["images"]):

		if img["id"] in inaccurate_id:
			print("img:",img["id"])
			dataset["images"].pop(ind)

	for ind,ann in enumerate(dataset["annotations"]):
		if ann["image_id"] in inaccurate_id:
			print("ann:",ann["image_id"])
			dataset["annotations"].pop(ind)

	js.seek(0,0)
	json.dump(dataset,js)
	js.truncate(js.tell())

time1 = time.perf_counter()
print(time1-time0)


print(dataset.keys())
#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
# print(len(dataset["info"])) #6
# print(len(dataset["licenses"]),dataset["licenses"]) #8
# print(len(dataset["images"]),dataset["images"][0].keys(),dataset["images"][1]["id"],dataset["images"][1]["file_name"]) #118287
# print(len(dataset["annotations"]),dataset["annotations"][0].keys(),dataset["annotations"][3]["id"],dataset["annotations"][3]["image_id"])#860001 918 200365
# print(len(coco_api.dataset["categories"])) #80











