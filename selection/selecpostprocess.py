import torch
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.structures import Boxes,Instances

#Refer to Detectron2
# TODO: In the future, mask prediction will be added below.
def sele_posprocess(predbox:Boxes,predcls:torch.Tensor,out_height,out_width,cur_size:list,scores,nms_thre=0.5,mask_threshould:float=0.5):
    """
    This function is for getting the results ready for evaluation.
    It contains 3 steps:
    1. find the remaining index with predbox and classes;
    2. rescale results to original image size and encapsulate mask to bitmask;
    3. Wrap all of them into an Instance.
    """
    processed_results = []
    # fakescore = torch.ones(predbox.tensor.shape[0])
    # fakescore = fakescore.to(predbox.tensor)
    # keep = batched_nms(predbox.tensor,fakescore,predcls,nms_thre)
    # predbox.tensor,predcls = predbox.tensor[keep],predcls[keep]
    if isinstance(out_height,torch.Tensor):
        orig_size = torch.stack([out_height,out_width]).to(torch.float32)
    else:
        orig_size = (float(out_height),float(out_width))
    print("orig_size:",orig_size)
    scale_x,scale_y = (orig_size[0]/cur_size[0],orig_size[1]/cur_size[1])
    print(scale_x,scale_y)
    result = Instances(orig_size)
    finalbox = predbox.clone()
    finalbox.scale(scale_x,scale_y)
    finalbox.clip(result.image_size)#(426.0, 640.0)
    result.pred_boxes = finalbox
    # result.scores = scores[keep]
    result.scores = scores
    result.pred_classes = predcls
    processed_results.append({"instances":result})
    return processed_results






















