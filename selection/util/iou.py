
import torch
from detectron2.structures.boxes import Boxes
# from torchvision.ops.boxes import box_area

# This file is for processing predbox and matched gt_box and used to calculate giou loss.
# The difference between with detectron2 is the input and output size. predbox and gtbox shape are alwayas same.
# Output size is one dimension not [N,M].

def giou_generate(iou:torch.Tensor,predbox,gtbox) ->torch.Tensor:
    assert predbox.shape==gtbox.shape,"predbox and gtbox's shape are not matched."
    assert type(predbox)==type(gtbox),"predbox and gtbox has different type!"
    if isinstance(predbox,torch.Tensor):
        predbox_area = (predbox[:,2]-predbox[:,0])*(predbox[:,3]-predbox[:,1])
        gtbox_area = (gtbox[:,2]-gtbox[:,0])*(gtbox[:,3]-gtbox[:,1])
    if isinstance(predbox,Boxes):
        predbox_area = predbox.area()
        gtbox_area = gtbox.area()
    # print("area:",predbox_area,gtbox_area)
    #In selection batch size is always 1. so box shape is a 2d array[-1,4].
    inter_lt = torch.max(predbox[:,:2],gtbox[:,:2])
    inter_rb = torch.min(predbox[:,2:],gtbox[:,2:])
    inter_wh = (inter_rb-inter_lt).clamp(min=0) #[N:2]
    # print(inter_lt,inter_rb,inter_wh)
    intersec = inter_wh[:,0]*inter_wh[:,1]
    union = predbox_area+gtbox_area-intersec
    # iou_test = torch.round((intersec/union),decimals=6)
    # print("iou_check:",intersec.shape,union.shape,iou_test.shape,iou.shape)
    iou_check = torch.round((intersec/union),decimals=3)==torch.round(iou,decimals=3)
    assert iou_check.all(),"there is a bad iou rate!"
    # print(intersec,union,intersec/union)
    clo_lt = torch.min(predbox[:,:2],gtbox[:,:2])
    clo_rb = torch.max(predbox[:,2:],gtbox[:,2:])
    clo_wh = (clo_rb-clo_lt).clamp(min=0)
    # print("clo:",clo_lt,clo_rb,clo_wh)
    closure = clo_wh[:,0]*clo_wh[:,1]
    # print(closure,closure-union,(closure-union)/closure)
    giou = iou-(closure-union)/closure
    return giou

# if __name__=="__main__":
#     pred = torch.tensor([[1,1,5,6],[1,1,5,5]])
#     gt = torch.tensor([[3,3,8,10],[4,4,8,8]])
#     iou = torch.tensor([0.1224489,0.032258])
#     print(pred.shape,iou.shape)
#     giou = giou_generate(iou,pred,gt)
#     print(giou,giou.device)
#     loss = 1-giou
#     print(torch.mean(loss))