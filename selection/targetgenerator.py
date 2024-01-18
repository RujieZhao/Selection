import time

import torch
import torch.nn.functional as F
import numpy as np
import copy
from collections import OrderedDict
from typing import Dict,List,Optional
from detectron2.modeling.matcher import Matcher
from detectron2.config import configurable
from detectron2.structures import BoxMode,Boxes,pairwise_iou,Instances
from detectron2.data import transforms as T
from detectron2.utils.memory import retry_if_cuda_oom
import pycocotools.mask as mask_util
from .util import IS2box,pre_data,totensor,tonumpy

__all__=["transform_instance_annotations","target_IS_generate","target_FL_generate"]

def transform_instance_annotations(annotation, transforms, image_size):
    # print("use selec part")
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # print("bbox_mode:",annotation["bbox_mode"]) #XYWH
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    # if not "segmentation" in annotation:
    #     print("False")
    #coalesce segs within one instance
    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        # print("segm:", len(segm), type(segm), len(segm[0]))
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            # print("polygons shape:",polygons[0].shape) #(5, 2)
            polygons_trans = transforms.apply_polygons(polygons)
            # print("polygons_trans:",len(polygons_trans),type(polygons_trans),polygons_trans)
            # for j in polygons_trans:
                # print(j.shape,type(j)) #(66, 2) <class 'numpy.ndarray'>

            annotation["segmentation"] = [
                p.reshape(-1) for p in polygons_trans
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    return annotation

#TODO use C to recode it for future speed enhancement
def target_IS_generate(annos,image_shape,thre=3,test=False):
    """
    return IS_target shape=[img_h,img_w,3]; delta shape is corresponding to [height,width].
    """
    area =[]
    preserve = np.zeros((image_shape[0],image_shape[1],2),dtype=np.int32)
    target_mask = np.full((image_shape[0],image_shape[1]),0)
    target_delt = np.full((image_shape[0],image_shape[1],2),0)
    #have properties in each instances
    segms = copy.deepcopy([obj["segmentation"] for obj in annos])
    boxes = copy.deepcopy([obj["bbox"] for obj in annos])
    # print("boxes_num:",len(boxes))
    for ind,coord in enumerate(boxes):
        # print("orignal coords",coord,coord.shape) #[391.10050325 253.66666667 405.42823763 290.62666829] (4,)
        assert len(coord) == 4, "box coors len is not 4."
        area.append(int(coord[2] * coord[3]))
        coord = np.round(coord) #[tl_x,tl_y,br_x,br_y]
        # print("target_coor:",coord,type(coord)) #[112 379 222 507] <class 'numpy.ndarray'>
        cen_height = np.round((coord[1]+coord[3])/2)
        cen_width = np.round((coord[0]+coord[2])/2)
        # print("centercoor(w,h):",cen_width,cen_height) #253.0 109.0
        for m_ind, mask in enumerate(segms[ind]):
            # print("contour length:",len(mask),type(mask)) #70 <class 'numpy.ndarray'>
            assert isinstance(mask,np.ndarray)
            mask = np.round(mask).astype(int).reshape(-1,2)
            co_height = np.clip(mask[:,1],0,image_shape[0]-1)
            co_width = np.clip(mask[:,0],0,image_shape[1]-1)
            # print("mask:",mask)
            # print("co:",co_height,co_width)
            # print(co_height.shape,co_width.shape) #(64,) (64,)
            indice0 = np.where(target_mask[co_height,co_width]==0)[0]
            indice1 = np.where(target_mask[co_height, co_width] == 1)[0]
            empty_height = co_height[indice0]
            empty_width = co_width[indice0]
            target_mask[empty_height, empty_width] = 1
            preserve[empty_height, empty_width, 0] = ind
            preserve[empty_height, empty_width, 1] = m_ind
            delta = np.stack((empty_height - cen_height, empty_width - cen_width), axis=1)
            target_delt[empty_height, empty_width] = delta
            if mask.shape[0] != len(indice0):
                # The following 3 steps is to find the valid index for overlap mask points. It contains 3 conditions to leach them out.
                con_count=0
                # print(f"Discover overlap points, total num is {len(indice1)};instance index in one pic: {ind + 1}")
                overlap_height = co_height[indice1]
                overlap_width = co_width[indice1]
                preserve_id = preserve[overlap_height,overlap_width] #[n,2]
                overlap_l = preserve_id.shape[0]
                curr_area = area[ind]
                for order in range(overlap_l):
                    # print("dealing with overlap point NO.{}".format(order))
                    num_pre_points = int(len(segms[preserve_id[order,0]][preserve_id[order,1]])/2)
                    curr_height = overlap_height[order]
                    curr_width = overlap_width[order]
                    #first condition is to guarantee the mask counture has min points num so that the object's feature property can be saved cosistently
                    # print("condition1 details: pre points num {}; curr num {}".format(num_pre_points,mask.shape[0]))
                    if ((num_pre_points>thre) & ((mask.shape[0]-(order-con_count))==thre)):
                        # print("condition1 matched: pre points is {};currentpoints is {}".format(num_pre_points,mask.shape[0]))
                        con_count+=1
                        preserve[curr_height,curr_width,0] = ind
                        preserve[curr_height,curr_width,1] = m_ind
                        delta = np.stack((curr_height-cen_height,curr_width-cen_width))
                        target_delt[curr_height,curr_width]=delta
                        continue
                    elif ((num_pre_points==thre) & ((mask.shape[0]-(order-con_count))>thre)):
                        # print("condition1 matched: pre points is {};currentpoints is {}".format(num_pre_points,mask.shape[0]))
                        continue
                    elif((num_pre_points<=thre) & ((mask.shape[0]-(order-con_count))<=thre)):
                        raise ValueError("previous preserve and current mask all have 3 points")
                    #second consition is to guarantee the IOU rate of boxes. As my concerned, it is less important than feature proerty completeness
                    pre_vertex = True if (overlap_width[order] in boxes[preserve_id[order,0]].astype(int)) or (overlap_height[order] in boxes[preserve_id[order,0]].astype(int)) else False
                    # print("pre_vertex:",overlap_width[order],overlap_height[order],boxes[preserve_id[order,0]].astype(int))
                    curr_vertex = True if (overlap_width[order] in coord) or (overlap_height[order] in coord) else False
                    # print("curr_vertex:",coord)
                    # print("condition2 details: pre vertex {}; curr vertex {}".format(pre_vertex,curr_vertex))
                    if (curr_vertex & (not pre_vertex)):
                        # print("condition2 matched. w&h are {} and {}; orignal coors is {}".format(overlap_width,overlap_height,coord))
                        con_count += 1
                        preserve[curr_height,curr_width,0]=ind
                        preserve[curr_height,curr_width,1]=m_ind
                        delta=np.stack((curr_height-cen_height,curr_width-cen_width))
                        target_delt[curr_height,curr_width]=delta
                        continue
                    elif (pre_vertex & (not curr_vertex)):
                        # print("condition2 matched. w&h are {} and {}; orignal coors is {}".format(overlap_width,overlap_height,coord))
                        continue
                    elif (pre_vertex & curr_vertex & (area[preserve_id[order,0]]<curr_area)):
                        # print("condition2 matched. w&h are {} and {}; orignal coors is {}".format(overlap_width,overlap_height,coord))
                        con_count += 1
                        preserve[curr_height, curr_width, 0] = ind
                        preserve[curr_height, curr_width, 1] = m_ind
                        delta = np.stack((curr_height - cen_height, curr_width - cen_width))
                        target_delt[curr_height, curr_width] = delta
                        continue
                    #third consition is to guarantee the overlap point belongs to the smaller objects by compare boxes' areas, which is the least essential aspect with IOU and object feature quality.
                    if ((area[preserve_id[order,0]]>curr_area)&(num_pre_points>(mask.shape[0]-(order-con_count)))):
                        # print("condition3 matched: pre area is {}; current area is {}".format(area[preserve_id[order,0]],curr_area))
                        con_count += 1
                        preserve[curr_height, curr_width, 0] = ind
                        preserve[curr_height, curr_width, 1] = m_ind
                        delta = np.stack((curr_height-cen_height, curr_width-cen_width))
                        target_delt[curr_height, curr_width] = delta
                        continue
    # print("inside_mask_target:", target_mask.shape, type(target_mask),target_mask.dtype, "\n", "delta_target:", target_delt.shape, type(target_delt),target_delt.dtype) #(800, 1303, 2) <class 'numpy.ndarray'> int64
    target_mask = totensor(target_mask,cuda=False,test=test)
    target_delt = totensor(target_delt,cuda=False,test=test)

    # mas = target_mask.flatten().to(torch.bool)
    # de = target_delt.flatten(0,1)
    # print(mas.shape,de.shape)
    # de = de[mas]
    # print(de.shape)
    # max = torch.max(abs(de))
    # min = torch.min(abs(de))
    # print("max&min:",max,min)
    # loc = torch.where(de==min) # [77]), tensor([1]
    # print(loc)
    # v_del = de[loc[0]] #[-32.,   0.]
    # print("value:",v_del)

    return target_mask,target_delt

class target_FL_generate():

    @configurable
    def __init__(self,one_hot,num_class,num_FL,mask_th,patch,pylayer,num_ignore,amp,num_th,box_matcher,GIOU):
        self.num_class = num_class
        self.one_hot = one_hot
        self.num_FL = num_FL
        self.mask_th = mask_th
        self.patch = patch
        self.pylayer = pylayer
        self.num_ignore = num_ignore
        self.amp = amp
        self.num_th = num_th
        self.matcher = box_matcher
        self.GIOU = GIOU
    @classmethod
    def from_config(cls,cfg):
        ret = {
            "num_class": cfg.MODEL.SELECTION.NUM_CLASS,
            "one_hot": cfg.MODEL.SELECTION.ONEHOT,
            "num_FL": cfg.MODEL.GPU.FL,
            "mask_th": cfg.MODEL.PY.MASK_TH,
            "patch": cfg.MODEL.PY.PATCH,
            "pylayer": cfg.MODEL.PY.PYLAYER,
            "num_ignore": cfg.MODEL.PY.NUM_IGNORE,
            "amp": cfg.MODEL.PY.AMP,
            "num_th": cfg.MODEL.PY.NUM_TH,
            "GIOU": cfg.MODEL.FLhead.GIOU,
            "box_matcher": Matcher(cfg.MODEL.SELECTION.IOU_THRESHOLDS,cfg.MODEL.SELECTION.IOU_LABELS)}
        return ret

    def __call__(self,is_train,imagesize:torch.Tensor,FL_input:dict, val_ind=None, targets:Optional[List[Instances]] = None,device = None):
        """
        This mode is to genertate the FL level targets in accordance with the output of FL_head output(mask and delta). Meanwhile, output coresponding predicted boxes.

        FL_input: It contains mask([num_levels,1,img_h,img_w]) sigmoid values and delta([num_levels,img_h,img_w,2]).
        one_hot: False means multiple FL levels will be used to collect related boxes. Default set is True.

        Returns:FL_level_target: the levels indices in shape of [bs].pred_boxes: all predicted boxes with respect to chosen FL level. If the hot_one is not Ture, NMS filter will be added. In shape [bs,num_box,4], bs is batches, num_box represents FL levels and the related boxes coordinates(lt_x,lt_y,rb_x,rb,y).
        """
        #TODO: complete not one_hot part
        new_FL_input = dict()
        if self.one_hot:
            print(FL_input["mask"].shape,FL_input["delta"].shape)
            if val_ind is not None:
                new_FL_input["mask"] = FL_input["mask"][val_ind].squeeze().sigmoid()
                new_FL_input["delta"] = FL_input["delta"][val_ind]
            else:
                new_FL_input["mask"] = FL_input["mask"].squeeze().sigmoid()
                new_FL_input["delta"] = FL_input["delta"].squeeze()
            print("FL_inputcheck:",new_FL_input["mask"].shape,new_FL_input["delta"].shape)
            time0 = time.perf_counter()
            mask,delta,checkboard,img_height,img_width = pre_data(imagesize,new_FL_input,device=device,test=False)

            predbox = IS2box(mask,delta,checkboard,img_height,img_width,mask_th=self.mask_th)
            time1 = time.perf_counter()
            print("IS2box:",time1-time0,predbox.tensor.shape,predbox.device) #0.02202667703386396 torch.Size([30, 4]) #0.046
            del checkboard
            if not is_train:
                return predbox
            print("targets:",targets[0].gt_boxes.tensor.shape,targets[0].gt_classes.shape,targets[0].gt_classes) #['gt_boxes', 'gt_classes', 'gt_masks']
            cls_target = self.iou_generator(targets[0], predbox)
            return predbox,cls_target

        '''
        # mask = torch.squeeze(mask, 1).sigmoid()
        mask = F.softmax(torch.squeeze(mask,1))
        # print("input_shape:",mask.shape,delta.shape,imagesize,(mask.shape[1],mask.shape[2]))#[3, 800, 1303]) torch.Size([3, 800, 1303, 2]
        pred_boxes = []
        pred_boxes_lens = []
        for lv in range(self.num_FL):
            pred_boxes.append(IS2box(mask[lv],delta[lv],checkboard,img_height,img_width,self.mask_th,self.patch,self.pylayer,self.num_ignore,self.amp,self.num_th))
            pred_boxes_lens.append(pred_boxes[lv].tensor.shape[0])
            # print(pred_boxes_lens[lv]) #[-1,4]
            break #it is for not beyond gpu memory when pooling
        del checkboard
        assert len(pred_boxes_lens) != 0, "There is no predicted boxes!"
        pred_boxes = Boxes.cat(pred_boxes)
        print("pred_boxes_clip:",pred_boxes.tensor.shape) #[4, 4]
        pred_boxes.clip((mask.shape[1],mask.shape[2]))
        if not is_train:
            return pred_boxes

        iou_matrix,matched_ids,labels,cls_target = self.iou_generator(targets[0],pred_boxes)
        # print("iou shape check:",iou_matrix.shape,matched_ids.shape) #[4, 4]) torch.Size([4]
        # print("TEST:",pred_boxes.tensor.shape,iou_matrix.shape,labels.shape,cls_target.shape,cls_target[0:10]) #device='cuda:0'
        # lv_target = self.level_target_generator(pred_boxes_lens,iou_matrix,matched_ids,labels,self.one_hot)
        # print("lv_target:",lv_target)
        iou = iou_matrix[matched_ids,torch.arange(iou_matrix.shape[1])]
        gt_boxes = targets[0].gt_boxes.tensor[matched_ids]
        
        return 0,0,0,0 #pred_boxes,cls_target,iou,gt_boxes #,lv_target.to(pred_boxes.device)

        '''
    @torch.no_grad()
    def iou_generator(self,gt: Instances, pred_boxes: Boxes):
        """
        pred_boxes: It is the output from ISbackbon with shape of [-1,4]
        gt_box: [Xmin,Ymin,Xmax,Ymax]
        return:
            IOU_rate: the IOU rate with respect to corresponding boxes for selected focal levels.
            It will be used to compute GIOU loss in the following anxillary loss opearation.
            lv_label: the selected FL level as next layer's target.
            final_pred_boxes: THe pred boxes related to lv_label.
        """
        # print("box1 and box2 shape:",gt.gt_boxes.tensor.shape,pred_boxes.tensor.shape)
        #It will return [box1.shape,box2.shape]. In this case the output shape will be [gt.len,predbox.len]
        match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt.gt_boxes,pred_boxes)
        # print("iou_matrix:",match_quality_matrix.shape) #[10, 22]
        # matched_idxs has the predbox len. it will reture matched gt_boxes indices.
        matched_idxs,labels = retry_if_cuda_oom(self.matcher)(match_quality_matrix)
        print("matched_idxs:",matched_idxs.shape,labels.shape,matched_idxs,labels) #torch.Size([22]) torch.Size([22])
        gt_classes = gt.gt_classes
        has_gt = gt_classes.numel()>0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[labels==0] = self.num_class
        else:
            gt_classes = torch.zeros_like(matched_idxs)+self.num_class
        assert len(gt_classes)==len(pred_boxes),"gt_classes and pred_boxes are not matched!"
        del match_quality_matrix, matched_idxs, labels
        return gt_classes

    #NOTE: pay attention to the running speed. All operations here is under CUDA.
    def level_target_generator(self,predbox_len:list,iou_matrix:torch.Tensor,matched_id:torch.Tensor,label:torch.Tensor,one_hot=True):
        """
        Arguments:
            predbox_len: A list contains the predboxes num for each FL levels. It is used to divide the iou_matrix into to num_levels.
            iou_matrix: A list [N,M](N:gt,M:pred) concludes all iou rates between gt_box and pred_boxes.
            matched_id: A [M] len list where the predbox is the domain. Every element shows which gt_box index matches current predbox.
            label: If the matched iou is larger than threshold, '1' will be given, otherwise it is '0'. The len is [M]
        return:
            The level target out of num_lvl(default is 3)
        """
        levelcollection = OrderedDict()
        num_lv = len(predbox_len)
        # print("test_point0:",num_lv,predbox_len,matched_id.shape)
        lv_matched_id=[k for k in matched_id.split(predbox_len)]
        lv_matched_label=[j for j in label.split(predbox_len)]
        lv_iou = [i for i in iou_matrix.split(split_size=predbox_len,dim=1)]
        # print("test_point1:",len(lv_iou),type(lv_iou[0]),lv_iou[0].shape)
        lv_gt_num = []
        lv_iou_totoal = []
        for id in range(num_lv):
            group_id = lv_matched_id[id]
            group_label = lv_matched_label[id]
            group_iou = lv_iou[id]#[N',M']
            mask_label = group_label.type(torch.bool)
            valid_matched_indxs = group_id[mask_label]
            # print("valid_matched_indxs:",valid_matched_indxs.shape)#[1251]
            # print("predbox_len:",predbox_len[id])
            id_pred_len = torch.arange(0,predbox_len[id],dtype=torch.long)
            # print("id_len:",id_pred_len)
            # print("group_iou_shape:",type(group_iou),group_iou.shape,group_id.shape,group_id,id_pred_len,mask_label.shape)
            valid_iou = group_iou[group_id,id_pred_len][mask_label]
            assert valid_iou.shape[0] !=0, "level "+str(id)+" has an empty valid_iou!"
            # print("valid_iou:",type(valid_iou),valid_iou.shape,valid_iou)
            valid_gt = torch.unique(valid_matched_indxs)
            # print("valid_gt:",valid_gt.shape,valid_gt,valid_gt.device)
            valid_gt_num = valid_gt.numel()
            # print("valid_gt_num:",valid_gt_num)
            #Here, the alternative computation of max iou can be mean of valid_pred_len
            lv_gt_num.append(valid_gt_num)
            max_iou=0
            for i in valid_gt:
                # print("valid_matched_idx:",valid_matched_indxs)
                valid_index = torch.where(valid_matched_indxs==i)
                # print("valid_index:",valid_index)
                valid_ind_iou = valid_iou[valid_index].max()
                # print("valid_ind_iou:",valid_ind_iou)
                max_iou += valid_ind_iou
            # print("max_iou:", max_iou)
            lv_iou_totoal.append(max_iou) #cuda
            # levelcollection[str(id)]=dict(gt_num = valid_gt_num,totaliou = max_iou.item())#using item bc the cuda device property
            # print(levelcollection)
        assert len(lv_gt_num)==len(lv_iou_totoal),"num of lv_gt and lv_iou_sum is not matched!"
        # print("test:",lv_gt_num,lv_iou_totoal)
        #There might be chance lv_gt_num is [0,0,0]
        lv_gt_num = torch.tensor(lv_gt_num)
        # print("lv_gt_num:",len(lv_gt_num),lv_gt_num)#3
        lv_target = torch.where(lv_gt_num==lv_gt_num.max())[0]
        lv_iou_totoal = torch.tensor(lv_iou_totoal)
        # print("test1",lv_gt_num,lv_target,lv_iou_totoal)
        # print("test2",lv_iou_totoal[lv_target])
        if one_hot and lv_target.shape[0] !=1:
            sum = lv_iou_totoal[lv_target]
            # print("sum:",sum,type(sum)) #tensor(2.9963, device='cuda:0') <class 'torch.Tensor'>
            max_id = torch.argmax(sum)
            lv_target = lv_target[max_id]
        #lv_target new will convert the dtype from torch int64 to float32
        lv_target = torch.zeros(num_lv).scatter_(0,lv_target,1)

        return lv_target












