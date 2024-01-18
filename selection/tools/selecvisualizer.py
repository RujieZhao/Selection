import time
import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.utils.colormap import random_color
from ..util import IS2box,pre_data,tonumpy

__all__=["IS_Visualizer","pyramid_Visualizer"]

class IS_Visualizer(Visualizer):
    def __init__(self,IS_target,img_rgb, metadata=None,scale=1.0,_SMALL_OBJECT_AREA_THRESH=1000,test=True):
        super(IS_Visualizer,self).__init__(img_rgb, metadata=metadata, scale=scale)
        self._SMALL_OBJECT_AREA_THRESH = _SMALL_OBJECT_AREA_THRESH
        if test:
            self.IS_contour_x,self.IS_contour_y,self.IS_center_x,self.IS_center_y = self.convert_coordinate(
                IS_target.selecgt_mask,IS_target.selecgt_delta)
        else:
            self.IS_contour_x, self.IS_contour_y, self.IS_center_x, self.IS_center_y = self.convert_coordinate(
                tonumpy(IS_target.selecgt_mask), tonumpy(IS_target.selecgt_delta))
        # print("test_type:",type(self.IS_center_x))
        # print("IS contour:", len(self.IS_contour_x), self.IS_contour_x, self.IS_contour_y)

    def convert_coordinate(self,IS_contour,IS_delta):
        IS_coordinate = np.where(IS_contour>0.5)
        coor_height = IS_coordinate[0]
        coor_width = IS_coordinate[1]

        # valid_delta = len(set(np.where(IS_delta[coor_height,coor_width,0]!=0)[0].tolist()+np.where(IS_delta[coor_height,coor_width,1]!=0)[0].tolist()))
        # valid_mask = len(IS_coordinate[0])
        # print(valid_delta,valid_mask)
        #
        # assert valid_delta==valid_mask

        delta_height = IS_delta[coor_height,coor_width,0]
        delta_width = IS_delta[coor_height,coor_width,1]
        # print("type check:",type(coor_width),type(delta_width))
        center_x = coor_width - delta_width
        center_y = coor_height - delta_height
        return coor_width,coor_height,center_x,center_y

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)

        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                        instance_area < self._SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )
        for x,y in zip(self.IS_contour_x,self.IS_contour_y):
            self.draw_circle((x,y),color=(0,0,0),radius=2)
        for d_x,d_y in zip(self.IS_center_x,self.IS_center_y):
            self.draw_circle((d_x,d_y),color=(1.,1.,0),radius=2)
        return self.output

class pyramid_Visualizer(Visualizer):
    def __init__(self,IS_target,img_rgb, metadata=None,scale=1.0,_SMALL_OBJECT_AREA_THRESH=1000,test=False, img_size=None, mask_th=0.85):
        super(pyramid_Visualizer,self).__init__(img_rgb, metadata=metadata, scale=scale)
        self._SMALL_OBJECT_AREA_THRESH = _SMALL_OBJECT_AREA_THRESH
        self.mask_th = mask_th
        if test:
            if img_size is not None:
                self.img_size = tonumpy(img_size)
            else:
                # print(type(IS_target.selecgt_mask)) #<class 'numpy.ndarray'>
                self.img_size = np.array(IS_target.selecgt_mask.shape)/2.
            self.IS_contour_x,self.IS_contour_y,self.IS_center_x,self.IS_center_y = self.convert_coordinate(IS_target.selecgt_mask,IS_target.selecgt_delta,test=test)

        else:
            if img_size is not None:
                self.img_size = tonumpy(img_size)
            else:
                self.img_size = np.array(IS_target["mask"].shape)/2.
            self.IS_contour_x, self.IS_contour_y, self.IS_center_x, self.IS_center_y = self.convert_coordinate(IS_target["mask"], IS_target["delta"],test=test)

        # print("img_size in selev:",img_size) #[ 704 1146]
        # print("IS center:",len(self.IS_contour_x),self.IS_center_x,self.IS_center_y)
        time0 = time.perf_counter()
        mask,delta,checkboard,img_height,img_width = pre_data(self.img_size,IS_target,test=test)
        time1 = time.perf_counter()
        print("pre_data_time:",time1-time0)
        # print("input_type:", type(mask),mask.shape,mask.device, type(delta),delta.shape,delta.device) #<class 'torch.Tensor'> torch.Size([959, 640]) cuda:0 <class 'torch.Tensor'> torch.Size([959, 640, 2]) cuda:0
        time1 = time.perf_counter()

        # print("vis_mask:",mask.shape)
        # print(mask[100:120,100])
        # print("vis_delta:",delta.shape)
        # print(delta[100:120,100,1])
        # print("vis_size:",img_height,img_width)
        # print("vis_checkboard:",checkboard.shape)
        # print("vis_th:",self.mask_th)


        self.predict_box=IS2box(mask,delta,checkboard,img_height,img_width,mask_th=self.mask_th).tensor
        time2 = time.perf_counter()
        print("IS2box_time:",time2-time1)
        print("total_operation_time:",time2-time0)
        print("pred_box:",self.predict_box.shape,self.predict_box)
    def convert_coordinate(self,IS_contour,IS_delta,test=True):

        # print(type(IS_contour),IS_contour.shape,type(IS_delta),IS_delta.shape) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
        IS_coordinate = np.where(IS_contour>self.mask_th)
        # print("vis val mask check:",IS_coordinate)
        coor_height = IS_coordinate[0]
        coor_width = IS_coordinate[1]
        if not test:
            IS_delta = IS_delta*self.img_size
        delta_height = IS_delta[coor_height,coor_width,0]
        delta_width = IS_delta[coor_height,coor_width,1]
        # print("type check:", type(coor_width),coor_width.shape, type(delta_width),delta_width.shape)# <class 'numpy.ndarray'> (290,) <class 'numpy.ndarray'> (290,)
        center_x = coor_width - delta_width
        center_y = coor_height - delta_height
        return coor_width,coor_height,center_x,center_y

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors:float=None,
        alpha=0.5,
        test_colors=[1.,0.,0.]
    ):
        # print("param check:",boxes.shape,labels,masks,keypoints)
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
            # print("num_instances:",num_instances)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)

        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                # print("assigned_color:",assigned_colors)
                self.draw_box(boxes[i], edge_color=test_colors) #color

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                        instance_area < self._SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )
        for x,y in zip(self.IS_contour_x,self.IS_contour_y):
            # print("type check:",x.dtype,type(x)) #int64 <class 'numpy.int64'>
            self.draw_circle((x,y),color=(0,0,0),radius=2)
        for d_x,d_y in zip(self.IS_center_x,self.IS_center_y):
            self.draw_circle((d_x,d_y),color=(1.,1.,0),radius=2)
        # selecbox = np.asarray(self.predict_box.cpu())
        # num_selecbox = len(selecbox)
        # for s in range(num_selecbox):
        #     self.draw_box(selecbox[s],edge_color=[1.,0,0])
        return self.output

















