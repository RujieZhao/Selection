import torch
from ..maskcreate import maskcreate
from .. import selcuda
from detectron2.config import configurable
import time

class pixel_perception:
    """
    This processing is the very beginning of human-like visual system which is sensing the contrast of pixels, adjusting focal length and having a descendant concept for the graph.
    """
    @configurable
    def __init__(self,*,maskarea,masktype,lv_num,delta,ratio):
        self.maskarea = maskarea
        self.masktype = masktype
        self.lv_num = lv_num
        self.delta = delta
        self.ratio = torch.as_tensor([ratio],dtype=torch.float32)
        # print(self.ratio)
        self.mask,self.num = maskcreate(self.maskarea,self.masktype)
    @classmethod
    def from_config(cls,cfg):
        ret = {"maskarea": cfg.MODEL.GPU.MA,
               "masktype": cfg.MODEL.GPU.MT,
               "lv_num":   cfg.MODEL.GPU.FL,
               "delta":    cfg.MODEL.GPU.DEL,
               "ratio":    cfg.MODEL.GPU.RAT
               }
        return ret

    def __call__(self,img:torch.Tensor):
        assert img.type()=="torch.cuda.FloatTensor","input image has wrong type!"
        # mask,num = maskcreate(self.maskarea,self.masktype)
        # time8 = time.perf_counter()
        mask = self.mask.to(img.device)
        ratio = self.ratio.to(img.device)
        # time7 = time.perf_counter()
        # print(time7-time8)#4.889300180366263e-05
        # time9=time.perf_counter()
        # test = selcuda.selection(img,mask,self.num,self.delta,ratio) # 5.1196002459619194e-05
        # print("img:",img.shape) #[704, 1146, 3]
        selecout = torch.permute(selcuda.selection(img,mask,self.num,self.delta,ratio),(0,3,1,2)).contiguous()
        # time99 = time.perf_counter()
        # print("selcuda time:",time99-time9) #8.145199899445288e-05 5.5170999985421076e-05

        # print(selecout.shape,selecout.device,selecout.dtype)
        del mask
        del ratio
        return selecout








































