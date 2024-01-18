

import numpy as np
import torch

def sizecreator(size:int)->dict:
	sizeoutput = {}
	if isinstance(size, int):
		sizeoutput["mask"]=torch.zeros((size,size))
		sizeoutput["center"]=size//2
		return sizeoutput

def shapecreator(mask:torch.Tensor,center:int,type:str):
	i = center+1
	bound = mask.shape[0]
	if isinstance(type,str):
		if type=="cross":
			num = 2*(bound-1)
			while i<bound:
				mask[i,center]=1
				mask[center,i]=1
				mask[(i-center-1),center]=1
				mask[center,(i-center-1)] = 1
				i += 1
		if type=="star":
			num = 4*(bound-1)
			while i<bound:
				mask[i,center] =1
				mask[(i-center-1),center]=1
				mask[center,i]=1
				mask[center,(i-center-1)]=1
				mask[i,i]=1
				mask[(i-center-1),(i-center-1)]=1
				mask[(i-center-1),(bound-i+center)]=1
				mask[(bound-i+center),(i-center-1)]=1
				i+=1
		if type=="contr":
			num = bound*bound-1
			mask[:,:]=1
			mask[center,center]=0
	return mask,num

def maskcreate(size=5,type="star"):
	if not type in {"cross","star","contr"}:
		raise ValueError("type can not be found")

	sizeoutput = sizecreator(size)
	mask = sizeoutput["mask"]
	center = sizeoutput["center"]
	mask,num = shapecreator(mask,center,type)
	return mask,num

# class maskcreate:
# 	def __init__(self,size=5,type="star"):
# 		self.size = size
# 		self.type = type

# 	def __call__(self, *args, **kwargs):
# 		sizeoutput=sizecreator(self.size)
# 		mask = sizeoutput["mask"]
# 		center = sizeoutput["center"]
# 		mask = shapecreator(mask,center,self.type)
# 		return mask

# if __name__=="__main__":
	# size=31
	# output = sizecreator(size)
	# print(len(output),type(output),output["mask"].shape)
	# mask=output["mask"]
	# center = output["center"]
	# mask = shapecreator(mask,center,type="contr")

	# mask = maskcreate(31,"contr")
	# print(len(mask),mask[1],mask[0].shape)






















