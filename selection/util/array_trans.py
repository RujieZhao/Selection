import torch
import numpy as np

torch.cuda.set_device(0)
# device = torch.device("cuda:0" if use_cuda else "cpu")

def tonumpy(data):
	if isinstance(data,np.ndarray):
		return data
	if isinstance(data,torch.Tensor):
		return data.cpu().detach().numpy()

def totensor(data,cuda=True,test=False):
	"""
	if it used in test mode, then data required numpy ndarray in the plotting process(test_IStarget_visualizer.py).
	So the return will not change data's type.
	"""
	if test:
		# print("data type:",type(data)) #class 'numpy.ndarray'
		return data
	else:
		if isinstance(data,np.ndarray):
			ten = torch.from_numpy(data)
			if ten.device==torch.device("cpu") and cuda:
				ten = ten.cuda()
		if isinstance(data,torch.Tensor):
			ten = data.detach()#target mask/delta does not need gradient.
			if ten.device==torch.device("cpu") and cuda:
				ten = ten.cuda()
		ten = ten.to(dtype = torch.float32)
	return ten

def pre_data(imagesize,data,device=None,test=False):
	"""
	convert numpy array to torch cuda GPU. At mean time, build up grid corrdinators for Pyramid operation to prevent duplication execution in levels' loop.

	data: It is the IS backbone output and will be used as Pyramid network's input. It contains "selecgt_mask" and "selecgt_delta" two categories.
	device: default is cuda:0.

	return:
		mask and delta for next gpu operation.
		checkboard: grid corrdinators
		img_height,img_width
	"""
	# print("input_key_test:",len([*data]))
	# assert len([*data])==2, "augments data should contain 2 keys!"
	# print("test:",test)
	if test: # direct from anno
		mask = data.selecgt_mask
		delta = data.selecgt_delta
	else:
		# print([*data]) #['mask', 'delta']
		mask = data[[*data][0]]
		delta = data[[*data][1]] #(704, 1146, 2) <class 'numpy.ndarray'> float32
		# print("pre_data:",delta.shape,type(delta),delta.dtype,imagesize,imagesize.dtype)
		delta = delta * imagesize
		# print("delatcheck:",delta.shape,delta[0:5]) #(704, 1146, 2)

	# print("shapecheck:",mask.shape,delta.shape,mask.dtype) #[3, 1, 800, 1303]) torch.Size([3, 800, 1303, 2]) torch.float32
	# print("device_check:",mask.device, imagesize.device,imagesize.shape) #([3, 800, 1303, 2]

	if not isinstance(mask,torch.Tensor):
		mask = torch.from_numpy(mask)
		delta = torch.from_numpy(delta)
		# imagesize = torch.tensor(imagesize)

	assert mask.device == delta.device, "mask and delta device is not mathched!" # == imagesize.device
	img_height,img_width = mask.shape[-2:]
	height = torch.arange(0,img_height)
	width = torch.arange(0,img_width)
	grid_x,grid_y = torch.meshgrid(height,width,indexing="ij")
	checkboard = torch.stack([grid_x,grid_y],dim=-1)
	# print("checkboard:",checkboard.type(),checkboard.shape,checkboard[2,3]) #torch.LongTensor torch.Size([640, 790, 2]) tensor([2, 3])

	if not mask.is_cuda:
		# print("outputmask_is_cuda:",True)
		use_cuda = torch.cuda.is_available()
		assert use_cuda, "GPU is not available!"
		if device != None:
			device = torch.device(device if use_cuda else "cpu")
			mask = mask.to(device)
			delta = delta.to(device)
		else:
			mask = mask.cuda()
			delta = delta.cuda()
	# if test:
	# 	imagesize = imagesize.to(mask.device)
	# print(type(mask),mask.device,delta.device,imagesize.device)

	return mask,delta,checkboard,img_height,img_width



# if __name__ == "__main__":
# 	a = torch.tensor([1,2,34,5])
# 	b = totensor(a)
# 	print(b)






















