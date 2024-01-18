import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .build import SELEIS_BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
sys.path.append("../")
# from mmcv_custom import load_checkpoint

__all__=["build_is_backbone","PatchEmbed","BasicLayer","SwinTransformerBlock","WindowAttention","Mlp","window_reverse","window_partition","PatchMerging","swin"]

class Mlp(nn.Module):
	def __init__(self, in_feature, hidden_features = None, out_features = None, act_layer=nn.GELU, drop=0.):
		super(Mlp, self).__init__()
		output_feature = out_features or in_feature
		hidden_features = hidden_features or in_feature
		self.fc1 = nn.Linear(in_feature,hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features,output_feature)
		self.drop = nn.Dropout(drop)

	def forward(self,x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x
def window_partition(x,window_size):
	"""
	Args:
	x: (B, H, W, C)
	window_size (int): window size

	Returns:
	windows: (num_windows*B, window_size, window_size, C)
	"""
	B, H, W, C = x.shape
	# print(x.shape,window_size) #torch.Size([1, 115, 160, 1]) 7
	x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
	windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
	return windows

def window_reverse(windows, window_size, H, W):
	"""
	Args:
	windows: (num_windows*B, window_size, window_size, C)
	window_size (int): Window size
	H (int): Height of image
	W (int): Width of image

	Returns:
	x: (B, H, W, C)
	"""
	B = int(windows.shape[0]/(H * W/ window_size/ window_size))
	x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
	x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
	return x

class WindowAttention(nn.Module):
	r""" Window based multi-head self attention (W-MSA) module with relative position bias.
	It supports both of shifted and non-shifted window.

	Args:
	dim (int): Number of input channels.
	window_size (tuple[int]): The height and width of the window.
	num_heads (int): Number of attention heads.
	qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
	qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
	attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
	proj_drop (float, optional): Dropout ratio of output. Default: 0.0
	"""
	def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.dim = dim
		self.window_size = window_size
		self.num_head = num_heads
		head_dim = dim//num_heads
		self.scale = qk_scale or head_dim**-0.5

		self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1),num_heads))

		coords_h = torch.arange(self.window_size[0])
		coords_w = torch.arange(self.window_size[1])
		coords = torch.stack(torch.meshgrid([coords_h,coords_w])) # 2,WH,WW
		coords_flatten = torch.flatten(coords,1)
		relative_coords = coords_flatten[:,:,None]-coords_flatten[:,None,:]
		relative_coords = relative_coords.permute(1,2,0).contiguous()
		relative_coords[:,:,0] += self.window_size[0] -1
		relative_coords[:, :, 1] += self.window_size[1] - 1
		relative_coords[:,:,0] *= 2*self.window_size[1] - 1
		relative_position_index = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
		self.register_buffer("relative_position_index", relative_position_index)

		self.qkv = nn.Linear(dim,dim*3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim,dim)
		self.proj_drop = nn.Dropout(proj_drop)
		trunc_normal_(self.relative_position_bias_table, std=.02)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self,x, mask=None):
		"""
		Args:
			x: input features with shape of (num_windows*B, N, C)
			mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
		"""
		B_, N, C = x.shape # N: number of all patches in one window
		qkv = self.qkv(x).reshape(B_, N, 3, self.num_head, C//self.num_head).permute(2,0,3,1,4)
		q,k,v = qkv[0], qkv[1], qkv[2]
		q = q*self.scale
		attn = (q@k.transpose(-2,-1))
		relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
			self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
		attn = attn+relative_position_bias.unsqueeze(0)

		if mask is not None:
			nW = mask.shape[0]
			attn = attn.view(B_//nW,nW,self.num_head,N,N) + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1,self.num_head,N,N)
			attn = self.softmax(attn)
		else:
			attn = self.softmax(attn)
		attn = self.attn_drop(attn)
		x = (attn @ v).transpose(1,2).reshape(B_, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

	# def extra_repr(self) -> str:
	# 	return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_head}"
	#
	# def flops(self,N):
	# 	# calculate flops for 1 window with token length of N
	# 	flop = 0
	# 	flop += N * self.dim * 3 * self.dim
	# 	flop += self.num_head * N * (self.dim//self.num_head) * N
	# 	flop += self.num_head * N * N * (self.dim // self.num_head)
	# 	flop += N * self.dim * self.dim
	# 	return flop

class SwinTransformerBlock(nn.Module):
	r""" Swin Transformer Block.

	Args:
	dim (int): Number of input channels.
	# input_resolution (tuple[int]): Input resulotion.
	num_heads (int): Number of attention heads.
	window_size (int): Window size.
	shift_size (int): Shift size for SW-MSA.
	mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
	qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
	qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
	drop (float, optional): Dropout rate. Default: 0.0
	attn_drop (float, optional): Attention dropout rate. Default: 0.0
	drop_path (float, optional): Stochastic depth rate. Default: 0.0
	act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
	norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
	"""
	def __init__(self, dim, num_head, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer = nn.LayerNorm):
		super().__init__()
		# self.input_resolution = input_resolution
		self.num_heads = num_head
		self.window_size = window_size
		self.shift_size = shift_size
		self.mlp_ratio = mlp_ratio

		assert 0<= self.shift_size < self.window_size, "shift_size must in 0-window_size"
		self.norm1 = norm_layer(dim)
		self.attn = WindowAttention(dim, window_size=to_2tuple(window_size), num_heads=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
		self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_feature=dim,hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
		self.H = None
		self.W = None

	def forward(self,x, mask_matrix):
		# H,W = self.input_resolution
		B,L,C = x.shape
		H,W = self.H, self.W
		assert L == H*W, "input feature has wrong size"
		shortcut = x
		x = self.norm1(x)
		x = x.view(B,H,W,C)

		pad_l = pad_t = 0
		pad_r = (self.window_size - W % self.window_size) % self.window_size
		pad_b = (self.window_size-H % self.window_size) % self.window_size
		x = F.pad(x,(0,0,pad_l,pad_r,pad_t,pad_b))
		_,Hp,Wp,_ = x.shape

		if self.shift_size > 0:
			shifted_x = torch.roll(x,shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
			attn_mask = mask_matrix
		else:
			shifted_x = x
			attn_mask = None

		x_windows = window_partition(shifted_x, self.window_size)# nW*B, window_size, window_size, C
		x_windows = x_windows.view(-1,self.window_size*self.window_size,C)
		attn_windows = self.attn(x_windows, mask=attn_mask)
		attn_windows = attn_windows.view(-1,self.window_size,self.window_size,C)
		shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
		# print("shifted_x:",shifted_x.shape)
		if self.shift_size >0:
			x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
		else:
			x = shifted_x

		if pad_r > 0 or pad_b >0:
			x = x[:,:H,:W,:].contiguous()
		x = x.view(B, H*W, C)
		x = shortcut + self.drop_path(x)
		x = x + self.drop_path(self.mlp(self.norm2(x)))

		return x

	# def extra_repr(self) -> str:
	# 	return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "\
	# 			f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
	#
	# def flops(self):
	# 	flops = 0
	# 	H, W = self.input_resolution
	# 	flops += self.dim*H*W
	# 	nW = H * W / self.window_size / self.window_size
	# 	flops += nW * self.attn.flops(self.window_size * self.window_size)
	# 	flops += self.dim * H * W
	# 	return flops

class PatchMerging(nn.Module):
	r""" Patch Merging Layer.

	Args:
	input_resolution (tuple[int]): Resolution of input feature.
	dim (int): Number of input channels.
	norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
	"""
	def __init__(self, dim, norm_layer=nn.LayerNorm):
		super().__init__()
		# self.input_resolution = input_resolution
		self.dim = dim
		self.reduction = nn.Linear(4*dim,2*dim,bias=False)
		self.norm = norm_layer(4*dim)

	def forward(self,x, H, W):
		B,L,C = x.shape
		assert L == H*W, "input feature has wrong size"
		x = x.view(B,H,W,C)
		pad_input = (H % 2 == 1) or (W % 2 == 1)
		if pad_input:
			x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
		x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
		x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
		x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
		x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
		x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
		# print("x0-1:",x0.shape,x1.shape,x2.shape,x3.shape,x.shape) #torch.Size([4, 100, 150, 96]) torch.Size([4, 100, 150, 96]) torch.Size([4, 100, 150, 96]) torch.Size([4, 100, 150, 96]) torch.Size([4, 100, 150, 384])

		x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

		x = self.norm(x)
		x = self.reduction(x)
		return x

	# def extra_repr(self) -> str:
	# 	return f"input_resolution={self.input_resolution}, dim={self.dim}"
	#
	# def flops(self):
	# 	H,W = self.input_resolution
	# 	flops = H*W*self.dim
	# 	flops += (H//2) * (W//2) * 4 * self.dim * 2 * self.dim
	# 	return flops

class BasicLayer(nn.Module):
	def __init__(self, dim, depth, num_head, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
		super(BasicLayer, self).__init__()
		self.window_size = window_size
		self.dim = dim
		# self.depth = depth
		self.use_checkpoint = use_checkpoint
		self.shift_size = window_size // 2
		self.blocks = nn.ModuleList([
			SwinTransformerBlock(dim=dim,num_head=num_head, window_size=window_size, shift_size=0 if (i%2==0) else window_size//2, mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, qk_scale=qk_scale,drop=drop,attn_drop=attn_drop,drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,norm_layer=norm_layer) for i in range(depth)
		])
		if downsample is not None:
			self.downsample = downsample(dim=dim, norm_layer=norm_layer)
		else:
			self.downsample = None

	def forward(self,x, H, W):
		# print("H,W,x:",H,W,x.shape) #200 299 torch.Size([4, 59800, 96])
		Hp = int(np.ceil(H/self.window_size))*self.window_size
		Wp = int(np.ceil(W/self.window_size))*self.window_size
		img_mask = torch.zeros((1,Hp,Wp,1),device=x.device)
		h_slices = (slice(0, -self.window_size),slice(-self.window_size,-self.shift_size),slice(-self.shift_size,None))
		w_slices = (slice(0, -self.window_size),slice(-self.window_size, -self.shift_size),slice(-self.shift_size,None))
		cnt = 0
		for h in h_slices:
			for w in w_slices:
				img_mask[:,h,w,:] = cnt
				cnt += 1
		mask_windows = window_partition(img_mask, self.window_size) # nW, window_size, window_size, 1
		mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
		attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
		attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

		for blk in self.blocks:
			blk.H, blk.W = H, W
			if self.use_checkpoint:
				x = checkpoint.checkpoint(blk,x, attn_mask)
			else:
				x = blk(x, attn_mask)
		if self.downsample is not None:
			x_down = self.downsample(x, H, W)
			Wh, Ww = (H + 1) // 2, (W + 1) // 2 #add one to make sure thery are larger than 1 in path merging
			return x, H, W, x_down, Wh, Ww
		else:
			return x, H, W, x, H, W

class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
		super().__init__()
		patch_size = to_2tuple(patch_size)
		self.patch_size = patch_size

		self.in_chans = in_chans
		self.embed_dim = embed_dim

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
		if norm_layer is not None:
			self.norm = norm_layer(embed_dim)
		else:
			self.norm = None

	def forward(self, x):
		"""Forward function."""
		# padding
		# print("x:",x.shape) #torch.Size([3, 3, 800, 1202])
		_, _, H, W = x.size()
		# print("patch_size:",self.patch_size)#4
		if W % self.patch_size[1] != 0:
			x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
		if H % self.patch_size[0] != 0:
			x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
		# print("embeded_pad_shape:",x.shape)
		x = self.proj(x)  # B C Wh Ww
		# print("x_shape:",x.shape) #torch.Size([3, 96, 200, 301])
		if self.norm is not None:
			Wh, Ww = x.size(2), x.size(3)
			x = x.flatten(2).transpose(1, 2)
			x = self.norm(x)
			x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
		# print("final_x_shape:", x.shape)#torch.Size([3, 96, 200, 301])
		return x

	# def flops(self):
	# 	Ho,Wo = self.patches_resolution
	# 	flops = Ho * Wo * self.embed_dim * self.in_chans * self.patch_size[0] * self.patch_size[1]
	# 	if self.norm is not None:
	# 		flops += Ho * Wo * self.embed_dim
	# 	return flops

class swin(Backbone):
	"""
	patch_size change to 1 from 4
	"""
	def __init__(self,FLbackbone=None,pretrain_img_size=224,patch_en=True,  patch_size=4, in_chans=3, num_classes=1000,embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,norm_layer=nn.LayerNorm, ape=False, patch_norm=True,out_indices=(0,1,2,3), frozen_stages=-1,use_checkpoint=False,):
		super().__init__()
		self.FLbackbone = FLbackbone
		self.pretrain_img_size = pretrain_img_size
		self.patch_size = patch_size
		# self.num_classes = num_classes
		self.num_layers = len(depths)
		self.embed_dim = embed_dim
		self.ape = ape
		self.patch_norm = patch_norm
		self.out_indices = out_indices
		self.frozen_stages = frozen_stages
		self.stride = [4,8,16,32]
		# self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
		# self.mlp_ratio = mlp_ratio

		if patch_en:
			self.patch_embed = PatchEmbed(
				patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
				norm_layer=norm_layer if self.patch_norm else None)
		# absolute position embedding (224)
		if self.ape:
			pretrain_img_size = to_2tuple(pretrain_img_size)
			patch_size = to_2tuple(patch_size)
			patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

			self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
			trunc_normal_(self.absolute_pos_embed, std=.02)

		self.pos_drop = nn.Dropout(p=drop_rate)
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate,sum(depths))]
		self.layers = nn.ModuleList()
		for i_layer in range(self.num_layers):
			layer = BasicLayer(dim=int(embed_dim*(2**i_layer)),depth=depths[i_layer],num_head=num_heads[i_layer],window_size=window_size,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],norm_layer=norm_layer,downsample=PatchMerging if (i_layer<self.num_layers-1) else None, use_checkpoint = use_checkpoint)
			self.layers.append(layer)

		num_features = [int(embed_dim*2**i) for i in range(self.num_layers)]
		self.num_features = num_features

		for i_layer in out_indices:
			layer = norm_layer(num_features[i_layer])
			layer_name = f'norm{i_layer}'
			self.add_module(layer_name,layer)

		self._freeze_stages()

	# self.avgpool = nn.AdaptiveAvgPool1d(1)
	# self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
	# 	self.apply(self.init_weights)

	def output_shape(self):
		return {
			layer: ShapeSpec(
				channels= self.num_features[layer],
				stride=self.stride[layer],
			) for layer in range(self.num_layers)
		}

	def _freeze_stages(self):
		if self.frozen_stages >= 0:
			self.patch_embed.eval()
			for param in self.patch_embed.parameters():
				param.requires_grad = False
		if self.frozen_stages >= 1 and self.ape:
			self.absolute_pos_embed.requires_grad = False
		if self.frozen_stages >= 2:
			self.pos_drop.eval()
			for i in range(0,self.frozen_stages-1):
				m = self.layers[i]
				m.eval()
				for param in m.parmeters():
					param.requires_grad=False

	def init_weights(self,pretrain=None):
		def _init_weights(m):
			if isinstance(m,nn.Linear):
				trunc_normal_(m.weight,std=0.02)
				if isinstance(m,nn.Linear) and m.bias is not None:
					nn.init.constant_(m.bias,0)
			elif isinstance(m,nn.LayerNorm):
				nn.init.constant_(m.bias,0)
				nn.init.constant_(m.weight,1.0)

		# if isinstance(pretrain,str):
		# 	self.apply(_init_weights)
		# 	load_checkpoint(self,pretrain)
		# elif pretrain is None:
		# 	self.apply(_init_weights)
		# else:
		# 	raise TypeError("pretrain must be a str or none")

	# @property
	# def size_divisibility(self) -> int:
	# 	return self._size_divisibility

	def forward(self, data_IS,data_FL,train_stage):
		# print("IS_input_data:",data_IS.shape)
		x = self.patch_embed(data_IS) #torch.Size([4, 96, 200, 301])
		# print("x:",x.shape) #torch.Size([4, 96, 200, 299])
		Wh,Ww = x.size(2), x.size(3)
		if self.ape:
			absolute_pos_embed = F.interpolate(self.absolute_pos_embed,size=(Wh,Ww),mode="bicubic")
			x = (x+absolute_pos_embed).flatten(2).transpose(1,2)
		else:
			x = x.flatten(2).transpose(1,2)
		x = self.pos_drop(x) #torch.Size([4, 64000, 96])
		outs =[]
		res = {}
		for i in range(self.num_layers):
			# print("layers:",i)
			layer = self.layers[i]
			x_out, H, W, x, Wh, Ww = layer(x,Wh,Ww)
			# print(i,"basic_output:",x_out.shape,H,W,x.shape,Wh,Ww)
			if i in self.out_indices:
				norm_layer = getattr(self, f"norm{i}")
				x_out = norm_layer(x_out)
				out = x_out.view(-1,H,W,self.num_features[i]).permute(0,3,1,2).contiguous()
				# print("output "+str(i)+" shape:",out.shape)
				outs.append(out) #torch.Size([4, 15000, 192]) torch.Size([4, 3750, 384]) torch.Size([4, 950, 768]) torch.Size([4, 950, 768])
		res["ISbackbone"]=outs
		if train_stage==1:
			res["FLbackbone"]=self.FLbackbone(data_FL)
		return res#tuple(outs)

	def train(self, mode=True):
		super(swin,self).train(mode)
		self._freeze_stages()

	# def output_shape(self):
	# 	return{
	# 		"swin": ShapeSpec(channels=self.num_features)
	# 	}
	#
	# def forward_features(self,x):
	# 	x = self.patch_embed(x)
	# 	if self.ape:
	# 		x=x+self.absolute_pos_embed
	# 	x = self.pos_drop(x)
	# 	for layer in self.layers:
	# 		x = layer(x)
	# 	x = self.norm(x) # B L C
	# 	x = self.avgpool(x.transpose(1,2)) # B C 1
	# 	x = torch.flatten(x,1) # B C
	# 	return x
	#
	# def forward(self,x):
	# 	x = self.forward_features(x)
	# 	x = self.head(x) # x.shape = (B, num_classes)
	# 	return x

	# def flops(self):
	# 	flops = 0
	# 	flops += self.patch_embed.flops()
	# 	for i,layer in enumerate(self.layers):
	# 		flops += layer.flops()
	# 	flops += self.num_features*self.patches_resolution[0] * self.patches_resolution[1] // (2**self.num_layers)
	# 	flops +=self.num_features * self.num_classes
	# 	return flops

@SELEIS_BACKBONE_REGISTRY.register()
def build_selec_backbone(cfg,input_shape):
	FLbackbone = build_resnet_fpn_backbone(cfg,input_shape)
	model = swin(FLbackbone=FLbackbone,pretrain_img_size=cfg.MODEL.BACKBONE.PRETRAIN_IMAGE_SIZE,
			patch_en=cfg.MODEL.BACKBONE.PATCH_EN,
			patch_size=cfg.MODEL.BACKBONE.PATCH_SIZE,
			in_chans=cfg.MODEL.BACKBONE.IN_CHAN,
			num_classes=cfg.MODEL.BACKBONE.NUM_CLASSES,
			embed_dim=cfg.MODEL.BACKBONE.EMBED_DIM,
			depths=cfg.MODEL.BACKBONE.DEPTHS,
			num_heads=cfg.MODEL.BACKBONE.NUM_HEAD,
			window_size=cfg.MODEL.BACKBONE.WINDOW_SIZE,
			mlp_ratio=cfg.MODEL.BACKBONE.MLP_RATIO,
			qkv_bias=cfg.MODEL.BACKBONE.QKV_BIAS,
			qk_scale=cfg.MODEL.BACKBONE.QK_SCALE,
			drop_rate=cfg.MODEL.BACKBONE.DROP_RATE,
			attn_drop_rate= cfg.MODEL.BACKBONE.ATTN_DROP_RATE,
			drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE,
			ape=cfg.MODEL.BACKBONE.APE,
			patch_norm=cfg.MODEL.BACKBONE.PATCH_NORM,
			out_indices=cfg.MODEL.BACKBONE.OUT_INDICES,
			frozen_stages= cfg.MODEL.BACKBONE.FROZEN_STAGES,
			use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT)
	return model



'''
mmdbackbone=MMDetBackbone(
		backbone=dict(
			type="DetectoRS_ResNet",
			conv_cfg=dict(type="ConvAWS"),
			sac=dict(type="SAC", use_deform=True),
			stage_with_sac=(False, True, True, True),
			depth=50,
			num_stages=4,
			out_indices=(0, 1, 2, 3),
			frozen_stages=1,
			norm_cfg=dict(type="BN", requires_grad=True),
			norm_eval=True,
			style="pytorch",
		),
		neck=dict(
			type="FPN",
			in_channels=[256, 512, 1024, 2048],
			out_channels=256,
			num_outs=5,
		),
		# skip pretrained model for tests
		# pretrained_backbone="torchvision://resnet50",
		output_shapes=[ShapeSpec(channels=256, stride=s) for s in [4, 8, 16, 32, 64]],
		output_names=["p2", "p3", "p4", "p5", "p6"],
	)


try:
	import mmdet.models
	HAS_MMDET = True
except ImportError:
	HAS_MMDET = False

@unittest.skipIf(not HAS_MMDET, "mmdet not available")
class selec_MMDetWrapper(unittest.TestCase):
	def test_backbone(self):
		MMDetBackbone(
			backbone=dict(
				type="DetectoRS_ResNet",
				conv_cfg=dict(type="ConvAWS"),
				sac=dict(type="SAC", use_deform=True),
				stage_with_sac=(False, True, True, True),
				depth=50,
				num_stages=4,
				out_indices=(0, 1, 2, 3),
				frozen_stages=1,
				norm_cfg=dict(type="BN", requires_grad=True),
				norm_eval=True,
				style="pytorch",
			),
			neck=dict(
				type="FPN",
				in_channels=[256, 512, 1024, 2048],
				out_channels=256,
				num_outs=5,
			),
			# skip pretrained model for tests
			# pretrained_backbone="torchvision://resnet50",
			output_shapes=[ShapeSpec(channels=256, stride=s) for s in [4, 8, 16, 32, 64]],
			output_names=["p2", "p3", "p4", "p5", "p6"],
		)
'''












































