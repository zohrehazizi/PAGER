########################
# Author: Zohreh Azizi #
########################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_configs import device, convert_to_torch, convert_to_numpy
from torch.nn import Fold
from torch.nn import Unfold
#device = 'cpu'

class image_regressor(nn.Module):
	def __init__(self, regressorArgs):
		super().__init__()
		type_ = regressorArgs['type']
		regressorArgs.pop('type')
		if type_=='linear':
			regressorArgs['alpha']=0
			self.regressor=Ridge(**regressorArgs)
		elif type_=='ridge':
			self.regressor=Ridge(**regressorArgs)
		elif type_=='xgboost':
		   assert False, "XGB regression not implemented. use Ridge"
		else:
			raise(f"unsupported regressor type: {regressorType}")

		if 'window_size' in regressorArgs and regressorArgs['window_size'] is not None:
			self.patch_size = regressorArgs['window_size']
			if 'stride' in regressorArgs:
				self.stride = regressorArgs['stride']
			else:
				self.stride = regressorArgs['window_size']
			self.patch_extractor = Unfold(kernel_size=self.patch_size, dilation=1, padding=0, stride=self.stride)
		else:
			self.patch_extractor = None
	def extract_patches(self, X):
		assert len(X.shape)==4
		X_torch = X.transpose(0,3,1,2)
		X_torch = convert_to_torch(X_torch)
		X_patches = self.patch_extractor(X_torch).cpu().numpy() #<n,p*p*3, patches_per_image>
		X_patches = X_patches.transpose(0,2,1) #<n, patches_per_image, p*p*3>
		X = X_patches.reshape(X_patches.shape[0]*X_patches.shape[1], -1) #<nxpatches_per_image, p*p*3>
		return X


	def fit(self,X,Y):
		if self.patch_extractor is not None:
			X = self.extract_patches(X)
			Y = self.extract_patches(Y)
		X = X.reshape(X.shape[0],-1)
		Y = Y.reshape(Y.shape[0],-1)
		self.regressor.fit(X,Y)
	def forward(self,X):
		orig_shape = X.shape
		n,h,w,c = X.shape
		if self.patch_extractor is not None:
			X = self.extract_patches(X)
		X = convert_to_torch(X)
		X = X.reshape(X.shape[0],-1)
		Y = self.regressor(X)
		if self.patch_extractor is not None:
			Y = Y.reshape(n,-1,Y.shape[-1])
			Y = Y.permute(0,2,1)
			Y = Fold(output_size=(h,w), kernel_size=self.patch_size, dilation=1, padding=0, stride=self.stride)(Y)
			divisor = np.ones((c,h,w))
			divisor = np.expand_dims(divisor, axis=0)
			divisor_torch = convert_to_torch(divisor)
			divisor_patches = self.patch_extractor(divisor_torch) #<n,p*p*3, patches_per_image>
			divisor_reconst = Fold(output_size=(h,w), kernel_size=self.patch_size, dilation=1, padding=0, stride=self.stride)(divisor_patches)
			Y = Y.cpu().numpy().transpose(0,2,3,1)
			divisor_reconst = divisor_reconst.cpu().numpy().transpose(0,2,3,1)
			Y = Y/divisor_reconst
			return Y
		else:
			Y = Y.reshape(orig_shape[0],orig_shape[1],orig_shape[2],orig_shape[3])
			return Y.cpu().numpy()
	def get_size(self):
		return torch.prod(convert_to_torch(self.regressor.w.data.size())).item()

class Ridge(nn.Module):
	##### credits to this class go to https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
	def __init__(self, alpha = 0, fit_intercept = True,**kwargs):
		super().__init__()
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.w = nn.parameter.Parameter(None, requires_grad=False)
		
	def fit(self, X: torch.tensor, y: torch.tensor) -> None:
		X = convert_to_torch(X)
		y = convert_to_torch(y)
		X = X.rename(None)
		y = y.rename(None)
		if self.fit_intercept:
			X = torch.cat([torch.ones(X.shape[0], 1).to(device), X], dim = 1)
		# Solving X*w = y with Normal equations:
		# X^{T}*X*w = X^{T}*y 
		lhs = X.T @ X 
		rhs = X.T @ y
		if self.alpha == 0:
			self.w.data, _ = torch.lstsq(rhs, lhs)
		else:
			ridge = self.alpha*torch.eye(lhs.shape[0]).to(device)
			self.w.data, _ = torch.lstsq(rhs, lhs + ridge)
			
	def forward(self, X: torch.tensor) -> None:
		X = convert_to_torch(X)
		X = X.rename(None)
		if self.fit_intercept:
			X = torch.cat([torch.ones(X.shape[0], 1).to(device), X], dim = 1)
		out = X @ self.w
		return out