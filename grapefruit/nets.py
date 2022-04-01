import os
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import utils


class GSNN(nn.Module):

	def __init__(self, input_dir, architecture_idx):
		super(GSNN, self).__init__()
		self.input_dir = input_dir
		self.architecture_idx = architecture_idx

		filename = sorted(os.listdir(input_dir))[architecture_idx]
		self.type = int( re.search(r"K([0-9]+)_", filename).group(1) )

		with np.load(os.path.join(input_dir, filename)) as f:
			self.U = torch.tensor(f["U"], dtype=torch.float32)
			self.Ws = torch.tensor(f["Ws"], dtype=torch.float32)
			self.z = torch.tensor(f["z"], dtype=torch.float32)

		self.a = nn.Parameter(2*torch.bernoulli(torch.tensor(0.5))-1)
		self.b = nn.Parameter(torch.tensor(0.0))
		if self.type == 2:
			self.b.requires_grad = False

		self.W_unconstrained = nn.Parameter(utils.normalize(torch.randn(self.Ws.shape[0])))
		self.c_unconstrained = nn.Parameter(0.1*torch.randn(self.U.shape[0]))

		self.d = nn.Parameter(torch.tensor(0.0))


	def forward(self, X):
		W = torch.einsum("i,ijk->jk", self.W_unconstrained, self.Ws).t()
		bz = (self.b*self.z).unsqueeze(0)
		c = torch.einsum("i,ij->j", self.c_unconstrained, self.U)
		out = self.a*F.relu(X.mm(W)+bz).sum(1) + X.mv(c) + self.d
		return out
