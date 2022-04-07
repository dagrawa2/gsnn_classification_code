import os
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GSNN(nn.Module):

	def __init__(self, input_dir, architecture_idx, a_init, w_unconstrained_init):
		super(GSNN, self).__init__()
		self.input_dir = input_dir
		self.architecture_idx = architecture_idx
		self.a_init = a_init

		filename = sorted(os.listdir(input_dir))[architecture_idx]
		self.type = int( re.search(r"K([0-9]+)_", filename).group(1) )

		with np.load(os.path.join(input_dir, filename)) as f:
			self.U = torch.tensor(f["U"], dtype=torch.float32)
			self.Ws = torch.tensor(f["Ws"], dtype=torch.float32)
			self.z = torch.tensor(f["z"], dtype=torch.float32)

		w_unconstrained_init = int(w_unconstrained_init)
		self.w_unconstrained_init = np.zeros((self.Ws.shape[0]), dtype=np.float32)
		self.w_unconstrained_init[np.abs(w_unconstrained_init)-1] = np.sign(w_unconstrained_init)

		self.a = nn.Parameter(torch.tensor(self.a_init, dtype=torch.float32))
		self.b = nn.Parameter(torch.tensor(0.0))
		if self.type == 2:
			self.b.requires_grad = False

		self.w_unconstrained = nn.Parameter(torch.tensor(self.w_unconstrained_init))
		self.c_unconstrained = nn.Parameter(torch.zeros(self.U.shape[0]))
		self.d = nn.Parameter(torch.tensor(0.0))


	def forward(self, X):
		W = torch.einsum("i,ijk->jk", self.w_unconstrained, self.Ws).t()
		bz = (self.b*self.z).unsqueeze(0)
		c = torch.einsum("i,ij->j", self.c_unconstrained, self.U)
		out = self.a*F.relu(X.mm(W)+bz).sum(1) + X.mv(c) + self.d
		return out
