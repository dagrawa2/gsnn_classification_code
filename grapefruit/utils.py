import torch
from torch import nn
from torch.nn import functional as F


def normalize(x):
	if x.ndim == 1:
		return F.normalize(x.unsqueeze(0)).squeeze(0)
	return F.normalize(x)
