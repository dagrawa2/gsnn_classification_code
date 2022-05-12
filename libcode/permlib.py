import itertools
import numpy as np
import networkx as nx
import galois


class Permutation(object):

	def __init__(self, perm):
		self.perm = np.asarray(perm, dtype=int)
		self.degree = len(self.perm)

	def inverse(self):
		inverse_perm = self.inverse_array()
		inverse = Permutation(inverse_perm)
		return inverse

	def inverse_array(self):
		indices = np.abs(self.perm)
		signs = np.sign(self.perm)
		inverse_indices = np.argsort(indices)
		inverse_signs = signs[inverse_indices]
		inverse_perm = (inverse_indices+1)*inverse_signs
		return inverse_perm

	def action(self, X, galois_field=False):
		inverse_perm = self.inverse_array()
		indices = np.abs(inverse_perm)-1
		signs = np.sign(inverse_perm)
		if galois_field:
			signs = galois.GF2((1-signs)//2)
			out = signs[(..., *( [np.newaxis]*(X.ndim-1) ))] + X[indices]
		else:
			out = signs[(..., *( [np.newaxis]*(X.ndim-1) ))] * X[indices]
		return out

	def matrix(self):
		I = np.eye(self.degree)
		A = self.action(I)
		return A


def projection_matrix(perms, return_order=False):
	order = len(perms)
	P = sum([Permutation(perm).matrix() for perm in perms])/order
	if return_order:
		return P, order
	return P
