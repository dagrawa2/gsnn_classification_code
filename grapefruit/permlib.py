import itertools
import numpy as np
import networkx as nx
import galois


class Permutation(object):

	def __init__(self, perm):
		self.perm = np.asarray(perm, dtype=int)
		self.degree = len(self.perm)

	def inverse(self):
		indices = np.abs(self.perm)
		signs = np.sign(self.perm)
		inverse_indices = np.argsort(indices)
		inverse_signs = signs[inverse_indices]
		inverse_perm = (inverse_indices+1)*inverse_signs
		return inverse_perm

	def action(self, X, galois_field=False):
		inverse_perm = self.inverse()
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

def binary_matrix(degree):
	A = galois.GF2([list(tup) for tup in itertools.product([0, 1], repeat=degree)]).T
	return A

def binary_matrix2int_vector(A):
	out = np.array([int("".join(row), 2) for row in np.array(A.T).astype(str)])
	return out

def zorbit_transversal(generators):
	degree = len(generators[0])
	binmatrix = binary_matrix(degree)
	vertices = binary_matrix2int_vector(binmatrix)
	neighbors = np.stack([binary_matrix2int_vector( Permutation(gen).action(binmatrix, galois_field=True) ) for gen in generators], 0)
	adjacency_dict = {v: N for (v, N) in zip(vertices, neighbors.T)}
	G = nx.Graph(adjacency_dict)
	components = nx.connected_components(G)
	repr_vertices = sorted([min(C) for C in components])
	repr_vectors = []
	for v in repr_vertices:
		vec = list(bin(v)[2:])
		vec = ["0"]*(degree-len(vec)) + vec
		repr_vectors.append(vec)
	repr_vectors = 1-2*np.array(repr_vectors, dtype=int)
	return repr_vectors
