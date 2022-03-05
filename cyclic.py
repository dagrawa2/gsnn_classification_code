import code
import numpy as np
import networkx as nx
from functools import reduce
from scipy.linalg import circulant as sp_linalg_circulant


class Zorbits(object):

	def __init__(self, dim):
		self.dim = dim

	def cycle_bits(self, n):
		return n//2 + 2**(self.dim-1)*(n%2)

	def flip_bits(self, n):
		mask = 2**self.dim - 1
		return ~n & mask

	def transversal(self):
		vertices = list(range(2**self.dim))
		edges = [(v, f(v)) for v in vertices for f in [self.cycle_bits, self.flip_bits]]
		adjacency_dict = {v: [] for v in vertices}
		for (v, w) in edges:
			adjacency_dict[v].append(w)
		G = nx.Graph(adjacency_dict)
		components = nx.connected_components(G)
		repr_vertices = sorted([min(C) for C in components])
		repr_vectors = []
		for v in repr_vertices:
			vec = list(bin(v)[2:])
			vec = ["0"]*(self.dim-len(vec)) + vec
			repr_vectors.append(vec)
		repr_vectors = 1-2*np.array(repr_vectors, dtype=int)
		return repr_vectors

def circulant(x):
	return sp_linalg_circulant(x).T

def factors(n):
	S = set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
	return sorted(list(S))

def projector_range(P, eps=1e-5):
	lams, U = np.linalg.eigh(P)
	U = U[:, np.abs(lams-1)<eps]
	return U

def kron_with_e1(x, n, e1_first=False):
	A = np.zeros((x.shape[0], n))
	A[:,0] = x
	if e1_first:
		A = A.T
	return A.reshape((-1))


order = 12
facs = factors(order)

for p in facs:
	q = order//p
	P_H_vector = kron_with_e1(np.ones((p)), q)/p

	Z = Zorbits(q).transversal()
	for z in Z:
		if z.min() == 1: continue
		print("H = Z_{:d}".format(p))
		print("z = {}".format( list(z.astype(int)) ))
		print("===")
		conv = circulant(z).dot(z)
		zg_vector = kron_with_e1(conv, p, e1_first=True) \
			- z.sum()**2/order
		A_vector = P_H_vector - zg_vector
		A = circulant(A_vector)
		I = np.eye(order)
		P_A = I - np.linalg.pinv(I-A).dot(I-A)

		skip = False
		mav = np.mean(np.abs(P_A))
		for m in range(1, q):
			if mav == 0 or np.mean(np.abs((P_A-np.roll(P_A, -m, 0)))) < mav*1e-10:
				skip = True
				break
		if skip:
			print("skip\n")
			continue

		U = projector_range(P_A)
		i = 1
		for u in U.T:
			W = z[:,None]*circulant(u)[:q]
			print("W_{:d} =".format(i))
			print(np.round(W, 3))
			i += 1
		print()


###

for p_1 in facs:
	p_2 = p_1//2
	if p_2 not in facs:
		continue
	q_1 = order//p_1
	q_2 = order//p_2
	P_H_vector = kron_with_e1(np.ones((p_1)), q_1)/p_1
	P_K_vector = kron_with_e1(np.ones((p_2)), q_2)/p_2

	Z = Zorbits(q_1).transversal()
	for z in Z:
		if z.min() == 1: continue
		print("H = Z_{:d}".format(p_1))
		print("K = Z_{:d}".format(p_2))
		print("z = {}".format( list(z.astype(int)) ))
		print("===")
		conv = circulant(z).dot(z)
		zg_vector = kron_with_e1(conv, p_1, e1_first=True)
		A_vector = P_K_vector - P_H_vector - zg_vector
		A = circulant(A_vector)
		I = np.eye(order)
		P_A = I - np.linalg.pinv(I-A).dot(I-A)

		skip = False
		mav = np.mean(np.abs(P_A))
		for m in range(1, q_1):
			if mav == 0 or np.mean(np.abs((P_A-np.roll(P_A, -m, 0)))) < mav*1e-10 \
				or np.mean(np.abs((P_A+np.roll(P_A, -m, 0)))) < mav*1e-10:
				skip = True
				break
		if skip:
			print("skip\n")
			continue

		U = projector_range(P_A)
		i = 1
		for u in U.T:
			W = z[:,None]*circulant(u)[:q_1]
			print("W_{:d} =".format(i))
			print(np.round(W, 3))
			i += 1
		print()
