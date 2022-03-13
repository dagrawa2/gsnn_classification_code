import code
import itertools
import numpy as np
import networkx as nx
from functools import reduce
from scipy.linalg import circulant as sp_linalg_circulant
from scipy.linalg import null_space


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

def kron_with_e1(x, n, e1_first=False):
	A = np.zeros((x.shape[0], n))
	A[:,0] = x
	if e1_first:
		A = A.T
	return A.reshape((-1))

def list_factors(n):
	S = set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
	return sorted(list(S))


order = 8
factors = list_factors(order)
bools = [False, True]

counter = {"type1_proposed": 0, "type1_accepted": 0, "type2_proposed": 0, "type2_accepted": 0}

for (type2, p) in itertools.product(bools, factors):
	q = order//p
	if type2:
		p2 = p//2
		if p2 not in factors:
			continue
		q2 = order//p2

	PH_vector = kron_with_e1(np.ones((p)), q)/p
	if type2:
		PH_vector = kron_with_e1(np.ones((p2)), q2)/p2 - PH_vector

	Z = Zorbits(q).transversal()
	for z in Z:
		conv = circulant(z).dot(z)
		zg_vector = kron_with_e1(conv, p, e1_first=True)
		if not type2:
			zg_vector = zg_vector - z.sum()**2/order
		A_vector = PH_vector - zg_vector
		A = circulant(A_vector)
		U = null_space( np.eye(order)-A )

		if type2:
			counter["type2_proposed"] += 1
		else:
			counter["type1_proposed"] += 1

		if U.size == 0:
			continue

		skip = False
		mav = np.mean(np.abs(U))
		eps = 1e-10
		for m in range(1, q):
			if mav == 0 or np.mean(np.abs(U-np.roll(U, -m, 0))) < eps*mav:
				skip = True
				break
		if skip:
			continue

		if type2:
			counter["type2_accepted"] += 1
		else:
			counter["type1_accepted"] += 1

		print("H = Z_{:d}".format(p))
		if type2:
			print("K = Z_{:d}".format(p2))
		print("z = {}".format( list(z.astype(int)) ))
		print("===")

		i = 1
		for u in U.T:
			W = z[:,None]*circulant(u)[:q]
			print("W_{:d} =".format(i))
			print(np.round(W, 3))
			i += 1
			break
		print()


print("type 1 accepted/proposed: {:d}/{:d}".format(counter["type1_accepted"], counter["type1_proposed"]))
print("type 2 accepted/proposed: {:d}/{:d}".format(counter["type2_accepted"], counter["type2_proposed"]))
