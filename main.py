import os
import argparse
import numpy as np
from scipy.linalg import null_space

import grapefruit as gf


# command-line arguments
parser=argparse.ArgumentParser()
parser.add_argument('--group', '-g', required=True, type=str, help='Symmetry group.')
parser.add_argument('--output_dir', '-o', required=True, type=str, help='Output directory.')
args=parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)

group_generators = gf.groups.group(args.group)
gap_output = gf.gapcode.generate_representations(group_generators, os.path.join(args.output_dir, "gap_output.json"))

csv_file = open(os.path.join(args.output_dir, "numbers.csv"), "w")
csv_file.write("json_idx_H,json_idx_K,order_H,order_K,type,num_neg_z,dim,accepted,npz_idx\n")

for (json_idx_H, dict_H) in enumerate(gap_output):
	P_H, order_H = gf.permlib.projection_matrix(dict_H["H"], return_order=True)
	transversal = [gf.permlib.Permutation(t).matrix() for t in dict_H["transversal"]]
	for (json_idx_K, dict_K) in enumerate(dict_H["out"]):
		order_K = len(dict_K["K"])
		typenum = 1 if order_K==order_H else 2
		if typenum == 1:
			P_G = gf.permlib.projection_matrix(dict_H["transversal"]).dot(P_H)
		else:
			P_K = gf.permlib.projection_matrix(dict_K["K"])

		Z = gf.permlib.zorbit_transversal(dict_K["centralizer_generators"])
		for z in Z:
			z_dot_transversal = sum([z_i*t_i for (z_i, t_i) in zip(z, transversal)])
			T = z_dot_transversal.T.dot(z_dot_transversal)
			if typenum == 1:
				A = P_H+z.sum()**2*P_G - T
			else:
				A = P_K-P_H - T

			csv_file.write("{:d},{:d},{:d},{:d},{:d},{:d}".format( \
				json_idx_H, json_idx_K, order_H, order_K, typenum, np.sum(z<0) ))

			I = np.eye(A.shape[0])
			U = null_space(I-A)
			if U.size == 0:
				csv_file.write("0,False,nan\n")
				continue

			skip = False
			mav = np.mean(np.abs(U))
			eps = 1e-10
			tUs = []
			for t in transversal:
				tU = t.dot(U)
				tUs.append(tU)
				if mav == 0 or np.mean(np.abs(tU-U)) < eps*mav \
					or (typenum==2 and np.mean(np.abs(tU+U)) < eps*mav):
					skip = True
					break
			if skip:
				csv_file.write("{:d},False,nan\n".format(U.shape[1]))
				continue

			Ws = np.stack([z_i*tU.T for (z_i, tU) in zip(z, tUs)], 1)
			weights_dir = os.path.join(args.output_dir, "type{:d}".format(typenum))
			os.makedirs(weights_dir, exist_ok=True)
			npz_idx = len(os.listdir(weights_dir))
			np.savez(os.path.join(weights_dir, "{:d}.npz".format(npz_idx) ), z=z, Ws=Ws)
			csv_file("{:d},True,{:d}\n".format(U.shape[1], npz_idx))


csv_file.close()

print("Done!")