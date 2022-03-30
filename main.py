import os
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from scipy.linalg import null_space

import grapefruit as gf


# command-line arguments
parser=argparse.ArgumentParser()
parser.add_argument('--group', '-g', required=True, type=str, help='Symmetry group.')
parser.add_argument('--results_dir', '-r', required=True, type=str, help='Master results directory.')
parser.add_argument('--overwrite', '-o', action="store_true", help='Overwrite existing results directory.')
args=parser.parse_args()


# mark initial time
time_0 = time.time()

# create output directories
output_dir = os.path.join(args.results_dir, args.group)
if args.overwrite and os.path.isdir(output_dir):
	shutil.rmtree(output_dir)
for typenum in [1, 2]:
	os.makedirs(os.path.join(output_dir, "type{:d}".format(typenum)), exist_ok=True)

# run GAP script
print("Running GAP script . . . ")
group_generators = gf.groups.group(args.group)
gap_output = gf.gapcode.generate_representations(group_generators, os.path.join(output_dir, "gap_output.json"))

# initialize CSV log file
csv_file = open(os.path.join(output_dir, "numbers.csv"), "w")
csv_file.write("json_idx_H,json_idx_K,order_H,order_K,type,class,num_neg_z,rank_PA,accepted,npz_idx\n")

# apply linear constraints
print("Applying linear constraints . . . ")
for (json_idx_H, dict_H) in enumerate(gap_output):
	P_H, order_H = gf.permlib.projection_matrix(dict_H["H"], return_order=True)
	transversal = [gf.permlib.Permutation(t).matrix() for t in dict_H["transversal"]]
	assert np.array_equal(transversal[0].astype(int), np.eye(transversal[0].shape[0], dtype=int)), "The first transversal element is expected to be the identity."
	for (json_idx_K, dict_K) in enumerate(dict_H["out"]):
		order_K = len(dict_K["K"])
		typenum = 1 if order_K==order_H else 2
		if typenum == 1:
			P_G = gf.permlib.projection_matrix(dict_H["transversal"]).dot(P_H)
		else:
			P_K = gf.permlib.projection_matrix(dict_K["K"])

		Z = gf.permlib.zorbit_transversal(dict_K["J_generators"])
		for z in Z:
			z_dot_transversal = sum([z_i*t_i for (z_i, t_i) in zip(z, transversal)])
			T = z_dot_transversal.T.dot(z_dot_transversal)
			if typenum == 1:
				A = P_H+z.sum()**2*P_G - T
			else:
				A = P_K-P_H - T

			csv_file.write("{:d},{:d},{:d},{:d},{:d},{:d},{:d},".format( \
				json_idx_H, json_idx_K, order_H, order_K, typenum, json_idx_K, np.sum(z<0) ))

			I = np.eye(A.shape[0])
			U = null_space(I-A)
			if U.size == 0:
				csv_file.write("0,False,nan\n")
				continue

			skip = False
			mav = np.mean(np.abs(U))
			eps = 1e-10
			tUs = [U]
			for t in transversal[1:]:
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
			weights_dir = os.path.join(output_dir, "type{:d}".format(typenum))
			os.makedirs(weights_dir, exist_ok=True)
			npz_idx = len(os.listdir(weights_dir))
			filename = "{:d}-{:d}x{:d}_cls{:d}_neg{:d}.npz".format( \
				npz_idx, Ws.shape[1], Ws.shape[2], json_idx_K, (z<0).sum() )
			np.savez(os.path.join(weights_dir, filename), Ws=Ws, z=z, \
				rho_generators=np.array(dict_K["rho_generators"]) )
			csv_file.write("{:d},True,{:d}\n".format(U.shape[1], npz_idx))


# close CSV file
csv_file.close()

# generate notes
print("Generating notes . . . ")
df = pd.read_csv(os.path.join(output_dir, "numbers.csv"))
with open(os.path.join(output_dir, "notes.txt"), "w") as f:
	# accepted/proposed
	for typenum in [1, 2]:
		f.write("Type {:d} accepted/proposed: {:d}/{:d}\n".format( \
			typenum, df[df.type==typenum].accepted.values.sum(), df[df.type==typenum].shape[0] ))
	# |G/H| | rank
	order_G = df.order_H.values.max()
	order_Hs = np.unique(df[df.accepted].order_H.values)
	f.write("\n|G/H| | rank(P_A)\n===\n")
	for order_H in order_Hs:
		ranks = np.unique( df[(df.order_H==order_H) & (df.accepted)].rank_PA.values )
		f.write("{:d} | {}\n".format( \
			order_G//order_H, ", ".join(map(str, ranks)) ))
	# negative z
	f.write("\n|G/H| | |z<0|\n===\n")
	for order_H in order_Hs:
		negs = np.unique( df[(df.order_H==order_H) & (df.accepted)].num_neg_z.values )
		f.write("{:d} | {}\n".format( \
			order_G//order_H, ", ".join(map(str, negs)) ))
	# time
	f.write("\nTime: {:d} s\n".format( \
		int(time.time()-time_0) ))

print("Done!")
