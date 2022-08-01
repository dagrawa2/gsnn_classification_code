import os
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from scipy.linalg import null_space

import libcode as lc


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
npzs_dir = os.path.join(output_dir, "npzs")
if args.overwrite and os.path.isdir(output_dir):
	shutil.rmtree(output_dir)
os.makedirs(npzs_dir, exist_ok=True)

# run GAP script
print("Running GAP script . . . ")
group_generators = lc.groups.group(args.group)
gap_output = lc.gapcode.generate_representations(group_generators, os.path.join(output_dir, "gap_output.json"))

# initialize CSV log file
csv_file = open(os.path.join(output_dir, "numbers.csv"), "w")
csv_file.write("json_idx_H,json_idx_K,order_H,order_K,type,class,proj_rank,accepted,npz_idx\n")

# apply linear constraints
print("Applying linear constraints . . . ")
for (json_idx_H, dict_H) in enumerate(gap_output):
	P_H, order_H = lc.permlib.projection_matrix(dict_H["H"], return_order=True)
	transversal = [lc.permlib.Permutation(t).matrix() for t in dict_H["transversal"]]
	assert np.array_equal(transversal[0].astype(int), np.eye(transversal[0].shape[0], dtype=int)), "The first transversal element is expected to be the identity."
	for (json_idx_K, dict_K) in enumerate(dict_H["out"]):
		order_K = len(dict_K["K"])
		typenum = 1 if order_K==order_H else 2
		if typenum == 1:
			P = np.copy(P_H)
		else:
			P_K = lc.permlib.projection_matrix(dict_K["K"])
			P = P_K-P_H

		csv_file.write("{:d},{:d},{:d},{:d},{:d},{:d},".format( \
			json_idx_H, json_idx_K, order_H, order_K, typenum, json_idx_K ))

		I = np.eye(P.shape[0])
		U = null_space(I-P)
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

		Ws = np.stack([tU.T for tU in tUs], 1)
		npz_idx = len(os.listdir(npzs_dir))
		filename = "{:d}-H{:d}_K{:d}_{:d}x{:d}.npz".format( \
			npz_idx, json_idx_H, json_idx_K, Ws.shape[1], Ws.shape[2] )
		np.savez(os.path.join(npzs_dir, filename), Ws=Ws, \
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
	f.write("\n|G/H| | rank(P)\n===\n")
	for order_H in order_Hs:
		ranks = np.unique( df[(df.order_H==order_H) & (df.accepted)].proj_rank.values )
		f.write("{:d} | {}\n".format( \
			order_G//order_H, ", ".join(map(str, ranks)) ))
	# time
	f.write("\nTime: {:.1f} s\n".format( \
		time.time()-time_0 ))

print("Done!")
