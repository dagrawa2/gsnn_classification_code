import os
import itertools
import colorsys
import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=8)
matplotlib.rc("ytick", labelsize=8)

np.random.seed(123)


### visualize weight matrix

def generate_cmap(n, signed=False):
	hues = np.linspace(0, 1, n, endpoint=False)
	cmap_i = lambda i: colorsys.hls_to_rgb(hues[np.abs(i)-1], 0.5+0.3*int(signed)*np.sign(i), 1)
	cmap_i = np.vectorize(cmap_i)
	cmap = lambda A: np.stack(cmap_i(A), -1)
	return cmap


def labeled_W(Ws, eps=1e-5):
	coefficients = np.random.uniform(-1, 1, (Ws.shape[0]))
	coefficients = coefficients/np.sqrt(np.sum(coefficients**2))
	W = np.einsum("i,ijk->jk", coefficients, Ws)
	W = (W/eps).astype(int)
	W[W==0] = W.max()+1
	values, indices = np.unique(np.abs(W), return_index=True)
	values = values[np.argsort(indices)]
	signs = np.sign(W)
	A = 0
	for (i, value) in enumerate(values):
		B = (np.abs(W)==value).astype(int) * signs
		B = B.ravel()[ np.nonzero(B.ravel())[0][0] ] * B
		A = A + (i+1)*B
	return A


def plot_W(Ws, filename, format="png"):
	W = labeled_W(Ws)
	filename = "{}.{}".format(filename, format)
	if format == "txt":
		with open(filename, "w") as f:
			print(np.array2string(W, separator=", "), file=f)
		return
	cmap = generate_cmap(np.max(np.abs(W)), signed=(W<0).any())
	A = cmap(W)
	plt.figure()
	plt.matshow(A)
	plt.xlabel("Input neurons", fontsize=8)
	plt.ylabel("Hidden neurons", fontsize=8)
	plt.savefig(filename)
	plt.close("all")


if __name__ == "__main__":
	results_dir = "results"
	plots_dir = "plots"
	textplots_dir = "textplots"

	group_dirs = sorted(os.listdir(results_dir))
	type_dirs = ["type1", "type2"]

	def plot(plots_dir, format):
		for (group_dir, type_dir) in itertools.product(group_dirs, type_dirs):
			input_dir = os.path.join(results_dir, group_dir, type_dir)
			output_dir = os.path.join(plots_dir, group_dir, type_dir)
			os.makedirs(output_dir, exist_ok=True)
			for filename in sorted(os.listdir(input_dir)):
				input_file = os.path.join(input_dir, filename)
				output_file = os.path.splitext(os.path.join(output_dir, filename))[0]
				plot_W( np.load(input_file)["Ws"], output_file, format=format)


	print("Plotting . . . ")
	plot(plots_dir, "png")
	plot(textplots_dir, "txt")

	print("Done!")
