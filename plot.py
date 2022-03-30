import os
import itertools
import colorsys
import numpy as np
import pandas as pd
import networkx as nx

import grapefruit as gf

import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=8)
matplotlib.rc("ytick", labelsize=8)

np.random.seed(123)


### visualize weight matrix

def generate_cmap(n, signed=False):
	hues = np.linspace(0, 1, n, endpoint=False)
	def cmap(i):
		if i == 0:
			return (1, 1, 1)
		hue_idx = np.abs(i)-1
		if signed:
			return colorsys.hls_to_rgb(hues[hue_idx], 0.5-0.3*np.sign(i), 1)
		return colorsys.hls_to_rgb(hues[hue_idx], 0.5, 1)
	cmap_vec = lambda A: np.array([list(cmap(a)) for a in A.ravel()]).reshape((*A.shape, 3))
	return cmap_vec


def label_W(Ws, eps=1e-5):
	coefficients = np.random.uniform(-1, 1, (Ws.shape[0]))
	coefficients = coefficients/np.sqrt(np.sum(coefficients**2))
	W = np.einsum("i,ijk->jk", coefficients, Ws)
	W = (W/eps).astype(int)
	values, indices = np.unique(np.abs(W), return_index=True)
	if 0 in values:
		idx = np.where(values==0)[0][0]
		values = np.delete(values, idx)
		indices = np.delete(indices, idx)
	values = values[np.argsort(indices)]
	signs = np.sign(W)
	A = 0
	for (i, value) in enumerate(values):
		B = (np.abs(W)==value).astype(int) * signs
		B = B.ravel()[ np.nonzero(B.ravel())[0][0] ] * B
		A = A + (i+1)*B
	return A


def plot_W(Ws, filename):
	W = label_W(Ws)
	cmap = generate_cmap(np.max(np.abs(W)), signed=(W<0).any())
	A = cmap(W)
	plt.figure()
	plt.matshow(A)
	plt.xlabel("Input neurons", fontsize=8)
	plt.ylabel("Hidden neurons", fontsize=8)
	plt.savefig(filename+".png")
	plt.close("all")


def textplot_W(Ws, filename):
	W = label_W(Ws)
	with open(filename+".txt", "w") as f:
		print(np.array2string(W, separator=", "), file=f)


### visualize cohomology

def build_graph(rho_generators, z, colors):
	n = len(z)
	domain = np.arange(n)+1
	G = nx.DiGraph()
	G.add_nodes_from(domain)
	for (gen, color) in zip(rho_generators, colors):
		permuted = z*gf.permlib.Permutation(gen).action(z*domain)
		permuted, signs = np.abs(permuted), np.sign(permuted)
		for (i, j, sign) in zip(domain, permuted, signs):
			G.add_edge(i, j, color=color, sign=sign)
	return G


def plot_cohomology(rho_generators, z, colors, filename):
	colors = colors[:len(rho_generators)]
	G = build_graph(rho_generators, z, colors)
	pos = nx.planar_layout(G)
	signs = [1, -1]
	styles = {1: "solid", -1: "dotted"}
	nx.draw(G, pos=pos, edgelist=[], node_color="black", with_labels=True)
	for (color, sign) in itertools.product(colors, signs):
		edgelist = [key for (key, value) in dict(G.edges).items() if value["color"]==color and value["sign"]==sign]
		nx.draw(G, pos=pos, nodelist=[], edgelist=edgelist, edge_color=color, style=styles[sign])
	plt.savefig(filename+".png")
	plt.close()


def textplot_cohomology(rho_generators, z, colors, filename):
	colors = colors[:len(rho_generators)]
	G = build_graph(rho_generators, z, colors)
	with open(filename+".csv", "w") as f:
		f.write("i,j,color,sign\n")
		for (key, value) in dict(G.edges).items():
			f.write("{:d},{:d},{},{:d}\n".format( \
				key[0], key[1], value["color"], value["sign"] ))


if __name__ == "__main__":
	results_dir = "results"
	plots_dir = "plots"

	groups = sorted(os.listdir(results_dir))
	colors = ["red", "blue", "green"]

	print("Plotting . . . ")
	for group in groups:
		os.makedirs(os.path.join(plots_dir, "vis", group, "weights"), exist_ok=True)
		os.makedirs(os.path.join(plots_dir, "vis", group, "cohomology"), exist_ok=True)
		os.makedirs(os.path.join(plots_dir, "text", group, "weights"), exist_ok=True)
		os.makedirs(os.path.join(plots_dir, "text", group, "cohomology"), exist_ok=True)
		for filename in sorted(os.listdir(os.path.join(results_dir, group, "npzs"))):
			with np.load(os.path.join(results_dir, group, "npzs", filename)) as f:
				filename = os.path.splitext(filename)[0]
				plot_W(f["Ws"], os.path.join(plots_dir, "vis", group, "weights", filename))
				plot_cohomology(f["rho_generators"], f["z"], colors, os.path.join(plots_dir, "vis", group, "cohomology", filename))
				textplot_W(f["Ws"], os.path.join(plots_dir, "text", group, "weights", filename))
				textplot_cohomology(f["rho_generators"], f["z"], colors, os.path.join(plots_dir, "text", group, "cohomology", filename))


	print("Done!")
