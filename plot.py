import os
import itertools
import colorsys
import numpy as np
import pandas as pd
import networkx as nx

import grapefruit as gf

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["hatch.linewidth"] = 2.0

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


def label_W(Ws, eps=1e-4):
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
	cmap = generate_cmap(np.max(np.abs(W)), signed=False)  # (W<0).any())
	A = cmap(np.abs(W))
	hatch_locs = np.argwhere(W<0)
	plt.figure()
	plt.matshow(A)
	for (i, j) in hatch_locs:
		plt.gca().add_patch(mpl.patches.Rectangle((j-0.5, i-0.5), 1, 1, hatch="//", fill=False, snap=False))
	plt.xticks(np.arange(A.shape[1]), np.arange(A.shape[1])+1, fontsize=20)
	plt.yticks(np.arange(A.shape[0]), np.arange(A.shape[0])+1, fontsize=20)
#	plt.xlabel("Input neurons", fontsize=16)
#	plt.ylabel("Hidden neurons", fontsize=16)
	plt.savefig(filename+".png")
	plt.close("all")


def textplot_W(Ws, filename):
	W = label_W(Ws)
	with open(filename+".txt", "w") as f:
		print(np.array2string(W, separator=", "), file=f)


### visualize cohomology

def build_graph(rho_generators, colors):
	n = len(rho_generators[0])
	domain = np.arange(n)+1
	G = nx.MultiDiGraph()
	G.add_nodes_from(domain)
	edge_counter = {}
	for (gen, color) in zip(rho_generators, colors):
		permuted = gf.permlib.Permutation(gen).inverse().action(domain)
		permuted, signs = np.abs(permuted), np.sign(permuted)
		for (i, j, sign) in zip(domain, permuted, signs):
			edge = tuple(sorted([i, j]))
			if edge not in edge_counter:
				rad = 0
				edge_counter[edge] = [0, 0, 0]
			else:
				idx = int(i>j) + 2*int(i==j)
				rad = 0.2*(1 + edge_counter[edge][idx])
				edge_counter[edge][idx] += 1
			G.add_edge(i, j, color=color, sign=sign, rad=rad)
	return G


def plot_cohomology(rho_generators, colors, filename, layout="planar"):
	colors = colors[:len(rho_generators)]
	G = build_graph(rho_generators, colors)
	pos = getattr(nx, "{}_layout".format(layout))(G)
	signs = [1, -1]
	styles = {1: "solid", -1: (0, (2, 4))}
	nx.draw_networkx_nodes(G, pos=pos, node_size=600, node_color="white", edgecolors="black")
	nx.draw_networkx_labels(G, pos=pos, font_size=24)
	for (color, sign) in itertools.product(colors, signs):
		edge_list = [key for (key, value) in dict(G.edges).items() if value["color"]==color and value["sign"]==sign]
		for edge in edge_list:
			nx.draw_networkx_edges(G, pos=pos, edgelist=[edge], edge_color=color, style=styles[sign], connectionstyle="arc3, rad={}".format(G.edges[edge]["rad"]), arrowsize=20)
	plt.savefig(filename+".png")
	plt.close()


def textplot_cohomology(rho_generators, colors, filename):
	colors = colors[:len(rho_generators)]
	G = build_graph(rho_generators, colors)
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
	layout = lambda group: "planar" if group[0]=="D" else "circular"

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
				plot_cohomology(f["rho_generators"], colors, os.path.join(plots_dir, "vis", group, "cohomology", filename), layout=layout(group))
				textplot_W(f["Ws"], os.path.join(plots_dir, "text", group, "weights", filename))
				textplot_cohomology(f["rho_generators"], colors, os.path.join(plots_dir, "text", group, "cohomology", filename))


	print("Done!")
