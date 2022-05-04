import itertools
import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt


def plot(G, filename):
	pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
	pos = {node: -coords for (node, coords) in pos.items()}

	labels = {node: kwds["label"] for (node, kwds) in dict(G.nodes).items()}

	nx.draw_networkx_nodes(G, pos=pos, node_size=600, node_color="white", edgecolors="black")
	nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=11, font_weight="bold")
	for (edge, kwds) in dict(G.edges).items():
		color = kwds["color"]
		arrowstyle = "-|>" if color=="black" else "<|-|>"
		connectionstyle = "arc3" if color=="black" else "arc3, rad=0.2"
		nx.draw_networkx_edges(G, pos=pos, edgelist=[edge], edge_color=color, arrowstyle=arrowstyle, arrowsize=20, connectionstyle=connectionstyle)
	plt.savefig(filename)
	plt.close()


# C_6

G = nx.DiGraph()
G.add_node(0, label="0.0", layer=0)
G.add_node(1, label="1.0", layer=1)
G.add_node(2, label="2.0", layer=1)
G.add_node(3, label="1.1", layer=1)
G.add_node(4, label="3.0", layer=2)
G.add_node(5, label="3.1", layer=2)

G.add_edge(1, 0, color="black")
G.add_edge(2, 0, color="black")
G.add_edge(3, 0, color="black")
G.add_edge(4, 1, color="black")
G.add_edge(4, 2, color="black")
G.add_edge(5, 2, color="black")
G.add_edge(5, 3, color="black")

G.add_edge(1, 3, color="red")
G.add_edge(4, 5, color="red")

plot(G, "plots/vis/C_6/morphisms.png")


# D_6

G = nx.DiGraph()
G.add_node(0, label="0.0", layer=0)
G.add_node(1, label="1.0", layer=1)
G.add_node(2, label="1.1", layer=1)
G.add_node(3, label="2.0", layer=1)
G.add_node(4, label="2.1", layer=1)
G.add_node(5, label="3.0", layer=1)
G.add_node(6, label="3.1", layer=1)
G.add_node(7, label="4.0", layer=2)
G.add_node(8, label="4.1", layer=2)
G.add_node(9, label="4.2", layer=2)
G.add_node(10, label="4.3", layer=2)
G.add_node(11, label="5.0", layer=2)
G.add_node(12, label="6.0", layer=3)
G.add_node(13, label="6.1", layer=3)

for i in range(1, 7):
	G.add_edge(i, 0, color="black")
for i in [1, 3, 5]:
	G.add_edge(7, i, color="black")
for i in [1, 4, 6]:
	G.add_edge(8, i, color="black")
for i in [2, 3, 6]:
	G.add_edge(9, i, color="black")

for i in [2, 4, 5]:
	G.add_edge(10, i, color="black")
G.add_edge(11, 5, color="black")
G.add_edge(12, 7, color="black")
G.add_edge(12, 11, color="black")
G.add_edge(13, 10, color="black")
G.add_edge(13, 11, color="black")

for i in [1, 3, 5]:
	G.add_edge(i, i+1, color="red")
for (i, j) in itertools.product([7, 8, 9, 10], repeat=2):
	if i < j:
		G.add_edge(i, j, color="red")
G.add_edge(12, 13, color="red")

plot(G, "plots/vis/D_6/morphisms.png")

print("Done!")