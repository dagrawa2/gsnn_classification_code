import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def plot(m, tau, angle, filename, line=None, title=None, ylabel=None):
	n = m if tau==0 else m//2
	ks = np.arange(n)
	w_x = np.cos(2*np.pi*ks/m + angle)
	w_y = np.sin(2*np.pi*ks/m + angle)
	W = np.stack([w_x, w_y], 1)
	b = -0.5 if tau==0 else 0
	c = -W.sum(0)/2
	x = np.linspace(-2, 2, 1000, endpoint=True)
	y = np.linspace(-2, 2, 1000, endpoint=True)
	input = np.stack([np.tile(x[None,:], [len(y), 1]), np.tile(y[:,None], [1, len(x)])], 0)
	z = np.maximum(0, np.einsum("ij,jkl->ikl", W, input)+b).sum(0) \
		+ np.einsum("j,jkl->kl", c, input)
	levels = np.linspace(0.01, z.max(), 20)
	plt.figure()
	plt.contour(x, y, z, levels=levels)
	plt.quiver(-b*w_x, -b*w_y, w_x, w_y, color="blue")
	if tau == 1:
		plt.quiver(0, 0, c[0], c[1], color="red")
	for (i, xy) in enumerate((1.05-b)*W):
		plt.annotate(str(i+1), xy, color="blue", fontsize=14, weight="bold")
	if line is not None:
		plt.plot([-2, 2], [-2*np.tan(line), 2*np.tan(line)], linestyle="dashed")
	if title is not None:
		plt.title(title, fontsize=18)
	if ylabel is not None:
		plt.ylabel(ylabel, fontsize=18)
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


# C_6
os.makedirs("plots/vis/C_6/contour", exist_ok=True)
plot(6, 0, 0, "plots/vis/C_6/contour/type1.png", title="Type 1")
plot(6, 1, 0, "plots/vis/C_6/contour/type2.png", title="Type 2")

# D_6
os.makedirs("plots/vis/D_6/contour", exist_ok=True)
plot(12, 0, 0, "plots/vis/D_6/contour/K0_type1.png", line=np.pi/12, title="Architecture 0.0", ylabel="Type 1")
plot(12, 1, 0, "plots/vis/D_6/contour/K0_type2.png", line=np.pi/12, title="Architecture 1.1", ylabel="Type 2")
plot(6, 0, np.pi/12, "plots/vis/D_6/contour/K1_type1.png", line=np.pi/12, title="Architecture 2.0")
plot(6, 1, np.pi/12, "plots/vis/D_6/contour/K1_type2.png", line=np.pi/12, title="Architecture 4.2")
plot(6, 0, -5*np.pi/12, "plots/vis/D_6/contour/K2_type1.png", line=np.pi/12, title="Architecture 3.0")
plot(6, 1, -5*np.pi/12, "plots/vis/D_6/contour/K2_type2.png", line=np.pi/12, title="Architecture 4.3")

print("Done!")
