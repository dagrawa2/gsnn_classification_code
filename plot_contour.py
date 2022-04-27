import os
import code
import numpy as np

import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=10)
matplotlib.rc("ytick", labelsize=10)


def plot(m, tau, angle, filename):
	n = m if tau==0 else m//2
	ks = np.arange(n)
	w_x = np.cos(2*np.pi*ks/m + angle)
	w_y = np.sin(2*np.pi*ks/m + angle)
	W = np.stack([w_x, w_y], 1)
	b = 0.5 if tau==0 else 0
	c = -W.sum(0)/2
	x = np.linspace(-2, 2, 1000, endpoint=True)
	y = np.linspace(-2, 2, 1000, endpoint=True)
	input = np.stack([np.tile(x[None,:], [len(y), 1]), np.tile(y[:,None], [1, len(x)])], 0)
	z = np.maximum(0, np.einsum("ij,jkl->ikl", W, input)+b).sum(0) \
		+ np.einsum("j,jkl->kl", c, input)
	zeros = np.zeros((n))
	plt.figure()
	plt.contour(x, y, z, levels=20)
	plt.quiver(zeros, zeros, w_x, w_y, color="blue")
	plt.quiver(0, 0, c[0], c[1], color="red")
	plt.savefig(filename)
	plt.close()


# C_6
os.makedirs("plots/vis/C_6/contour", exist_ok=True)
plot(6, 0, 0, "plots/vis/C_6/contour/type1.png")
plot(6, 1, 0, "plots/vis/C_6/contour/type2.png")

# D_6
os.makedirs("plots/vis/D_6/contour", exist_ok=True)
plot(12, 0, 0, "plots/vis/D_6/contour/K0_type1.png")
plot(12, 1, 0, "plots/vis/D_6/contour/K0_type2.png")
plot(6, 0, np.pi/12, "plots/vis/D_6/contour/K1_type1.png")
plot(6, 1, np.pi/12, "plots/vis/D_6/contour/K1_type2.png")
plot(6, 0, -5*np.pi/12, "plots/vis/D_6/contour/K2_type1.png")
plot(6, 1, -5*np.pi/12, "plots/vis/D_6/contour/K2_type2.png")

print("Done!")
