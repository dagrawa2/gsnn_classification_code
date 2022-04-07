import json
import itertools
import numpy as np

ranks = [6, 4, 5, 3, 2, 2, 1]


def mean_loss(gen_arch, reg_arch):
	gen_rank = ranks[gen_arch]
	reg_rank = ranks[reg_arch]
	gen_inits = [i for i in range(-gen_rank, gen_rank+1) if i != 0]
	reg_inits = [i for i in range(-reg_rank, reg_rank+1) if i != 0]
	losses = []
	val_losses = []
	for (gen_init, reg_init) in itertools.product(gen_inits, reg_inits):
		with open("results/C_6/gsnns/archs{:d}_{:d}/inits{:d}_{:d}/results.json".format(gen_arch, reg_arch, gen_init, reg_init), "r") as f:
			D = json.load(f)
		idx = np.array(D["train"]["val_loss"]).argmin()
		losses.append( D["train"]["loss"][idx] )
		val_losses.append( D["train"]["val_loss"][idx] )
	losses = np.array(losses).reshape((len(gen_inits), len(reg_inits)))
	val_losses = np.array(val_losses).reshape((len(gen_inits), len(reg_inits)))
	if gen_arch == reg_arch:
		for i in range(losses.shape[0]):
			losses[i, i] = np.inf
			val_losses[i, i] = np.inf
	losses = losses.min(1)
	val_losses = val_losses.min(1)
	D = {"loss_mean": losses.mean(), "loss_std": losses.std(), "val_loss_mean": val_losses.mean(), "val_loss_std": val_losses.std()}
	return D


archs = [0, 1, 2, 3, 4, 5, 6]
D = {(i, j): mean_loss(i, j) for (i, j) in itertools.product(archs, repeat=2)}

with open("out.csv", "w") as f:
	f.write("reg_arch,gen_arch,train_loss_mean,train_loss_std,val_loss_mean,val_loss_std\n")
	for (i, j) in itertools.product(archs, repeat=2):
		f.write("{:d},{:d},{:.7f},{:.7f},{:.7f},{:.7f}\n".format(i, j, D[(j, i)]["loss_mean"], D[(j, i)]["loss_std"], D[(j, i)]["val_loss_mean"], D[(j, i)]["val_loss_std"]))
