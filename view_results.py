import json
import itertools
import numpy as np

archidxs = [3, 4]
seeds = [0, 1, 2, 3, 4, 5]

for (genidx, regidx) in itertools.product(archidxs, repeat=2):
	losses = []
	val_losses = []
	for (genseed, regseed) in itertools.product(seeds, repeat=2):
		with open("results/C_6/gsnns/archs{:d}_{:d}_seeds{:d}_{:d}/results.json".format(genidx, regidx, genseed, regseed), "r") as f:
			D = json.load(f)
		losses.append( D["train"]["loss"][-1] )
		val_losses.append( D["train"]["val_loss"][-1] )
	loss = np.array(losses).reshape((len(seeds), len(seeds))).min(1).mean()
	val_loss = np.array(val_losses).reshape((len(seeds), len(seeds))).min(1).mean()
	print("Archs {:d}, {:d}: {}, {}".format(genidx, regidx, loss, val_loss))
