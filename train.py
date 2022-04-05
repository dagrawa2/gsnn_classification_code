import gc
import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")  # due to no GPU

import random
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
#from sklearn.model_selection import train_test_split

import grapefruit as gf

# command-line arguments
parser=argparse.ArgumentParser()
# dataset
parser.add_argument('--dimension', '-di', default=6, type=int, help='Dimension of input space.')
parser.add_argument('--n_points', '-np', default=10, type=int, help='Number of points per dimension of hypercube of data points.')
# network architecture
parser.add_argument('--input_dir', '-i', required=True, type=str, help='Directory of architectures.')
parser.add_argument('--generator_idx', '-gi', required=True, type=int, help='Index of generator architecture in input directory.')
parser.add_argument('--regressor_idx', '-ri', required=True, type=int, help='Index of regressor architecture in input directory.')
parser.add_argument('--generator_seed', '-gs', default=0, type=int, help='Pytorch RNG seed for generator network.')
parser.add_argument('--regressor_seed', '-ds', default=0, type=int, help='Pytorch RNG seed for regressor network.')
# SGD hyperparameters
parser.add_argument('--batch_size', '-b', default=128, type=int, help='Minibatch size.')
parser.add_argument('--epochs', '-e', default=10, type=int, help='Number of epochs for training.')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='Learning rate.')
# validation
parser.add_argument('--val_batch_size','-vbs', default=512, type=int, help='Minibatch size during validation/testing.')
parser.add_argument('--val_interval','-vi', default=0, type=int, help='Epoch interval at which to record validation metrics. If 0, test metrics are not recorded.')
# misc
parser.add_argument('--device', '-dv', default="cpu", type=str, help='Device.')
parser.add_argument('--output_dir', '-o', required=True, type=str, help='Output directory.')
parser.add_argument('--exist_ok', '-ok', action="store_true", help='Allow overwriting the output directory.')
args=parser.parse_args()


# fix the random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# record initial time
time_start = time.time()

# create output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=args.exist_ok)
os.makedirs(os.path.join(output_dir, "generator"), exist_ok=args.exist_ok)
os.makedirs(os.path.join(output_dir, "regressor"), exist_ok=args.exist_ok)

# build generating network
generator = gf.nets.GSNN(args.input_dir, args.generator_idx, seed=args.generator_seed, generate_data=True)
trainer = gf.trainers.Trainer(generator, epochs=args.epochs, lr=args.lr, device=args.device)
trainer.save_model(os.path.join(output_dir, "generator/model.pth"))
trainer.save_W(os.path.join(output_dir, "generator/W.npy"))

# generate data
x_train = np.linspace(-1, 1, args.n_points, endpoint=True)
x_val = ( x_train+(x_train[1]-x_train[0])/2 )[:-1]
datasets = []
for x in [x_train, x_val]:
	x = x.astype(np.float32)
	X = np.array(list(map(list, itertools.product(x, repeat=args.dimension))))
	data_loader = DataLoader(TensorDataset(torch.as_tensor(X)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
	Y = gf.trainers.Trainer(generator, epochs=args.epochs, lr=args.lr, device=args.device).predict(data_loader)
	datasets.append((X, Y))
train_loader = DataLoader(TensorDataset(torch.as_tensor(datasets[0][0]), torch.as_tensor(datasets[0][1])), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
val_loader = DataLoader(TensorDataset(torch.as_tensor(datasets[1][0]), torch.as_tensor(datasets[1][1])), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
y_scale = datasets[0][1].std()
del datasets, generator; gc.collect()

# build model
model = gf.nets.GSNN(args.input_dir, args.regressor_idx, seed=args.regressor_seed)

# create trainer and callbacks
trainer = gf.trainers.Trainer(model, epochs=args.epochs, lr=args.lr, y_scale=y_scale, device=args.device)
callbacks = [gf.callbacks.Training()]
if args.val_interval > 0:
	callbacks.append( gf.callbacks.Validation(trainer, val_loader, epoch_interval=args.val_interval) )

# train model
trainer.fit(train_loader, callbacks)

# function to convert np array to list of python numbers
ndarray2list = lambda arr, dtype: [getattr(__builtins__, dtype)(x) for x in arr]

# collect results
results_dict = {
#	"data_shapes": {name: list(A.shape) for (name, A) in [("X_1", X_1), ("X_2", X_2)]}, 
	"train": {key: ndarray2list(value, "float") for cb in callbacks for (key, value) in cb.history.items()}, 
}

# add command-line arguments and script execution time to results
results_dict["args"] = dict(vars(args))
results_dict["time"] = time.time()-time_start

# save results
print("Saving results ...")
with open(os.path.join(output_dir, "results.json"), "w") as fp:
	json.dump(results_dict, fp, indent=2)
trainer.save_model(os.path.join(output_dir, "regressor/model.pth"))
trainer.save_W(os.path.join(output_dir, "regressor/W.npy"))

print("Done!")
