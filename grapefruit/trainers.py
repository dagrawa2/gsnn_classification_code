import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from . import utils


class Trainer(object):

	def __init__(self, model, epochs=1, lr=1e-3, y_scale=1, device="cpu"):
		self.model = model.to(device)
		self.epochs = epochs
		self.lr = lr
		self.y_scale = y_scale
		self.device = device

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.loss_fn = lambda preds, targets: (preds-targets).pow(2).mean()/self.y_scale**2

	def fit(self, train_loader, callbacks):
		# callbacks before training
		callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
		for cb in callbacks:
			cb.start_of_training()
		print("Training model ...")
		print("---")
		for epoch in range(self.epochs):
			# callbacks at the start of the epoch
			for cb in callbacks:
				cb.start_of_epoch(epoch)
			batch_logs = {"loss": []}
			for (X_batch, Y_batch) in train_loader:
				X_batch = X_batch.to(self.device)
				Y_batch = Y_batch.to(self.device)
				preds_batch = self.model(X_batch)
				loss = self.loss_fn(preds_batch, Y_batch)
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
				batch_logs["loss"].append(loss.item())
				with torch.no_grad():
					self.model.W_unconstrained.data = utils.normalize(self.model.W_unconstrained.data)
			# callbacks at the end of the epoch
			for cb in callbacks:
				cb.end_of_epoch(epoch, batch_logs)

		# callbacks at the end of training
		for cb in callbacks:
			cb.end_of_training()
		print("---")


	def evaluate(self, val_loader):
		preds = []
		with torch.no_grad():
			for (X_batch, _) in val_loader:
				X_batch = X_batch.to(self.device)
				pred_batch = self.model(X_batch)
				preds.append(pred_batch)
			preds = torch.cat(preds)

			loss = self.loss_fn(preds, val_loader.dataset.tensors[1]).item()

		return loss


	def predict(self, data_loader):
		preds = []
		with torch.no_grad():
			for (X_batch, ) in data_loader:
				X_batch = X_batch.to(self.device)
				preds_batch = self.model(X_batch)
				preds.append( preds_batch.cpu().numpy() )
		preds = np.concatenate(preds, 0)

		return preds


	def save_model(self, filename):
		torch.save(self.model.state_dict(), filename)


	def load_model(self, filename):
		self.model.load_state_dict(torch.load(filename))

	def save_W(self, filename):
		with torch.no_grad():
			W = torch.einsum("i,ijk->jk", self.model.W_unconstrained, self.model.Ws).cpu().numpy()
		np.save(filename, W)
