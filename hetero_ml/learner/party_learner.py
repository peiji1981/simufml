
from typing import Dict
import torch as th
import numpy as np

from simufml.utils.util import Role
from simufml.common.hetero_model import HeteroModel
from simufml.common.comm.communication import Communicator
from simufml.common.dataloader import DataLoader


class PartyLearner(Communicator):
    def __init__(
        self, 
        role: str,
        model: HeteroModel, 
        train_dataloader: DataLoader=None,
        valid_dataloader: DataLoader=None,
        optimizer=None
    ):
        super().__init__(
            role=role
            )
        self.model = model
        self.adapt_dataloaders(train_dataloader, valid_dataloader)
        self.adapt_optimizer(optimizer)


    def adapt_dataloaders(self, train_dataloader, valid_dataloader):
        self.train_dataloader = DataLoader() if train_dataloader is None else train_dataloader
        self.valid_dataloader = DataLoader() if valid_dataloader is None else valid_dataloader
        self.train_dataloader.role = self.role
        self.valid_dataloader.role = self.role


    def adapt_optimizer(self, optimizer):
        if optimizer is not None:
            optimizer.model = self.model
        self.optimizer = optimizer


    async def sync_ciphers(self):
        await self.model.sync_ciphers()


    async def fit(self, epochs: int, metrics: Dict):
        if self.role==Role.active:
            print_fit_head(metrics.keys())
        for self._epoch in range(epochs):
            await self.fit_one_epoch()
            await self.validate(metrics.values())
            if self.role==Role.active:
                print_fit_line(self._epoch, self.epoch_loss, self.metric_results)


    async def fit_one_epoch(self):
        await self.train_dataloader.sync_batches()

        losses = []
        for self._iter, (batch_x, batch_y) in enumerate(self.train_dataloader):
            await self.model.gradient_func(batch_x=batch_x, batch_y=batch_y)
            if self.optimizer is not None:
                self.optimizer.step()

            loss = await self.model.loss_func(batch_x=batch_x, batch_y=batch_y)
            losses.append(0 if loss is None else loss)

        self.epoch_loss = np.mean(losses)


    async def validate(self, metrics):
        await self.predict(self.valid_dataloader)
        if self.role==Role.active:
            self.metric_results = []
            for func in metrics:
                res = func(self.valid_dataloader.dataset.Y, self.preds)
                self.metric_results.append(res)


    async def predict(self, dataloader=None):
        dataloader = self.test_dataloader if dataloader is None else dataloader
        with dataloader.no_shuffle():
            await dataloader.sync_batches()
            preds = []
            for batch_x, _ in dataloader:
                pred = await self.model.predict(batch_x=batch_x)
                pred = pred.detach().numpy() if isinstance(pred, th.Tensor) else pred
                preds.append(pred)
            if self.role==Role.active:
                self.preds = np.concatenate(preds)



def print_fit_head(metric_names):
    format_str = ''.join(['{:<7}', '{:<15}'] + ["{:<15}" for _ in metric_names])
    head_str = ['epoch', 'train_loss'] + [name[:14] for name in metric_names]
    print(format_str.format(*head_str))


def print_fit_line(epoch, train_loss, metrics):
    format_str = ''.join(['{:<7}', '{:<15.3e}'] + ["{:<15.3e}" for _ in metrics])
    values = [epoch, train_loss] + metrics
    print(format_str.format(*values))