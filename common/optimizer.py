import numpy as np
import torch as th

from simufml.common.hetero_model import HeteroModel

class SGD:
    def __init__(self, model: HeteroModel=None, lr: float=1e-3, wd: float=0, grad_clip: float=-1):
        '''
        Arguments:
            model:
            lr:
            wd:
                L2 norm regularization coefficient. We are using weight-decay schema, instead of L2-norm schema.
                The weight-decay schema directly update weights; while L2-norm schema update gradients.
                In vanila SGD, the effection of this argument on loss is : 
                    L = loss + 1/2 * wd * weight^2;
                or on gradient is:
                    g = g + wd * weight
                or on weight is:
                    weight = (1 - lr * wd) * weight
        '''
        self.model = model
        self.lr = lr
        self.wd = wd
        self.grad_clip = grad_clip
        self._epoch = 0
        self._iter = 0

    def step(self):
        '''
        We should define a base class named Optimizer, and this function `step` should be defined there.
        And user only need to write `_update` function.
        '''
        for k in self.model._param_paths:
            param = self.model.parameters[k]
            grad = self.model.gradients[k]
            self.update_inplace(
                param.data if isinstance(param, th.Tensor) else param, 
                grad
            )


    def update_inplace(self, param, grad):
        param -= param * self.lr * self.wd
        if self.grad_clip > 0:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        param -= grad * self.lr
             


