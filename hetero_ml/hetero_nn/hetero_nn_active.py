from typing import Tuple

import numpy as np
from numpy import array
import torch as th
from torch import nn
from torch.nn import functional as F

from simufml.utils.util import Role
from simufml.hetero_ml.hetero_nn import HeteroNNBase
from simufml.utils.random_number_generator import RandomNumberGenerator

class HeteroNN_ACT(HeteroNNBase):
    def __init__(
        self,
        bottom_model,
        top_model,
        mix_layer_io_shape: Tuple[int],
        loss_func: callable,
        # TODO: It feels not prefect to put `lr` in model. What's a better way? 
        lr: float=1e-3,
        mix_layer_active: callable=F.relu,
        cipher_keylen: int=1024
    ):
        '''
        Arguments:
            bottom_model:
            top_model:
            mix_layer_io_shape:
                A tuple of three ints, which are 
                    (1) output dim of ACTIVE's bottom model, 
                    (2) output dim of PASSIVE's bottom model, and
                    (3) number of neurons of mix layer
            loss_func:
            mix_layer_active:
                Default relu. If is None, means no activate function.
            paillier_keylen:

        '''
        super().__init__(
            role = Role.active,
            cipher_keylen = cipher_keylen
        )

        self.bottom_model = bottom_model
        self.top_model = top_model
        # TODO: Looks bad to name it self._loss_func, but we have define
        # a method with this name. So, we need to think about this later.
        self._loss_func = loss_func
        self.lr = lr
        self.mix_layer_active = mix_layer_active
        self.mix_in_ACT, self.mix_in_PAS, self.mix_out = mix_layer_io_shape
        self.mix_wA = nn.Parameter(th.empty(self.mix_out, self.mix_in_PAS))
        self.mix_wB = nn.Parameter(th.empty(self.mix_out, self.mix_in_ACT))
        self.mix_bias = nn.Parameter(th.empty(self.mix_out))

        for k in ('top_model', 'bottom_model'):
            m = getattr(self, k)
            for n,_ in m.named_parameters():
                self.add_param(f"{k}.{n}")
        
        self.rng_generator = RandomNumberGenerator()
        self.initialize_param()


    def _initialize_param(self):
        for m in (self.top_model, self.bottom_model):
            if hasattr(m, 'initialize'): 
                m.initialize()

        _mix_layer = nn.Linear(self.mix_in_ACT+self.mix_in_PAS, self.mix_out)    
        _mix_layer.reset_parameters()
        self.mix_wA.data, self.mix_wB.data = _mix_layer.weight.data.split((self.mix_in_PAS, self.mix_in_ACT), 1)
        self.mix_bias.data = _mix_layer.bias.data


    @ property
    def np_wA(self):
        return self.mix_wA.detach().numpy()


    async def predict(self, batch_x: array) -> array:
        self.top_model.eval()
        self.bottom_model.eval()
        aT = await self.forward(batch_x)
        self.top_model.train()
        self.top_model.train()
        return th.sigmoid(aT)


    async def forward(self, batch_x: array) -> array:
        # aB: forward output of ACTIVE's bottom model
        aB = self.bottom_model(batch_x) # [batch_size, botB_out_size]

        # aI: forward output of Interactive layer
        aI = await self._mix_forward(aB)

        # aT: forward output of ACTIVE's top model
        aT = self.top_model(aI)

        return aT


    async def _mix_forward(self, aB):
        '''
        Argument:
            aB:
                Forward output of bottom model of ACTIVE(Party B). 
                Shape: [batch_size, input_size_B]
            en_aA:
                Encrypted forward output of bottom model of PASSIVE(Party A). 
                Shape: [batch_size, input_size_A]
        '''
        batch_size = aB.shape[0]
        output_size = self.np_wA.shape[0]

        # en_aA: get from PASSIVE, encrypted forward output of PASSIVE's bottom model
        en_aA = await self.get('en_aA')
        # store en_aA as attribute, it's will be needed when do mix backward
        self.en_aA = en_aA

        # compute [[w_A * α_A + ε_B]]
        eB = self.rng_generator.generate_random_number((batch_size, output_size))
        en_wAaA_plus_eB = np.dot(en_aA, self.np_wA.T) + eB # [batch_size, output_size]
        # send to PASSIVE
        await self.send('en_wAaA_plus_eB', en_wAaA_plus_eB)

        # get from PASSIVE: w_A * α_A + ε_B + ε_acc * α_A
        wAaA_plus_eB_plus_aAeAcc = await self.get('wAaA_plus_eB_plus_aAeAcc')
        
        # zA = (wA + ε_acc) * α_A
        zA = wAaA_plus_eB_plus_aAeAcc - eB

        # zB = wB * aB
        zB = F.linear(aB, self.mix_wB, self.mix_bias) # [batch_size, output_size]

        # z = zA + zB
        # make zA an Tensor that requires grad, so we can get it's grad after backward
        self.zA = th.tensor(zA, requires_grad=True)
        z = self.zA + zB

        # aI: activation of mix layer
        aI = z
        if self.mix_layer_active is not None:
            aI = self.mix_layer_active(aI)
        return aI
        

    async def _mix_backward(self):
        # compute: [[g_z_A * α_A + ε_B]]
        eB = self.rng_generator.generate_random_number(self.np_wA.shape)
        """
        For: 
            X : (batch_size, in_size)
            w : (out_size, in_size)
        ----------------------------------
        Forward:  Z                      =  X                     * w.T
        shape:    (batch_size, out_size) <- (batch_size, in_size) * (in_size, out_size)
        ----------------------------------
        Backward: d_w                 =  (d_Z).T                * X
        shape:    (out_size, in_size) <- (out_size, batch_size) * (batch_size, in_size)
        """
        en_gzAaA_plus_eB = np.dot(self.zA.grad.T, self.en_aA) + eB
        # send to Host
        await self.send('en_gzAaA_plus_eB', en_gzAaA_plus_eB)

        # send to Host
        await self.send('learning_rate', self.lr)

        # get from host: g_z_A * α_A + ε_B + ε_A/ŋ 
        gzAaA_plus_eB_plus_eA_ov_eta = await self.get('gzAaA_plus_eB_plus_eA_ov_eta')

        # compute: g_z_A * α_A + ε_A/ŋ
        gzAaA_plus_eA_ov_eta = gzAaA_plus_eB_plus_eA_ov_eta - eB
        self.mix_wA.grad = th.tensor(gzAaA_plus_eA_ov_eta)
        
        # get from host: [[ε_acc]]
        en_eAcc = await self.get('en_eAcc')

        # compute: [[g_α_A]] = g_z_A * ([[wA]] + [[ε_acc]])
        """
        For: 
            X : (batch_size, in_size)
            w : (out_size, in_size)
        ----------------------------------
        Forward:  Z                      =  X                     * w.T
        shape:    (batch_size, out_size) <- (batch_size, in_size) * (in_size, out_size)
        ----------------------------------
        Backward: d_X                   =  d_Z                    * w
        shape:    (batch_size, in_size) <- (batch_size, out_size) * (out_size, in_size)
        """
        # en_gaA = np.dot(self.zA.grad, self.cipher.encrypt(self.np_wA) + en_eAcc)
        en_gaA = np.dot(self.zA.grad, self.np_wA + en_eAcc)
        # send to host
        await self.send('en_gaA', en_gaA)

    # TODO: Is it possible to do this in optimizer? Would it be a better way?
    def _update_mix_layer(self):
        self.mix_wA.data -= self.lr * self.mix_wA.grad
        self.mix_wB.data -= self.lr * self.mix_wB.grad
        self.mix_bias.data -= self.lr * self.mix_bias.grad

        self.mix_wA.grad = None
        self.mix_wB.grad = None
        self.mix_bias.grad = None


    async def gradient_func(self, batch_x: array, batch_y: array):
        loss = await self.loss_func(batch_x, batch_y, compute_and_regist=True)

        # backward propagation to the start of ACTIVE's branch, and to zA of the Host's branch 
        loss.backward()

        await self._mix_backward()

        self._update_mix_layer()

        self.gradients = {}
        for k,p in self.parameters.items():
            self.gradients[k] = p.grad
            p.grad = None


    async def loss_func(self, batch_x: array, batch_y: array, compute_and_regist: bool=False):
        '''
        Arguments:
            compute_and_regist:

        '''
        if (not compute_and_regist) and (self._registed_loss is not None):
            loss = self._registed_loss
            self._registed_loss = None
            return loss

        aT = await self.forward(batch_x)
        loss = self._loss_func(aT.squeeze(), batch_y)
            
        if compute_and_regist:
            self._registed_loss = loss.detach()
        
        return loss
