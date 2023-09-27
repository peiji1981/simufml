from typing import Dict

import numpy as np
from numpy import array
import torch as th

from simufml.utils.util import Role
from simufml.hetero_ml.hetero_nn import HeteroNNBase
from simufml.utils.random_number_generator import RandomNumberGenerator

class HeteroNN_PAS(HeteroNNBase):
    def __init__(
        self,
        bottom_model,
        cipher_keylen: int=1024
    ):
        '''
        Arguments:
        '''
        super().__init__(
            role = Role.passive,
            cipher_keylen = cipher_keylen,
        )

        self.eAcc = None
        self.rng_generator = RandomNumberGenerator()

        self.bottom_model = bottom_model

        for n,_ in self.bottom_model.named_parameters():
            self.add_param(f"bottom_model.{n}")

        self.initialize_param()


    def _initialize_param(self):
        if hasattr(self.bottom_model, 'initialize'): 
            self.bottom_model.initialize()


    async def predict(self, batch_x: array) -> array:
        self.bottom_model.eval()
        await self.forward(batch_x)
        self.bottom_model.train()


    async def forward(self, batch_x: array) -> array:
        # bottom model forward
        self.aA = self.bottom_model(batch_x) # [batch_size, botB_out_size]

        # mix layer forward
        # Attention: do aA.detach().numpy() before go to mix forward, because 
        # Tensor is not supported very well by cipher encryption/decryption.
        await self._mix_forward(self.aA.detach().numpy())


    async def _mix_forward(self, aA):
        # [[aA]]
        en_aA = self.cipher.encrypt(aA)
        # send to ACTIVE
        await self.send('en_aA', en_aA)

        # get from ACTIVE: [[w_A * α_A + ε_B]]
        en_wAaA_plus_eB = await self.get('en_wAaA_plus_eB')

        # w_A * α_A + ε_B
        wAaA_plus_eB = self.cipher.decrypt(en_wAaA_plus_eB).astype(np.float32)

        if self.eAcc is None:
            self.eAcc = np.zeros(shape=(
                en_wAaA_plus_eB.shape[1],   # output_size
                aA.shape[1]),               # input_size
                dtype=np.float32
            )

        # wAaA_plus_eB_plus_aAeAcc
        wAaA_plus_eB_plus_aAeAcc = wAaA_plus_eB + np.dot(aA, self.eAcc.T)
        await self.send('wAaA_plus_eB_plus_aAeAcc', wAaA_plus_eB_plus_aAeAcc)
        

    async def _mix_backward(self):
        # get from ACTIVE: [[g_z_A * α_A + ε_B]]
        en_gzAaA_plus_eB = await self.get('en_gzAaA_plus_eB')
        # decrypt
        gzAaA_plus_eB = self.cipher.decrypt(en_gzAaA_plus_eB).astype(np.float32)

        # get from ACTIVE: ŋ
        eta = await self.get('learning_rate')
        # generate noise: ε_A
        eA = self.rng_generator.generate_random_number(gzAaA_plus_eB.shape)
        # compute: g_z_A * α_A + ε_B + ε_A/ŋ
        gzAaA_plus_eB_plus_eA_ov_eta = gzAaA_plus_eB + eA/eta
        # send to ACTIVE
        await self.send('gzAaA_plus_eB_plus_eA_ov_eta', gzAaA_plus_eB_plus_eA_ov_eta)

        # [[ε_acc]]
        en_eAcc = self.cipher.encrypt(self.eAcc)
        # send to ACTIVE
        await self.send('en_eAcc', en_eAcc)
        # update self.eAcc
        self.eAcc += eA

        # get from ACTIVE: [[g_α_A]]
        en_gaA = await self.get('en_gaA')
        # decrypt
        self.gaA = self.cipher.decrypt(en_gaA)


    async def gradient_func(self, batch_x: array, **kwargs):
        await self.loss_func(batch_x, compute_and_regist=True)

        # get grad of α_A
        await self._mix_backward()

        # backward propagation of bottom model
        self.aA.backward(th.tensor(self.gaA))

        self.gradients = {}
        for k,p in self.parameters.items():
            self.gradients[k] = p.grad
            p.grad = None


    async def loss_func(self, batch_x: array, compute_and_regist: bool=False, **kwargs):
        if (not compute_and_regist) and (self._registed_loss is not None):
            loss = self._registed_loss
            self._registed_loss = None
            return loss

        await self.forward(batch_x)

        if compute_and_regist:
            self._registed_loss = 0
