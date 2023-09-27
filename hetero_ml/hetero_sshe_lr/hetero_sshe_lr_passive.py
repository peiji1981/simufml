from typing import Tuple

import numpy as np
from numpy import array

from simufml.utils.util import Role
from simufml.hetero_ml.hetero_sshe_lr.hetero_sshe_lr_base import HeteroSSHELRBase
from simufml.secureprotol.spdz import FFTensor

class HeteroSSHELR_PAS(HeteroSSHELRBase):
    def __init__(
        self,
        feature_size: Tuple[int],
        cipher_keylen: int=1024
    ):
        '''
        Arguments:
            feature_size:
                A tuple of two ints, the first is feature size of ACTIVE party, the second is PASSIVE party.
        '''
        super().__init__(
            role = Role.passive,
            cipher_keylen = cipher_keylen
        )

        self.feature_size = feature_size

        self.add_param('wa_a')
        self.add_param('wb_a')
        self.add_param('bias_a')
        self.initialize_param()


    def _initialize_param(self):
        ACT_fs, PAS_fs = self.feature_size
        self.wa_a = FFTensor.encode(np.random.normal(0, 0.5, PAS_fs))
        self.wb_a = FFTensor.encode(np.random.normal(0, 0.5, ACT_fs))
        self.bias_a = FFTensor.encode(np.random.normal(0, 0.5, 1))


    async def predict(self, batch_x: array, reconstruct: bool=True) -> array:
        '''
        Arguments:
            batch_x:
                [batch_size, feature_size]
        Return:
            The probabilities, an array of shape [batch_size,].
        '''
        batch_x = FFTensor.encode(batch_x)
        za_a = batch_x @ self.wa_a + self.bias_a
        za_b_a = await self.spdz_util.private_matmul(x=batch_x, y=None, is_active=False)
        zb_a_a = await self.spdz_util.private_matmul(x=None, y=self.wb_a, is_active=True)
        self.z_a = za_a + za_b_a + zb_a_a
        
        await self.send('en_z_a', self.z_a.encrypt(self.cipher))
        y_a = (await self.spdz_util.get_share('en_y_a')).decrypt(self.cipher)

        if reconstruct:
            await self.spdz_util.reconstruct(y_a, mode='remote')
        else:
            return y_a
        

    async def gradient_func(self, batch_x: array, batch_y: array):
        self.y_a = await self.predict(batch_x, reconstruct=False)
        e_a = self.y_a
        gwb_a = (await self.spdz_util.get_share('gwb')).decrypt(self.cipher)
        gwa_b_a = await self.spdz_util.private_matmul(x=None, y=batch_x, is_active=False, suffix='gwa_b')
        gwa_a = e_a @ batch_x + gwa_b_a
        self.gradients = {
            'wa_a': gwa_a,
            'wb_a': gwb_a,
            'bias_a': e_a.mean()
        }


    async def loss_func(self, **kwargs):
        '''
        '''
        zy_a = await self.spdz_util.private_mul(x=self.z_a, y=None, is_active=False)
        en_z_b = await self.get('en_z_b')
        en_z_b_squ = await self.get('en_z_b_squ')
        en_z = self.z_a + en_z_b
        en_z_squ = self.z_a**2 + en_z_b_squ + self.z_a*en_z_b*2
        en_L_a = en_z*0.5 - zy_a + en_z_squ*0.125 - np.log(0.5)
        await self.send('en_L_a', en_L_a)