from typing import Tuple

import numpy as np
from numpy import array

from simufml.utils.util import Role
from simufml.hetero_ml.hetero_sshe_lr.hetero_sshe_lr_base import HeteroSSHELRBase
from simufml.secureprotol.spdz import FFTensor

class HeteroSSHELR_ACT(HeteroSSHELRBase):
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
            role = Role.active,
            cipher_keylen = cipher_keylen
        )

        self.feature_size = feature_size

        # 'a' for PASSIVE party, 'b' for ACTIVE party
        # the 'a' in 'wa' means this parameter is crosponding to features of party a
        # the 'b' in 'wa_b' means this is the fraction holded by party b
        self.add_param('wa_b')
        self.add_param('wb_b')
        self.add_param('bias_b')
        self.initialize_param()


    def _initialize_param(self):
        ACT_fs, PAS_fs = self.feature_size
        self.wa_b = FFTensor.encode(np.random.normal(0, 0.5, PAS_fs))
        self.wb_b = FFTensor.encode(np.random.normal(0, 0.5, ACT_fs))
        self.bias_b = FFTensor.encode(np.random.normal(0, 0.5, 1))


    async def predict(self, batch_x: array, reconstruct: bool=True) -> array:
        '''
        Arguments:
            batch_x:
                [batch_size, feature_size]
        Return:
            The probabilities, an array of shape [batch_size,].
        '''
        batch_x = FFTensor.encode(batch_x)
        zb_b = batch_x @ self.wb_b + self.bias_b
        za_b_b = await self.spdz_util.private_matmul(x=None, y=self.wa_b, is_active=True)
        zb_a_b = await self.spdz_util.private_matmul(x=batch_x, y=None, is_active=False)
        self.z_b = zb_b + za_b_b + zb_a_b

        en_z_a = await self.get('en_z_a')
        self.en_z = en_z_a + self.z_b
        self.en_y = self.en_z * 0.25 + 0.5
        y_b = await self.spdz_util.share(self.en_y, 'en_y_a')

        if reconstruct:
            y = await self.spdz_util.reconstruct(y_b, mode='local')
            return y.decode()
        else:
            return y_b
        

    async def gradient_func(self, batch_x: array, batch_y: array):
        self.y_b = await self.predict(batch_x, reconstruct=False)
        e_b = self.y_b - batch_y

        en_e = self.en_y - batch_y
        
        en_gwb = en_e @ batch_x
        gwb_b = await self.spdz_util.share(en_gwb, 'gwb')

        gwa_b_b = await self.spdz_util.private_matmul(x=e_b, y=None, is_active=True, suffix='gwa_b')
        gwa_b = gwa_b_b

        self.gradients = {
            'wa_b': gwa_b,
            'wb_b': gwb_b,
            'bias_b': e_b.mean()
        }


    async def loss_func(self, batch_y, **kwargs):
        '''
        '''
        batch_y = FFTensor.encode(batch_y)
        zy_b = self.z_b * batch_y
        zy_a_b = await self.spdz_util.private_mul(x=None, y=batch_y, is_active=True)
        zy_b = zy_b + zy_a_b
        await self.send('en_z_b', self.z_b.encrypt(self.cipher))
        await self.send('en_z_b_squ', (self.z_b*self.z_b).encrypt(self.cipher))
        L_a = (await self.get('en_L_a')).decrypt(self.cipher)
        loss = L_a - zy_b
        return np.mean(loss.decode())