from typing import Dict

import numpy as np
from numpy import array
from numpy import random

from simufml.utils.util import Role
from simufml.hetero_ml.hetero_fm.hetero_fm_base import HeteroFMBase
from .common import base_fm_forward, base_fm_vx, compute_local_grad

class HeteroFM_PAS(HeteroFMBase):
    def __init__(
        self,
        feature_size: int, 
        embed_size: int, 
        total_features: int,
        cipher_keylen: int=1024
    ):
        super().__init__(
            role = Role.passive,
            cipher_keylen = cipher_keylen
        )

        self.feature_size = feature_size
        self.embed_size = embed_size
        self.total_features = total_features

        self.add_param('w')
        self.add_param('embed')
        self.initialize_param()


    def _initialize_param(self):
        fs, es, tfs = self.feature_size, self.embed_size, self.total_features
        self.w = random.normal(
            scale=1/np.sqrt(tfs),
            size=(fs)
        )
        self.embed = random.normal(
            scale=1/np.sqrt(es*tfs*(tfs-1)/2),
            size=(fs, es)
        )


    async def predict(self, batch_x: array, **kwargs) -> array:
        PAS_z = base_fm_forward(batch_x, self.parameters) # [batch_size, ]
        en_PAS_z = self.cipher.encrypt(PAS_z)

        PAS_vx = base_fm_vx(batch_x, self.embed) # [batch_size, embed_size]
        en_PAS_vx = self.cipher.encrypt(PAS_vx)

        # remote to ACTIVE
        await self.send('en_PAS_forward', (en_PAS_z, en_PAS_vx))


    async def gradient_func(self, batch_x: array, **kwargs):
        '''
        Arguments:
            batch_x:
                [batch_size, feature_size]
        '''
        # ACTIVE side
        PAS_z = base_fm_forward(batch_x, self.parameters) # [batch_size, ]
        en_PAS_z = self.cipher.encrypt(PAS_z)

        PAS_vx = base_fm_vx(batch_x, self.embed) # [batch_size, embed_size]
        en_PAS_vx = self.cipher.encrypt(PAS_vx)

        # remote to ACTIVE
        await self.send('en_PAS_forward', (en_PAS_z, en_PAS_vx))

        # get from ASSISTANT
        grad_z = await self.get('grad_z')

        # get from ACTIVE
        en_aggvx = await self.get('en_aggvx')

        grad_w, en_grad_embed, _ = compute_local_grad(
            batch_x, 
            grad_z,
            en_aggvx,
            self.embed
        )

        # remote to ASSISTANT
        await self.send('en_PAS_grad_embed', en_grad_embed)
        # get from ASSISTANT
        grad_embed = await self.get('PAS_grad_embed')

        self.gradients = {
            'w': grad_w,
            'embed': grad_embed
        }


    async def loss_func(self, **kwargs) -> None:
        pass



