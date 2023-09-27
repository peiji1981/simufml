from typing import Dict

import numpy as np
from numpy import array
from numpy import random

from simufml.utils.util import Role
from simufml.utils.activation import sigmoid
from simufml.hetero_ml.hetero_fm.common import base_fm_forward, base_fm_vx, compute_local_grad
from simufml.hetero_ml.hetero_fm.hetero_fm_base import HeteroFMBase

class HeteroFM_ACT(HeteroFMBase):
    def __init__(
        self,
        feature_size: int, 
        embed_size: int, 
        total_features: int,
        cipher_keylen: int=1024
    ):
        '''
        Arguments:
            total_features:
                The number of features of PASSIVE and ACTIVE. This is used to initialize parameters.
        '''
        super().__init__(
            role = Role.active,
            cipher_keylen = cipher_keylen
        )

        self.feature_size = feature_size
        self.embed_size = embed_size
        self.total_features = total_features

        self.add_param('w')
        self.add_param('embed')
        self.add_param('b')
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
        self.b = np.array([0.])


    async def predict(self, batch_x: array) -> array:
        '''
        Arguments:
            batch_x:
                [batch_size, feature_size]
        Return:
            The probabilities, an array of shape [batch_size,].
        '''
        ACT_z = base_fm_forward(batch_x, self.parameters) # [batch_size, ]
        en_ACT_z = self.cipher.encrypt(ACT_z)

        ACT_vx = base_fm_vx(batch_x, self.embed) # [batch_size, embed_size]

        # get from PASSIVE
        en_PAS_z, en_PAS_vx = await self.get('en_PAS_forward')

        # aggregate z
        en_agg_z = en_ACT_z + en_PAS_z + np.multiply(ACT_vx, en_PAS_vx).sum(1) # [batch_size, ]
        await self.send('en_agg_z', en_agg_z) # [batch_size, ]
        agg_z = await self.get('agg_z')

        proba = sigmoid(agg_z)
        
        return proba


    @staticmethod
    def process_y(batch_y):
        '''
        make sure that y ∈ (-1, +1)
        '''
        return (batch_y==1)*2 - 1
        

    async def gradient_func(self, batch_x: array, batch_y: array):
        batch_y = self.process_y(batch_y)

        ACT_z = base_fm_forward(batch_x, self.parameters) # [batch_size, ]
        en_ACT_z = self.cipher.encrypt(ACT_z)

        ACT_vx = base_fm_vx(batch_x, self.embed) # [batch_size, embed_size]
        en_ACT_vx = self.cipher.encrypt(ACT_vx)

        # get from PASSIVE
        en_PAS_z, en_PAS_vx = await self.get('en_PAS_forward')

        # aggregate z
        en_agg_z = en_ACT_z + en_PAS_z + np.multiply(ACT_vx, en_PAS_vx).sum(1) # [batch_size, ]
        self.en_agg_z = en_agg_z # for loss computation

        # gradient of Loss on z
        en_grad_z = 0.25 * en_agg_z - 0.5 * batch_y # shape: [batch_size, ]
        # send to ASSISTANT
        await self.send('en_grad_z', en_grad_z)
        # get from ASSISTANT
        grad_z = await self.get('grad_z')

        # aggregrate vx
        en_aggvx = en_PAS_vx + en_ACT_vx
        # remote to PASSIVE
        await self.send('en_aggvx', en_aggvx)

        grad_w, en_grad_embed, grad_b = compute_local_grad(
            batch_x, 
            grad_z, 
            en_aggvx,
            self.embed
        )

        # remote to ASSISTANT
        await self.send('en_ACT_grad_embed', en_grad_embed)
        # get from ASSISTANT
        grad_embed = await self.get('ACT_grad_embed')

        self.gradients = {
            'w': grad_w,
            'embed': grad_embed,
            'b': grad_b
        }


    async def loss_func(self, batch_y: array, **kwargs):
        '''
        Regularization loss (weigth norm) is not included. I consider regularization as part of optimization.
        This way, the passive role doesn't need to send en_PAS_norm to ACTIVE, this will save some computation cost.

        Auguments:
            batch_y: 
                Lables of a batch. [batch_size, ]
            
        Return:
            The sum of losses of the batch.
        '''
        batch_y = self.process_y(batch_y)

        en_yz = self.en_agg_z * batch_y # scalar

        # remote to ASSISTANT
        await self.send("en_yz", en_yz)

        # get from ASSISTANT
        loss = await self.get('loss')
        return loss
