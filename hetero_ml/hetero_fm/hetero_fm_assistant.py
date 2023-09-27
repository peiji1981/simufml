import numpy as np

from simufml.utils.util import Role
from simufml.hetero_ml.hetero_fm.hetero_fm_base import HeteroFMBase

class HeteroFM_ASS(HeteroFMBase):
    def __init__(
        self, 
        cipher_keylen: int=1024
    ):
        super().__init__(
            role = Role.assistant,
            cipher_keylen = cipher_keylen
        )


    async def gradient_func(self, **kwargs) -> None:
        # get from ACTIVE
        en_grad_z = await self.get('en_grad_z')
        grad_z = self.cipher.decrypt(en_grad_z).clip(-1., 1.)

        # remote to ['ACTIVE', 'PASSIVE']
        await self.send('grad_z', grad_z)

        # get from ACTIVE
        en_ACT_grad_embed = await self.get('en_ACT_grad_embed')
        ACT_grad_embed = self.cipher.decrypt(en_ACT_grad_embed)
        # remote to ACTIVE
        await self.send('ACT_grad_embed', ACT_grad_embed)

        # get from PASSIVE
        en_PAS_grad_embed = await self.get('en_PAS_grad_embed')
        PAS_grad_embed = self.cipher.decrypt(en_PAS_grad_embed)
        # remote to PASSIVE
        await self.send('PAS_grad_embed', PAS_grad_embed)


    async def loss_func(self, **kwargs) -> None:
        # get from ACTIVE
        en_yz = await self.get('en_yz')
        yz = self.cipher.decrypt(en_yz)

        loss = np.log(2) - yz/2 + (yz**2)/8
        loss = loss.mean()

        # remote to ACTIVE
        await self.send('loss', loss)


    async def predict(self, **kwargs):
        en_agg_z = await self.get('en_agg_z')
        agg_z = self.cipher.decrypt(en_agg_z)
        await self.send('agg_z', agg_z)
