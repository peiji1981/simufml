
from simufml.common.cipher_synchronize import PaillierSynchronizer
from simufml.common.hetero_model import HeteroModel
from simufml.secureprotol.spdz import SpdzUtil


class HeteroSSHEModel(HeteroModel):
    def __init__(
        self,
        role: str,
        cipher_sync_class: callable=PaillierSynchronizer,
        cipher_keylen: int=1024,
        skip_cipher: bool=False,
    ):
        super().__init__(
            role,
            cipher_sync_class,
            cipher_keylen,
            skip_cipher
        )
        self.spdz_util = SpdzUtil(role, self.peers)


