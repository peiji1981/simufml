from typing import Dict, List
import numpy as np
from numpy import array
import torch as th

from simufml.common.comm.communication import Communicator
from simufml.common.cipher_synchronize import PaillierSynchronizer
from simufml.utils.util import RANDOM_SEED
from simufml.secureprotol.encrypt import DummyEncrypt


class HeteroModel(Communicator):
    dummy_cipher = DummyEncrypt()
    roles = None
    variables = None

    def __init__(
        self,
        role: str,
        cipher_sync_class: callable=PaillierSynchronizer,
        cipher_keylen: int=1024,
        skip_cipher: bool=False
    ):
        super().__init__(
            role,
            self.variables
        )

        self.cipher_kenlen = cipher_keylen
        self.cipher_sync = cipher_sync_class(role, self.roles, cipher_keylen)
        self._param_paths = []
        self.skip_cipher = skip_cipher


    async def sync_ciphers(self):
        self.ciphers = await self.cipher_sync.sync()


    def initialize_param(self):
        if RANDOM_SEED.seed is not None:
            th.manual_seed(RANDOM_SEED.seed)
            np.random.seed(RANDOM_SEED.seed)
        self._initialize_param()


    def _initialize_param(self):
        pass
        
        
    def add_param(self, name: str) -> None:
        setattr(self, name, None) # None until being initialized
        self._param_paths.append(name)


    def _getattr_from_path(self, path):
        attr = self
        for p in path.split('.'):
            attr = getattr(attr, p)
        return attr


    @property
    def parameters(self) -> Dict[str, array]:
        res = {}
        for path in self._param_paths:
            res[path] = self._getattr_from_path(path)
        return res


    @property
    def peers(self) -> List[str]:
        return [role for role in self.roles if not role==self.role]


    async def predict(self, batch_x: array) -> array:
        pass

    async def gradient_func(self, batch_x: array, batch_y: array) -> None:
        pass

    async def loss_func(self, batch_x: array, batch_y: array) -> array:
        pass


