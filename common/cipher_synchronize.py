from typing import List, Tuple

# simufml
from simufml.secureprotol.encrypt import PaillierEncrypt
from simufml.utils.util import Role
from simufml.common.comm.communication import Communicator

class CipherSynchronizer(Communicator):
    cipher_class: callable = None

    def __init__(
        self,
        role: str,
        roles: List[str],
        key_length: int=1024
    ):
        super().__init__(
            role=role, 
            variables=[(f'{r}_pubkey', r, Role.broadcast) for r in roles]
        )
        self.roles = roles
        self.key_length = key_length


    async def sync(self):
        cipher = self.cipher_class()
        cipher.generate_key(self.key_length)
        pubkey = cipher.get_public_key()
        await self.send(f'{self.role}_pubkey', pubkey)
        ciphers = {self.role: cipher}
        for peer in self.roles:
            if peer==self.role:
                continue
            ciphers[peer] = self.cipher_class()
            pubkey = await self.get(f'{peer}_pubkey')
            ciphers[peer].set_public_key(pubkey)
        return ciphers


class PaillierSynchronizer(CipherSynchronizer):
    cipher_class = PaillierEncrypt
