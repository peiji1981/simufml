
from typing import Callable
import numpy as np

from simufml.common.comm.communication import Communicator
from simufml.utils.util import Role
from simufml.secureprotol import spdz
from simufml.secureprotol.encrypt import Encrypt
from simufml.secureprotol.encrypt import PaillierEncrypt


class SpdzUtil(Communicator):
    def __init__(
        self,
        role: str,
        peer: str,
        cipher_class: Encrypt=PaillierEncrypt
    ):
        self.share_tag = 'share # '
        self.reconstruct_tag = 'reconstruct # '
        self.en_share_tag = 'en_share # '
        self.cross_tag = 'cross # '
        self.en_x_tag, self.en_y_tag = 'en_x # ', 'en_y # '
        super().__init__(
            role,
            variables=[
                (tag, Role.broadcast, Role.broadcast) for tag in 
                [
                    self.share_tag, 
                    self.reconstruct_tag, 
                    self.en_share_tag,
                    self.cross_tag,
                    self.en_x_tag, self.en_y_tag
                ]
            ]
        )
        self.peer = peer
        self.cipher = cipher_class()
        self.cipher.generate_key()


    async def share(self, fft: 'spdz.ff_tensor.FFTensor', suffix: str=''):
        if not isinstance(fft, spdz.ff_tensor.FFTensor):
            raise ValueError("Can only share a FFTensor object.")
        local_share = fft.random_alike()
        peer_share = fft - local_share
        await self.send(self.share_tag, peer_share, self.peer, suffix)
        return local_share


    async def get_share(self, suffix: str=''):
        res = await self.get(self.share_tag, suffix)
        return res


    async def reconstruct(self, local_share, mode: str='global', suffix: str=''):
        '''
        Reconstruct the initial value from multiple shares. The argument mode detemins who  
        will get the reconstructed value:
        - 'global': all parties will;
        - 'local': only the party itself will;
        - 'remote': the party itself will not.

        Argument:
            - local_share:
        '''
        if not mode in ['global', 'local', 'remote']:
            raise ValueError("Argument mode must be one of 'global'/'local'/'remote'.")

        if mode in ['global', 'remote']:
            await self.send(self.reconstruct_tag, local_share, suffix=suffix)

        if mode in ['global', 'local']:
            peer_share = await self.get(self.reconstruct_tag, suffix=suffix)
            res = local_share + peer_share
            return res


    async def _private_op(self, op: str, x, y, is_active: bool, suffix: str=''):
        '''
        Let two parties be A and B, hold x and y respectively. They call this function coordinately to 
        compute z = f(x,y), and each party gets a share of z. 
        Here f(x,y)=x*y if op=='*', and f(x,y)=x@y if op=='@'.
        One party should pass in x and let y=None, while the other party should pass in y and let x=None.
        One and only one party should be active, and the other should be passive.
        '''
        if (x is None and y is None) or (x is not None and y is not None):
            raise ValueError('One and only one of x/y should be None.')
        if is_active:
            # (1) ACT encrypts it's data
            v = x if x is not None else y
            en_v = spdz.ff_tensor.FFTensor(self.cipher.encrypt(v.fxp))
            # en_v = self.cipher.encrypt(v.fxp)
            # (2) ACT sends encrypted data
            en_v_tag = self.en_x_tag if x is not None else self.en_y_tag
            await self.send(en_v_tag, en_v, suffix=suffix)
            # (6) ACT gets encrypted share of result
            en_z_share = await self.get_share(suffix)
            # (7) ACT decrypts it's share
            # return en_z_share._wrap(self.cipher.decrypt(en_z_share.fxp))
            return spdz.ff_tensor.FFTensor(self.cipher.decrypt(en_z_share.fxp))
        else:
            if not op in ['*', '@']:
                raise ValueError("The argument 'op' can only be '*' or '@'.")
            # (3) PAS receives encrypted data
            en_v_tag = self.en_y_tag if x is not None else self.en_x_tag
            en_v = await self.get(en_v_tag, suffix)
            # (4) PAS computes mul/add operation between encrypted and public data
            operands = (x, en_v) if x is not None else (en_v, y)
            if op=='*':
                en_z = operands[0] * operands[1]
            else:
                en_z = operands[0] @ operands[1]
            # (5) PAS share the encrypted result, it keeps a plaintext random share
            z_share = await self.share(en_z, suffix)
            return z_share


    async def private_mul(self, x, y, is_active: bool, suffix: str=''):
        return await self._private_op('*', x, y, is_active, suffix)


    async def private_matmul(self, x, y, is_active: bool, suffix: str=''):
        return await self._private_op('@', x, y, is_active, suffix)


    async def beaver_triplets(self, a_shape, b_shape, dot_func, suffix: str=''):
        # (1) generate random share of `a` and `b`
        a_local = spdz.FFTensor.random(a_shape)
        b_local = spdz.FFTensor.random(b_shape)
        """
        TODO: write a notebook to explain why do '// 2'.
        """
        a_local.fxp = a_local.fxp // 2
        b_local.fxp = b_local.fxp // 2

        # (2) local-local cross
        cross_local = dot_func(a_local, b_local)

        # (3) local-peer cross
        # (3.1) exchange encrypted a_share
        en_a_local = spdz.FFTensor(self.cipher.encrypt(a_local.fxp))
        await self.send(self.en_share_tag, en_a_local, suffix=suffix)
        en_a_peer = await self.get(self.en_share_tag, suffix)
        # (3.2) encrypted local-peer cross, and send to peer
        r = cross_local.random_alike()
        await self.send(self.cross_tag, dot_func(en_a_peer, b_local) + r, suffix=suffix)
        # (3.3) get encrypted local-peer cross and decrypt
        en_cross_peer = await self.get(self.cross_tag, suffix)
        cross_peer = spdz.FFTensor(self.cipher.decrypt(en_cross_peer.fxp))

        # (4) combine local-local cross and local-peer cross
        c_local = cross_local + cross_peer - r
        return a_local, b_local, c_local


    async def _share_op_share(
        self, 
        op: Callable,
        x: 'spdz.ff_tensor.FFTensor', 
        y: 'spdz.ff_tensor.FFTensor',
        harder_worker: bool=False,
        suffix: str=''
    ):
        """
        TODO: write a notebook to explain this.
        """
        a, b, c = await self.beaver_triplets(
            x.shape, y.shape, op, suffix
        )

        delta = await self.reconstruct(x-a, mode='global')
        epsilon = await self.reconstruct(y-b, mode='global')
        
        z = c + op(a, epsilon) + op(delta, b)
        if harder_worker:
            z = z + op(delta, epsilon)
        return z


    async def share_mul_share(
        self, 
        x: 'spdz.FFTensor', 
        y: 'spdz.FFTensor', 
        harder_worker: bool=False, 
        suffix: str=''
    ):
        op = lambda x,y: x*y
        return await self._share_op_share(op, x, y, harder_worker, suffix)


    async def share_matmul_share(
        self, 
        x: 'spdz.FFTensor', 
        y: 'spdz.FFTensor', 
        harder_worker: bool=False, 
        suffix: str=''
    ):
        op = lambda x,y: x@y
        return await self._share_op_share(op, x, y, harder_worker, suffix)
