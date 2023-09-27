"""ABY3 Protocol.

ABY3 : A Mixed Protocol Framework for Machine Learning.
https://eprint.iacr.org/2018/403.pdf
"""
from math import ceil, log2
from typing import List, Tuple

from simufml.common.comm.communication import Communicator
from simufml.secureprotol import aby3
from simufml.secureprotol.aby3.aby3_tensor import ABY3Tensor
from simufml.utils.util import Role
from simufml.secureprotol.encrypt import Encrypt
from simufml.secureprotol.encrypt import PaillierEncrypt

import numpy as np

N_PARTIES = 3  # 3-party MPC protocol

# ===== Share types =====
ARITHMETIC = 0
BOOLEAN = 1


def logical_rshift_private(x: List[ABY3Tensor], steps: int) -> List[ABY3Tensor]:
    return [share.logical_rshift(steps) for share in x]


def rshift_private(x: List[ABY3Tensor], steps: int) -> List[ABY3Tensor]:
    return [share >> steps for share in x]


def lshift_private(x: List[ABY3Tensor], steps) -> List[ABY3Tensor]:
    return [share << steps for share in x]


def add_private_private(x: List[ABY3Tensor], y: List[ABY3Tensor]) -> List[ABY3Tensor]:
    z = [None, None]
    z[0] = x[0] + y[0]
    z[1] = x[1] + y[1]
    return z


def sub_private_private(x: List[ABY3Tensor], y: List[ABY3Tensor]) -> List[ABY3Tensor]:
    z = [None, None]
    z[0] = x[0] - y[0]
    z[1] = x[1] - y[1]
    return z


def negative_private(x: List[ABY3Tensor]) -> List[ABY3Tensor]:
    z = [None, None]
    z[0] = - x[0]
    z[1] = - x[1]
    return z


def B_and_private_public(x: List[ABY3Tensor], y) -> List[ABY3Tensor]:
    z = [None, None]
    z[0] = x[0] & y
    z[1] = x[1] & y
    return z


# local operation
def B_xor_private_private(x: List[ABY3Tensor], y: List[ABY3Tensor]) -> List[ABY3Tensor]:
    z = [None, None]
    z[0] = x[0] ^ y[0]
    z[1] = x[1] ^ y[1]
    return z


class ABY3Util(Communicator):
    def __init__(
            self,
            role: str,
            peer: str,
            helper: str,
            cipher_class: Encrypt = PaillierEncrypt
    ):
        """ perfomrs ABY3 protocol
        Args:
            role (str): The active role holds data, responding to P1 in ABY3's paper
            peer (str): The passive role is a party, responding to P2
            helper (str): The assistant role, responding to P3
        """
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
        self.role = role
        self.peer = peer
        self.helper = helper
        self.cipher = cipher_class()
        self.cipher.generate_key()
        # self.pairwise_keys = []
        # print("generate aby3 object")
        # self.b2a_keys_1, self.b2a_keys_2 = self.generate_b2a_generator()

    async def share(self, x: ABY3Tensor, share_type: int, factory=np.int64,
                    suffix: str = '' ) -> List[ABY3Tensor]:

        if not isinstance(x, ABY3Tensor):
            raise ValueError("Can only share a ABY3Tensor object.")

        # TODO: boolean sharing needs to generate boolean distribution
        share0 = x.random_alike(factory)
        share1 = x.random_alike(factory)

        if share_type == ARITHMETIC:
            share2 = x - share0 - share1
        elif share_type == BOOLEAN:
            share2 = x ^ share0 ^ share1
        else:
            raise NotImplementedError

        local_share = [share0, share1]

        # replicated sharing
        await self.send(self.share_tag, [share1, share2], dsts=[self.peer], suffix=suffix)
        await self.send(self.share_tag, [share2, share0], dsts=[self.helper], suffix=suffix)

        return local_share

    async def get_share(self, src: str, suffix: str = '') -> List[ABY3Tensor]:
        res = await self.get(self.share_tag, suffix, src)
        return res

    async def reshare(self, local_share: ABY3Tensor, suffix: str = '') -> List[ABY3Tensor]:
        await self.send(self.share_tag, local_share, dsts=[self.helper])
        remote_share = await self.get(self.share_tag, suffix=suffix, srcs=[self.peer])
        return [local_share, remote_share]

    async def reconstruct(self, local_share, share_type: int = ARITHMETIC,
                          mode: str = 'global', suffix: str = '') -> ABY3Tensor:
        """
        Reconstruct the initial value from multiple shares.

        Argument:
            - local_share: local share part
            - mode: 'global': each party will get others' share to reconstruct;
                    'local':  the party will get others' share to reconstruct but do not send self share;
                    'remote': send self to others.
            return: secret
        """

        def _re(s0, s1, s2):
            if share_type == ARITHMETIC:
                return s0 + s1 + s2
            elif share_type == BOOLEAN:
                return s0 ^ s1 ^ s2
            else:
                raise NotImplementedError(
                    "Only arithmetic and boolean sharings are supported."
                )

        if not mode in ['global', 'local', 'remote']:
            raise ValueError("Argument mode must be one of 'global'/'local'/'remote'.")

        if mode in ['global', 'remote']:
            # send to local_share to next party, in this framework is peer
            await self.send(self.reconstruct_tag, local_share, dsts=[self.peer], suffix=suffix)

        if mode in ['global', 'local']:
            # get local_share from previous party, in this framework is helper
            peer_share = await self.get(self.reconstruct_tag, suffix=suffix, srcs=self.helper)
            # TODO: only use peer_share[0], so the communication bandWidth can be reduce 1/2 of peer_share
            return _re(local_share[0], local_share[1], peer_share[0])

    async def generate_pairwise_randomness(self, shape=(2, ), suffix: str = '') -> List[ABY3Tensor]:
        """
        Initial setup for pairwise randomness: Every two parties hold a shared key.
        p1: (r0, r1)  p2: (r1, r2)  p3: (r2, r0)
        """
        r_local = ABY3Tensor.random(shape)
        # await self.send(self.share_tag, r_local, dsts=[self.helper])
        # r_remote = await self.get(self.share_tag, suffix, srcs=self.peer)
        # return [r_local, r_remote]
        return await self.reshare(r_local, suffix=suffix)

    async def generate_b2a_key(self, shape=(2, ), suffix: str = '',
                               two_key_party: str = Role.assistant) -> List[ABY3Tensor]:
        """
        Initial setup for generating shares during the conversion
        from boolean sharing to arithmetic sharing
        """
        r_local = ABY3Tensor.random(shape)
        await self.send(self.share_tag, r_local, dsts=[self.peer, self.helper])
        if self.role != two_key_party:
            r_remote = await self.get(self.share_tag, suffix, srcs=[self.peer, self.helper])
            if self.role == Role.active:
                return [r_local, r_remote[0], r_remote[1]]
            elif self.role == Role.passive:
                return [r_remote[1], r_local, r_remote[0]]
            else:
                return [r_remote[0], r_remote[1], r_local]
        else:
            # if two_key_party
            r_remote = await self.get(self.share_tag, suffix, srcs=self.peer)
            if self.role == Role.active:
                return [r_local, r_remote, None]
            else:
                return [r_remote, None, r_local]

    def _gen_b2a_sharing(self, b2a_keys: List[ABY3Tensor]) -> Tuple[ABY3Tensor, List]:
        """
        TODO: use pre-generated random seed to gen b2a sharing r
        """
        shares = [None, None]
        x = None
        if type(None) not in [type(i) for i in b2a_keys]:
            x = b2a_keys[0] ^ b2a_keys[1] ^ b2a_keys[2]
        if self.role == Role.active:
            # TODO: see b2a_keys to be random seed, and generate randomness here
            shares[0], shares[1] = b2a_keys[0], b2a_keys[1]
        elif self.role == Role.passive:
            shares[0], shares[1] = b2a_keys[1], b2a_keys[2]
        else:
            shares[0], shares[1] = b2a_keys[2], b2a_keys[0]
        return x, shares

    async def _gen_zero_sharing(self, shape, share_type=ARITHMETIC) -> ABY3Tensor:
        def helper(x: ABY3Tensor, y: ABY3Tensor):
            if share_type == ARITHMETIC:
                return x - y
            elif share_type == BOOLEAN:
                return x ^ y
            else:
                raise NotImplementedError(
                    "Only arithmetic and boolean sharings are supported."
                )

        f0, f1 = await self.generate_pairwise_randomness(shape)
        return helper(f0, f1)

    async def _B_not_private(self, x: List[ABY3Tensor]):

        z = [None, None]
        if self.role == Role.active:
            z[0] = ~ x[0]
            z[1] = x[1]
        elif self.role == Role.passive:
            z[0] = x[0]
            z[1] = x[1]
        else:
            z[0] = x[0]
            z[1] = ~ x[1]

        return z

    async def _A2B_private(self, x: List[ABY3Tensor], nbits=None) -> List[ABY3Tensor]:
        x_shares = x
        x_shape = x[0].shape
        zero = ABY3Tensor(np.zeros(x_shape, dtype=np.int64))
        a = await self._gen_zero_sharing(x_shape, share_type=BOOLEAN)

        operand1 = [None, None]
        operand2 = [None, None]

        # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
        # We need to reshare it into two operands that will be fed into an addition circuit:
        # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
        # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2
        if self.role == Role.active:
            x0_plus_x1 = x_shares[0] + x_shares[1]
            operand1[0] = x0_plus_x1 ^ a
            await self.send(self.share_tag, operand1[0], dsts=[self.helper], suffix="operand1")
            operand1[1] = await self.get(self.share_tag, suffix="a1", srcs=[self.peer])

            operand2[0] = zero
            operand2[1] = zero

        elif self.role == Role.passive:
            operand1[0] = a
            await self.send(self.share_tag, operand1[0], dsts=[self.helper], suffix="a1")
            operand1[1] = await self.get(self.share_tag, suffix="a2", srcs=[self.peer])

            operand2[0] = zero  # 0
            operand2[1] = x_shares[1]  # x2

        else:
            operand1[0] = a  # a2
            await self.send(self.share_tag, operand1[0], dsts=[self.helper], suffix="a2")
            operand1[1] = await self.get(self.share_tag, suffix="operand1", srcs=[self.peer])  # (x0+x1) XOR a0

            operand2[0] = x_shares[0]
            operand2[1] = zero

        # Step 2: Parallel prefix adder that requires log(k) rounds of communication
        result = await self.B_ppa_private_private(operand1, operand2, nbits)

        return result

    async def bit_extract_private(self, x: List[ABY3Tensor], i, share_type=ARITHMETIC) -> List[ABY3Tensor]:
        """
        Bit extraction: Extracts the `i`-th bit of an arithmetic sharing or boolean sharing
        to a single-bit boolean sharing.
        """
        x_shape = x[0].shape

        if share_type == ARITHMETIC:
            x_a2b = await self._A2B_private(x, i + 1)

        # Take out the i-th bit
        #
        # NOTE: Don't use x = x & 0x1. Even though we support automatic lifting of 0x1
        # to an ABY3Tensor, but it also includes automatic scaling to make the two operands have
        # the same scale, which is not what want here.
        #
        mask = ABY3Tensor((np.ones(x_shape, dtype=np.int64) * [0x1 << i]).astype(int).astype(object))
        x_mask = B_and_private_public(x_a2b, mask)

        result = [None, None]
        result[0] = ABY3Tensor(x_mask[0].fxp.astype(np.bool))
        result[1] = ABY3Tensor(x_mask[1].fxp.astype(np.bool))

        return result

    async def _B2A_private(self, x: List[ABY3Tensor],
                           b2a_keys_1: List[ABY3Tensor],
                           b2a_keys_2: List[ABY3Tensor],
                           nbits=None) -> List[ABY3Tensor]:
        """
        Bit composition: Convert a boolean sharing to an arithmetic sharing.
        """
        # In semi-honest, the following two calls can be further optimized because we don't
        # need the boolean shares of x1 and x2. We only need their original values on intended server
        x1, x1_shares = self._gen_b2a_sharing(b2a_keys_1)  # generate b2a_key with when role is assistant
        x2, x2_shares = self._gen_b2a_sharing(b2a_keys_2)  # generate b2a_key with when role is active

        a = await self._gen_zero_sharing(x[0].shape, share_type=BOOLEAN)

        neg_x1_neg_x2 = [None, None]
        # [(a0, (-x1 - x2)^a1), (-x1 - x2)^a1, a2), (a2, a0)]
        if self.role == Role.passive:
            # P1 reshares (-x1-x2) as private input
            value = -x1 - x2
            neg_x1_neg_x2[0] = value ^ a
            await self.send(self.share_tag, neg_x1_neg_x2[0], dsts=[self.helper], suffix="neg")
            neg_x1_neg_x2[1] = await self.get(self.share_tag, suffix="a2", srcs=self.peer)
        elif self.role == Role.active:
            # P0
            await self.send(self.share_tag, a, dsts=[self.helper], suffix="a0")
            neg_x1_neg_x2[0] = a
            neg_x1_neg_x2[1] = await self.get(self.share_tag, suffix="neg", srcs=self.peer)
        else:
            # P2
            await self.send(self.share_tag, a, dsts=[self.helper], suffix="a2")
            neg_x1_neg_x2[0] = a
            neg_x1_neg_x2[1] = await self.get(self.share_tag, suffix="a0", srcs=self.peer)

        # neg_x1_neg_x2 = ABY3Tensor(np.array(neg_x1_neg_x2))
        # compute x0 = x + (-x1 - x2) using the parellel prefix adder
        x0 = await self.B_ppa_private_private(x, neg_x1_neg_x2, nbits)

        # Reveal x0 to server 0 and 2 , aby3 figure 3 preprocess 4: reveal r1', r1 to party 1 and party3
        # Construct the arithmetic sharing , aby3 figure 3 preprocess 5
        if self.role == Role.active:
            x0_on_0 = await self.reconstruct(x0, share_type=BOOLEAN, mode='global')
            x1_on_0 = x1
            return [x0_on_0, x1_on_0]
        elif self.role == Role.passive:
            await self.reconstruct(x0, share_type=BOOLEAN, mode='remote')
            x1_on_1 = x1
            x2_on_1 = x2
            return [x1_on_1, x2_on_1]
        else:
            x2_on_2 = x2
            x0_on_2 = await self.reconstruct(x0, share_type=BOOLEAN, mode='global')
            return [x2_on_2, x0_on_2]

    async def _B_and_private_private(self, x: List[ABY3Tensor], y: List[ABY3Tensor]) -> List[ABY3Tensor]:
        a = await self._gen_zero_sharing(x[0].shape, share_type=BOOLEAN)

        tmp0 = x[0] & y[0]
        tmp1 = x[0] & y[1]
        tmp2 = x[1] & y[0]
        z_local = tmp0 ^ tmp1 ^ tmp2 ^ a

        z = await self.reshare(z_local)
        return z

    def _add_public_private(self, public_share: ABY3Tensor, private_share: List[ABY3Tensor]) -> List[ABY3Tensor]:
        z = [None, None]
        if self.role == Role.active:
            z[0] = private_share[0] + public_share
            z[1] = private_share[1]
        elif self.role == Role.passive:
            z[0] = private_share[0]
            z[1] = private_share[1]
        else:
            z[0] = private_share[0]
            z[1] = private_share[1] + public_share
        return z

    async def mul_private_private(self, x_shares: List[ABY3Tensor], y_shares: List[ABY3Tensor]) -> List[ABY3Tensor]:

        shape = x_shares[0].shape
        a = await self._gen_zero_sharing(shape)
        z = (
                x_shares[0] * y_shares[0]
                + x_shares[0] * y_shares[1]
                + x_shares[1] * y_shares[0]
                + a
        )

        local_share = await self.reshare(z)
        return await self._truncate_private_noninteractive(local_share)

    async def mul_public_private(self, public_share: ABY3Tensor, y_shares: List[ABY3Tensor], is_scaled=True) -> List[ABY3Tensor]:

        z = [None, None]
        z[0] = y_shares[0] * public_share
        z[1] = y_shares[1] * public_share

        # local_share = await self.reshare(z)
        if is_scaled:
            return await self._truncate_private_noninteractive(z)
        else:
            return z

    async def matmul_private_private(self, x_shares: List[ABY3Tensor], y_shares: List[ABY3Tensor]) -> List[ABY3Tensor]:

        shape = (*x_shares[0].shape[:-1], y_shares[0].shape[-1])
        a = await self._gen_zero_sharing(shape)
        z = (
                x_shares[0] @ y_shares[0]
                + x_shares[0] @ y_shares[1]
                + x_shares[1] @ y_shares[0]
                + a
        )

        local_share = await self.reshare(z)
        return await self._truncate_private_noninteractive(local_share)

    async def ot(self, sender: str, receiver: str, helper: str, m0: ABY3Tensor = None, m1: ABY3Tensor = None,
                 c_on_receiver: ABY3Tensor = None, c_on_helper: ABY3Tensor = None,
                 key_on_sender: List[ABY3Tensor] = List[None], key_on_helper: List[ABY3Tensor] = List[None]):

        if self.role == sender:
            masked_m0 = m0 ^ ABY3Tensor(key_on_sender.fxp[0])
            masked_m1 = m1 ^ ABY3Tensor(key_on_sender.fxp[1])
            await self.send(self.share_tag, [masked_m0, masked_m1], dsts=[receiver], suffix="mask")

        elif self.role == receiver:
            masked_m0, masked_m1 = await self.get(self.share_tag, suffix="mask", srcs=[sender])
            masked_m_c = ABY3Tensor(np.where(c_on_receiver.fxp, masked_m1.fxp, masked_m0.fxp))
            w_c = await self.get(self.share_tag, suffix="w_c", srcs=[helper])
            m_c = masked_m_c ^ w_c
            return m_c

        else:
            w_c = ABY3Tensor(np.where(c_on_helper.fxp, key_on_helper.fxp[1], key_on_helper.fxp[0]))
            await self.send(self.share_tag, w_c, dsts=[receiver], suffix="w_c")

    async def B_ppa_private_private(self, x: List[ABY3Tensor], y: List[ABY3Tensor],
                                    n_bits=None, topology="kogge_stone") -> List[ABY3Tensor]:
        """
      Parallel prefix adder (PPA). This adder can be used for addition of boolean sharings.

      `n_bits` can be passed as an optimization to constrain the computation for least significant
      `n_bits` bits.

      AND Depth: log(k)
      Total gates: klog(k)
      """

        if topology == "kogge_stone":
            return await self.B_ppa_kogge_stone_private_private(x, y, n_bits)
        elif topology == "sklansky":
            return await self.B_ppa_sklansky_private_private(x, y, n_bits)
        else:
            raise NotImplementedError("Unknown adder topology.")

    async def B_ppa_sklansky_private_private(self, x: List[ABY3Tensor], y: List[ABY3Tensor],
                                             n_bits) -> List[ABY3Tensor]:
        """
      Parallel prefix adder (PPA), using the Sklansky adder topology.
      """

        keep_masks = [
            0x5555555555555555,
            0x3333333333333333,
            0x0F0F0F0F0F0F0F0F,
            0x00FF00FF00FF00FF,
            0x0000FFFF0000FFFF,
            0x00000000FFFFFFFF,
        ]  # yapf: disable
        copy_masks = [
            0x5555555555555555,
            0x2222222222222222,
            0x0808080808080808,
            0x0080008000800080,
            0x0000800000008000,
            0x0000000080000000,
        ]  # yapf: disable

        # G = x & y
        G = await self._B_and_private_private(x, y)
        # P = x ^ y
        P = B_xor_private_private(x, y)

        k = aby3.FiniteField.bitlength
        if n_bits is not None:
            k = n_bits
        for i in range(ceil(log2(k))):
            c_mask = [ABY3Tensor(np.ones(x[0].shape, dtype=np.object) * copy_masks[i])] * 2
            k_mask = [ABY3Tensor(np.ones(x[0].shape, dtype=np.object) * keep_masks[i])] * 2
            # Copy the selected bit to 2^i positions:
            # For example, when i=2, the 4-th bit is copied to the (5, 6, 7, 8)-th bits
            #  G1 = (G & c_mask) << 1
            G1 = lshift_private(await self._B_and_private_private(G, c_mask), 1)
            # P1 = (P & c_mask) << 1
            P1 = lshift_private(await self._B_and_private_private(P, c_mask), 1)
            for j in range(i):
                # G1 = (G1 << (2 ** j)) ^ G1
                G1 = B_xor_private_private(lshift_private(G1, 2 ** j), G1)
                # P1 = (P1 << (2 ** j)) ^ P1
                P1 = B_xor_private_private(lshift_private(P1, 2 ** j), P1)
                """
              Two-round impl. using algo. that assume using OR gate is free, but in fact,
              here using OR gate cost one round.
              The PPA operator 'o' is defined as:
              (G, P) o (G1, P1) = (G + P*G1, P*P1), where '+' is OR, '*' is AND
              """
            # G1 and P1 are 0 for those positions that we do not copy the selected bit to.
            # Hence for those positions, the result is: (G, P) = (G, P) o (0, 0) = (G, 0).
            # In order to keep (G, P) for these positions so that they can be used in the future,
            # we need to let (G1, P1) = (G, P) for these positions, because (G, P) o (G, P) = (G, P)
            #
            # G1 = G1 ^ (G & k_mask)
            # P1 = P1 ^ (P & k_mask)
            #
            # G = G | (P & G1)
            # P = P & P1
                """
              One-round impl. by modifying the PPA operator 'o' as:
              (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
              This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
              the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
              at the same time.
              And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
              This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
              if we want to keep G and P bits.
              """
            # Option 1: Using (G, P) o (0, P) = (G, P)
            # P1 = P1 ^ (P & k_mask)
            # Option 2: Using (G, P) o (0, 1) = (G, P)
            # P1 = P1 ^ k_mask
            P1 = B_xor_private_private(P1, k_mask)

            # G = G ^ (P & G1)
            G = B_xor_private_private(G, await self._B_and_private_private(P, G1))
            # P = P & P1
            P = await self._B_and_private_private(P, P1)

        # G stores the carry-in to the next position
        # C = G << 1
        C = lshift_private(G, 1)
        # P = x ^ y
        P = B_xor_private_private(x, y)
        # z = C ^ P
        z = B_xor_private_private(C, P)

        return z

    async def B_ppa_kogge_stone_private_private(self, x: List[ABY3Tensor],
                                                y: List[ABY3Tensor], n_bits) -> List[ABY3Tensor]:
        """
      Parallel prefix adder (PPA), using the Kogge-Stone adder topology.
      """

        # assert isinstance(x, ABY3Tensor), type(x)
        # assert isinstance(y, ABY3Tensor), type(y)

        k = aby3.FiniteField.bitlength  # 64
        keep_masks = []
        for i in range(ceil(log2(k))):
            keep_masks.append((1 << (2 ** i)) - 1)
            """
        For example, if prot.nbits = 64, then keep_masks is:
        keep_masks = [0x0000000000000001, 0x0000000000000003, 0x000000000000000f,
                      0x00000000000000ff, 0x000000000000ffff, 0x00000000ffffffff]
        """
        # G = x & y
        G = await self._B_and_private_private(x, y)
        # P = x ^ y
        P = B_xor_private_private(x, y)
        k = aby3.FiniteField.bitlength if n_bits is None else n_bits  #64
        for i in range(ceil(log2(k))):
            k_mask = [ABY3Tensor(np.ones(x[0].shape, dtype=np.int64) * keep_masks[i])] * 2
            # G1 = G << (2 ** i)
            G1 = lshift_private(G, (2 ** i))
            # P1 = P << (2 ** i)
            P1 = lshift_private(P, (2 ** i))
            """
      One-round impl. by modifying the PPA operator 'o' as:
      (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
      This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
      the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
      at the same time.
      And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
      This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
      if we want to keep G and P bits.
      """
            # Option 1: Using (G, P) o (0, P) = (G, P)
            # P1 = P1 ^ (P & k_mask)
            # Option 2: Using (G, P) o (0, 1) = (G, P)
            # P1 = P1 ^ k_mask
            P1 = B_xor_private_private(P1, k_mask)
            # G = G ^ (P & G1)
            G = B_xor_private_private(G, await self._B_and_private_private(P, G1))
            # P = P & P1
            P = await self._B_and_private_private(P, P1)

        # G stores the carry-in to the next position
        # C = G << 1
        C = lshift_private(G, 1)
        # P = x ^ y
        P = B_xor_private_private(x, y)
        # z = C ^ P
        z = B_xor_private_private(C, P)
        return z

    async def _truncate_private(self, x: List[ABY3Tensor]) -> List[ABY3Tensor]:

        if aby3.FiniteField.use_noninteractive_truncation:
            return await self._truncate_private_noninteractive(x)

        return await self._truncate_private_interactive(x)

    # async def _truncate_private_interactive(self, local_share: List[ABY3Tensor],
    #                                            suffix: str = '') -> List[ABY3Tensor]:
    #     """
    #   See protocol TruncPr (3.1) in
    #     "Secure Computation With Fixed-Point Numbers" by Octavian Catrina and Amitabh
    #     Saxena, FC'10.
    #
    #   We call it "interactive" to keep consistent with the 2pc setting,
    #   but in fact, our protocol uses only one round communication, exactly the same as
    #   that in the "non-interactive" one.
    #   """

    async def _truncate_private_noninteractive(self, local_share: List[ABY3Tensor],
                                               suffix: str = '') -> List[ABY3Tensor]:

        """Performs the ABY3 truncation algorithm1.

        Args:
            local_share (List[ABY3Tensor],): Tensors to truncate

        Returns:
            List[ABY3Tensor], : Truncated shares.
        """

        base = aby3.FiniteField.base
        amount = aby3.FiniteField.precision_fractional

        # ((x1/2^d, (x2 + x3)/2^d -r), ((x2 + x3)/2^d - r, r), (r, x1/2^d)))
        # Step1: compute new shares
        if self.role == Role.active:
            # p1 truncate self
            x1_trunc = local_share[0].truncate(amount, base)
            # send x1_trunc to p3
            await self.send(self.share_tag, x1_trunc, dsts=[self.helper], suffix="x1'")
            # get x_trunc from p2
            x_trunc = await self.get(self.share_tag, suffix="x'", srcs=[self.peer])
            return [x1_trunc, x_trunc]  # p1 (x1', (x2'+x3')/2^d -r)
        elif self.role == Role.passive:
            # p2 get r from p3
            r = await self.get(self.share_tag, suffix="r3", srcs=[self.peer])
            # compute (x2'+x3')/2^d - r3
            x_trunc = (local_share[0] + local_share[1]).truncate(amount, base) - r
            # send x_trunc to p1
            await self.send(self.share_tag, x_trunc, dsts=[self.helper], suffix="x'")
            return [x_trunc, r]  # p2 ((x2' + x3' - r)/2^d, r))
        elif self.role == Role.assistant:
            # p3 generate r,and sends to p2
            r = local_share[0].random_alike()
            await self.send(self.share_tag, r, dsts=[self.helper], suffix="r3")
            # get x1_trunc from p1
            x1_trunc = await self.get(self.share_tag, suffix="x1'", srcs=[self.peer])
            return [r, x1_trunc]  # p3 (r, x1')

    async def mul_trunc2_private_private(self, x: List[ABY3Tensor], y: List[ABY3Tensor],
                                         b2a_keys_1: List[ABY3Tensor], b2a_keys_2: List[ABY3Tensor], suffix="") -> List[ABY3Tensor]:
        """
        Multiplication with the Trunc2 protocol in the ABY3 paper.
        This is more efficient (in terms of communication rounds)
        than `mul` in the onlline phase only when pre-computation
        is left out of consideration.
        """

        x_shares = x
        y_shares = y
        shape = x_shares[0].shape  # (2, 2)
        amount = aby3.finite_field.default_precision_fractional

        # Step1: Generate a random truncation pair
        r = await self.generate_pairwise_randomness(shape)  # aby3 figure3 preprocess 1: all parties locally compute [[r']]B
        r_trunc = rshift_private(r, amount)
        r = await self._B2A_private(r, b2a_keys_1, b2a_keys_2)
        r_trunc = await self._B2A_private(r_trunc,  b2a_keys_1, b2a_keys_2)

        # step2: compute 3-out-of-3 sharing of (x * y - r)
        a = await self._gen_zero_sharing(shape)
        z = (
                x_shares[0] * y_shares[0]
                + x_shares[0] * y_shares[1]
                + x_shares[1] * y_shares[0]
                + a
                - r[0]
        )
        await self.send(self.share_tag, z, dsts=[self.peer, self.helper])
        # step3: Reveal ( x * y - r) / 2^d
        z_remote = await self.get(self.share_tag, suffix, srcs=[self.peer, self.helper])
        xy_minus_r_trunc = z + z_remote[0] + z_remote[1]
        xy_minus_r_trunc = xy_minus_r_trunc >> amount


        # Step4: Final addition [[r]]A + (x' - r')/2d
        return self._add_public_private(xy_minus_r_trunc, r_trunc)

    async def pow_private(self, x: List[ABY3Tensor], p: int) -> List[ABY3Tensor]:
        x_shape = x[0].shape
        # result = [ABY3Tensor(np.ones(x_shape, dtype=np.int64)).encode(True)] * 2
        result = ABY3Tensor(np.ones(x_shape, dtype=np.int64)).encode(True)
        tmp = x
        while p > 0:
            bit = p & 0x1
            if bit > 0:
                if isinstance(result, ABY3Tensor):
                    result = await self.mul_public_private(result, tmp)
                else:
                    result = await self.mul_private_private(result, tmp)
            p >>= 1
            if p > 0:
                tmp = await self.mul_private_private(tmp, tmp)
        return result

    async def polynomial_private(self, x: List[ABY3Tensor], coeffs: List):

        x_shape = x[0].shape
        result = ABY3Tensor(np.zeros(x_shape, dtype=np.int64)).encode(True)
        for i in range(len(coeffs)):
            if i == 0:
                tmp = ABY3Tensor(np.array(coeffs[i])).encode(True)
                result = result + tmp
            elif coeffs[i] == 0:
                continue
            elif coeffs[i] - int(coeffs[i]) == 0:
                # Optimization when coefficient is integer: multiplication can be performed
                # locally without interactive truncation
                # tmp = tmp * (x ** i)
                tmp = ABY3Tensor(np.array(coeffs[i]).astype(int).astype(object))
                x_pow = await self.pow_private(x, i)
                tmp = await self.mul_public_private(tmp, x_pow, is_scaled=False)
                if isinstance(result, ABY3Tensor):
                    result = self._add_public_private(result, tmp)
                else:
                    result = add_private_private(result, tmp)
            else:
                tmp = ABY3Tensor(np.array(coeffs[i])).encode(True)
                x_pow = await self.pow_private(x, i)
                tmp = await self.mul_public_private(tmp, x_pow)
                if isinstance(result, ABY3Tensor):
                    result = self._add_public_private(result, tmp)
                else:
                    result = add_private_private(result, tmp)

        return result


