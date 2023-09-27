from simufml.common.comm.federation import Session
from simufml.common.comm.federation import fedrun
from simufml.secureprotol.aby3.aby3_tensor import ABY3Tensor
from simufml.secureprotol.aby3.aby3_util import B_xor_private_private, rshift_private, logical_rshift_private
from simufml.utils.util import Role
from simufml.secureprotol.aby3 import ABY3Util

from simufml.tests.util import err_ratio

import numpy as np

roles = [Role.active, Role.passive, Role.assistant]
aby3_ACT = ABY3Util(Role.active, Role.passive, Role.assistant)
aby3_PAS = ABY3Util(Role.passive, Role.assistant, Role.active)
aby3_ASS = ABY3Util(Role.assistant, Role.active, Role.passive)


def test_share():
    fft = ABY3Tensor.random((2, 3))
    share_type = 0
    with Session(roles):
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, fft, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

    assert (ABY3Tensor(sh0[0].fxp + sh1[0].fxp + sh2[0].fxp) == fft).all()
    assert (ABY3Tensor(sh0[1].fxp + sh1[1].fxp + sh2[1].fxp) == fft).all()


def test_reconstruct():
    secret = ABY3Tensor.random((2, 3))
    share_type = 0
    with Session(roles):
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, secret, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # reconstruct on each side
        res0, res1, res2 = fedrun((aby3_ACT.reconstruct, sh0, share_type, 'global'),
                                  (aby3_PAS.reconstruct, sh1, share_type, 'global'),
                                  (aby3_ASS.reconstruct, sh2, share_type, 'global'))

        assert (res0 == secret).all()
        assert (res1 == secret).all()
        assert (res2 == secret).all()

        # reconstruct on only ACT
        res0, res1, res2 = fedrun((aby3_ACT.reconstruct, sh0, share_type, 'local'),
                                  (aby3_PAS.reconstruct, sh1, share_type, 'remote'),
                                  (aby3_ASS.reconstruct, sh2, share_type, 'remote'))

        assert (res0 == secret).all()
        assert res1 is None
        assert res2 is None


def test_generate_pairwise_randomness() -> None:
    with Session(roles):
        key0, key1, key2 = fedrun((aby3_ACT.generate_pairwise_randomness, ''),
                                  (aby3_PAS.generate_pairwise_randomness, ''),
                                  (aby3_ASS.generate_pairwise_randomness, ''))

        assert (key0[0] == key2[1]).all()
        assert (key0[1] == key1[0]).all()
        assert (key1[1] == key2[0]).all()


def test_generate_b2a_key() -> None:
    with Session(roles):
        # assstant(P3) has two key, other role has three
        key0, key1, key2 = fedrun((aby3_ACT.generate_b2a_key, (2, ), '', Role.assistant),
                                  (aby3_PAS.generate_b2a_key, (2, ), '', Role.assistant),
                                  (aby3_ASS.generate_b2a_key, (2, ), '', Role.assistant))

        assert (key0[0] == key1[0]).all()
        assert (key0[1] == key1[1]).all()
        assert (key0[2] == key1[2]).all()
        assert (key2[0] == key0[0]).all()
        assert (key2[2] == key0[2]).all()
        assert (key2[1] is None)

        # active(P1) has two key, other role has three
        key3, key4, key5 = fedrun((aby3_ACT.generate_b2a_key, (2, ), '', Role.active),
                                  (aby3_PAS.generate_b2a_key, (2, ), '', Role.active),
                                  (aby3_ASS.generate_b2a_key, (2, ), '', Role.active))

        assert (key4[0] == key5[0]).all()
        assert (key4[1] == key5[1]).all()
        assert (key4[2] == key5[2]).all()
        assert (key3[0] == key4[0]).all()
        assert (key3[1] == key4[1]).all()
        assert (key3[2] is None)


def test_gen_zero_sharing() -> None:
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    share_type = 1
    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT._gen_zero_sharing, x.shape, share_type),
                               (aby3_PAS._gen_zero_sharing, x.shape, share_type),
                               (aby3_ASS._gen_zero_sharing, x.shape, share_type))
        print("end")


def test_3d_matmul_private() -> None:
    x = ABY3Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    y = ABY3Tensor(np.array([[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]))

    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))
        # share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # mul_trunc1_private_private
        z0, z1, z2 = fedrun((aby3_ACT.matmul_private_private, sh0, sh3),
                            (aby3_PAS.matmul_private_private, sh1, sh4),
                            (aby3_ASS.matmul_private_private, sh2, sh5))

        # reconstruct x_shares * y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 0, 'global'),
                                  (aby3_PAS.reconstruct, z1, 0, 'global'),
                                  (aby3_ASS.reconstruct, z2, 0, 'global'))

        assert (err_ratio(res3.decode(), np.array([[[94, 100], [229, 244]], [[508, 532], [697, 730]]]) < 0.001).all())



def test_boolean_share() -> None:
    # define plaintext x and y
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    y = ABY3Tensor(np.array([[7, 8, 9], [10, 11, 12]]))

    share_type = 1

    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))
        # share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # calculate binary xor between x_share and y_share
        z1_0 = B_xor_private_private(sh0, sh3)
        z1_1 = B_xor_private_private(sh1, sh4)
        z1_2 = B_xor_private_private(sh2, sh5)

        # reconstruct x_shares ^ y_shares result
        res0, res1, res2 = fedrun((aby3_ACT.reconstruct, z1_0, share_type, 'global'),
                                  (aby3_PAS.reconstruct, z1_1, share_type, 'global'),
                                  (aby3_ASS.reconstruct, z1_2, share_type, 'global'))

        assert (res0 == ABY3Tensor(np.array([[6, 10, 10], [14, 14, 10]]))).all()

        # calculate binary logical &(and) between x_share and y_share
        z2_0, z2_1, z2_2 = fedrun((aby3_ACT._B_and_private_private, sh0, sh3),
                                  (aby3_PAS._B_and_private_private, sh1, sh4),
                                  (aby3_ASS._B_and_private_private, sh2, sh5))

        # reconstruct x_shares ^ y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z2_0, share_type, 'global'),
                                  (aby3_PAS.reconstruct, z2_1, share_type, 'global'),
                                  (aby3_ASS.reconstruct, z2_2, share_type, 'global'))

        assert (res3 == ABY3Tensor(np.array([[1, 0, 1], [0, 1, 4]]))).all()
        assert (res4 == ABY3Tensor(np.array([[1, 0, 1], [0, 1, 4]]))).all()
        assert (res5 == ABY3Tensor(np.array([[1, 0, 1], [0, 1, 4]]))).all()


def test_not_private() -> None:

    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]])).encode(False)
    y = ABY3Tensor(np.array([[1, 0, 0], [0, 1, 0]])).encode(False)

    with Session(roles):
        # boolean share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, 1),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        n0, n1, n2 = fedrun((aby3_ACT._B_not_private, sh0),
                            (aby3_PAS._B_not_private, sh1),
                            (aby3_ASS._B_not_private, sh2))

        # reconstruct x_shares result
        res1, res2, res3 = fedrun((aby3_ACT.reconstruct, n0, 1, 'global'),
                                  (aby3_PAS.reconstruct, n1, 1, 'global'),
                                  (aby3_ASS.reconstruct, n2, 1, 'global'))

        assert (res1.fxp == np.array([[-2, -3, -4], [-5, -6, -7]])).all()

        # boolean share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, 1, np.bool),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        n3, n4, n5 = fedrun((aby3_ACT._B_not_private, sh3),
                            (aby3_PAS._B_not_private, sh4),
                            (aby3_ASS._B_not_private, sh5))

        # reconstruct y_shares result
        res4, res5, res6 = fedrun((aby3_ACT.reconstruct, n3, 1, 'global'),
                                  (aby3_PAS.reconstruct, n4, 1, 'global'),
                                  (aby3_ASS.reconstruct, n5, 1, 'global'))

        assert (res4.fxp == np.array([[0, 1, 1], [1, 0, 1]])).all()


def test_rshift_private() -> None:
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    y = ABY3Tensor(np.array([[-1, -2, -3], [-4, 5, 6]]))
    share_type = 1

    with Session(roles):
        # boolean share
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # x >> 1
        sh0_rshift, sh1_rshift, sh2_rshift = rshift_private(sh0, 1),\
                                             rshift_private(sh1, 1),\
                                             rshift_private(sh2, 1)

        # reconstruct x_shares result
        res1, res2, res3 = fedrun((aby3_ACT.reconstruct, sh0_rshift, share_type, 'global'),
                                  (aby3_PAS.reconstruct, sh1_rshift, share_type, 'global'),
                                  (aby3_ASS.reconstruct, sh2_rshift, share_type, 'global'))

        assert (res1.fxp == np.array([[0, 1, 1], [2, 2, 3]])).all()

        # share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # y >> 1
        sh3_rshift, sh4_rshift, sh5_rshift = rshift_private(sh3, 1), \
                                             rshift_private(sh4, 1), \
                                             rshift_private(sh5, 1)


        # reconstruct y_arith_shares result
        res4, res5, res6 = fedrun((aby3_ACT.reconstruct, sh3_rshift, share_type, 'global'),
                                  (aby3_PAS.reconstruct, sh4_rshift, share_type, 'global'),
                                  (aby3_ASS.reconstruct, sh5_rshift, share_type, 'global'))

        assert (res4.fxp == np.array([[-1, -1, -2], [-2, 2, 3]])).all()

        # logical_rshift
        sh3_logical_rshift, sh4_logical_rshift, sh5_logical_rshift = logical_rshift_private(sh3, 1), \
                                                                     logical_rshift_private(sh4, 1), \
                                                                     logical_rshift_private(sh5, 1)

        # # reconstruct y_logical_rshift_shares result
        res7, res8, res9 = fedrun((aby3_ACT.reconstruct, sh3_logical_rshift, share_type, 'global'),
                                  (aby3_PAS.reconstruct, sh4_logical_rshift, share_type, 'global'),
                                  (aby3_ASS.reconstruct, sh5_logical_rshift, share_type, 'global'))

        nbits = 64
        truth = np.array(
            [
                [
                    (-1 & ((1 << nbits) - 1)) >> 1,
                    (-2 & ((1 << nbits) - 1)) >> 1,
                    (-3 & ((1 << nbits) - 1)) >> 1
                ],
                [
                    (-4 & ((1 << nbits) - 1)) >> 1, 2, 3
                ]
            ]
        )
        assert (res7.fxp == truth).all()


def test_B_ppa_private_private() -> None:
    # define plaintext x and y
    apply_scaling = True
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    y = ABY3Tensor(np.array([[7, 8, 9], [10, 11, 12]]))

    share_type = 1

    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))
        # share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # kogge_stone
        z0, z1, z2 = fedrun((aby3_ACT.B_ppa_private_private, sh0, sh3, None, "kogge_stone"),
                            (aby3_PAS.B_ppa_private_private, sh1, sh4, None, "kogge_stone"),
                            (aby3_ASS.B_ppa_private_private, sh2, sh5, None, "kogge_stone"))

        # reconstruct x_shares ^ y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, share_type, 'global'),
                                  (aby3_PAS.reconstruct, z1, share_type, 'global'),
                                  (aby3_ASS.reconstruct, z2, share_type, 'global'))

        assert (res3 == ABY3Tensor(np.array([[8, 10, 12], [14, 16, 18]]))).all()
        assert (res4 == ABY3Tensor(np.array([[8, 10, 12], [14, 16, 18]]))).all()
        assert (res5 == ABY3Tensor(np.array([[8, 10, 12], [14, 16, 18]]))).all()

        # sklansky
        z3, z4, z5 = fedrun((aby3_ACT.B_ppa_private_private, sh0, sh3, None, "sklansky"),
                            (aby3_PAS.B_ppa_private_private, sh1, sh4, None, "sklansky"),
                            (aby3_ASS.B_ppa_private_private, sh2, sh5, None, "sklansky"))

        # reconstruct x_shares ^ y_shares result
        res6, res7, res8 = fedrun((aby3_ACT.reconstruct, z3, share_type, 'global'),
                                  (aby3_PAS.reconstruct, z4, share_type, 'global'),
                                  (aby3_ASS.reconstruct, z5, share_type, 'global'))

        assert (res6 == ABY3Tensor(np.array([[8, 10, 12], [14, 16, 18]]))).all()
        assert (res7 == ABY3Tensor(np.array([[8, 10, 12], [14, 16, 18]]))).all()
        assert (res8 == ABY3Tensor(np.array([[8, 10, 12], [14, 16, 18]]))).all()


def test_gen_b2a_sharing() -> None:

    with Session(roles):
        # assstant(P3) has two key, other role has three
        key0, key1, key2 = fedrun((aby3_ACT.generate_b2a_key, (2, 3), '', Role.assistant),
                                  (aby3_PAS.generate_b2a_key, (2, 3), '', Role.assistant),
                                  (aby3_ASS.generate_b2a_key, (2, 3), '', Role.assistant))

        # active(P1) has two key, other role has three
        key3, key4, key5 = fedrun((aby3_ACT.generate_b2a_key, (2, 3), '', Role.active),
                                  (aby3_PAS.generate_b2a_key, (2, 3), '', Role.active),
                                  (aby3_ASS.generate_b2a_key, (2, 3), '', Role.active))

    print("end")


def test_A2B_private() -> None:
    # define plaintext x and y
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]])).encode(True)

    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # A2B
        z0, z1, z2 = fedrun((aby3_ACT._A2B_private, sh0, None),
                            (aby3_PAS._A2B_private, sh1, None),
                            (aby3_ASS._A2B_private, sh2, None))

        # reconstruct x_shares ^ y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 1, 'global'),
                                  (aby3_PAS.reconstruct, z1, 1, 'global'),
                                  (aby3_ASS.reconstruct, z2, 1, 'global'))

        assert (res3.decode() == np.array([[1., 2., 3.], [4., 5., 6.]])).all()
        assert (res4.decode() == np.array([[1., 2., 3.], [4., 5., 6.]])).all()
        assert (res5.decode() == np.array([[1., 2., 3.], [4., 5., 6.]])).all()


def test_bit_extract_private() -> None:

    x = ABY3Tensor(np.array([[1, -2, 3], [-4, -5, 6]])).encode(True)

    y = ABY3Tensor(np.array([[1, -2, 3], [-4, -5, 6]]))

    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        i = 63
        z0, z1, z2 = fedrun((aby3_ACT.bit_extract_private, sh0, i),
                            (aby3_PAS.bit_extract_private, sh1, i),
                            (aby3_ASS.bit_extract_private, sh2, i))


        # reconstruct
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 1, 'global'),
                                  (aby3_PAS.reconstruct, z1, 1, 'global'),
                                  (aby3_ASS.reconstruct, z2, 1, 'global'))

        assert (res3.fxp == np.array([[0, 1, 0], [1, 1, 0]])).all()

        # share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        i = 1
        z3, z4, z5 = fedrun((aby3_ACT.bit_extract_private, sh3, i),
                            (aby3_PAS.bit_extract_private, sh4, i),
                            (aby3_ASS.bit_extract_private, sh5, i))

        # reconstruct  result
        res6, res7, res8 = fedrun((aby3_ACT.reconstruct, z3, 1, 'global'),
                                  (aby3_PAS.reconstruct, z4, 1, 'global'),
                                  (aby3_ASS.reconstruct, z5, 1, 'global'))

        assert (res6.fxp == np.array([[0, 1, 1], [0, 1, 1]])).all()


def test_B2A_private() -> None:
    # define plaintext x and y
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]]))

    share_type = 1
    shape = (2, 3)

    with Session(roles):

        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # assstant(P3) has two key, other role has three
        key0, key1, key2 = fedrun((aby3_ACT.generate_b2a_key, shape, '', Role.assistant),
                                  (aby3_PAS.generate_b2a_key, shape, '', Role.assistant),
                                  (aby3_ASS.generate_b2a_key, shape, '', Role.assistant))

        # active(P1) has two key, other role has three
        key3, key4, key5 = fedrun((aby3_ACT.generate_b2a_key, shape, '', Role.active),
                                  (aby3_PAS.generate_b2a_key, shape, '', Role.active),
                                  (aby3_ASS.generate_b2a_key, shape, '', Role.active))

        # B2A
        z0, z1, z2 = fedrun((aby3_ACT._B2A_private, sh0, key0, key3, None),
                            (aby3_PAS._B2A_private, sh1, key1, key4, None),
                            (aby3_ASS._B2A_private, sh2, key2, key5, None))

        # reconstruct x_shares ^ y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 0, 'global'),
                                  (aby3_PAS.reconstruct, z1, 0, 'global'),
                                  (aby3_ASS.reconstruct, z2, 0, 'global'))

        assert (res3 == ABY3Tensor(np.array([[1., 2., 3.], [4., 5., 6.]]))).all()
        assert (res4 == ABY3Tensor(np.array([[1., 2., 3.], [4., 5., 6.]]))).all()
        assert (res5 == ABY3Tensor(np.array([[1., 2., 3.], [4., 5., 6.]]))).all()


def test_ot() -> None:

    m0 = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]])).encode(False)
    m1 = ABY3Tensor(np.array([[2, 3, 4], [5, 6, 7]])).encode(False)

    c_on_receiver = ABY3Tensor(np.array([[True, False, True], [False, True, False]])).encode(False)
    c_on_helper = ABY3Tensor(np.array([[True, False, True], [False, True, False]])).encode(False)

    shape = [2, 2, 3]

    with Session(roles):
        pair1, pair2, pair3 = fedrun((aby3_ACT.generate_pairwise_randomness, shape),
                                     (aby3_PAS.generate_pairwise_randomness, shape),
                                     (aby3_ASS.generate_pairwise_randomness, shape))

        key_on_sender = pair2[0]
        key_on_helper = pair1[1]
        # sender: str, receiver: str, helper: str, m0: ABY3Tensor, m1: ABY3Tensor,
        # c_on_receiver: ABY3Tensor, c_on_helper: ABY3Tensor,
        # key_on_sender: List[ABY3Tensor], key_on_helper: List[ABY3Tensor]
        ot1, ot2, ot3 = fedrun((aby3_ACT.ot, Role.passive, Role.assistant, Role.active,
                                None, None, None, c_on_helper, None, key_on_helper),
                               (aby3_PAS.ot, Role.passive, Role.assistant, Role.active,
                                m0, m1, None, None, key_on_sender, None),
                               (aby3_ASS.ot, Role.passive, Role.assistant, Role.active,
                                None, None, c_on_receiver, None, None, None))

        assert (err_ratio(ot3.decode(False), np.array([[2., 2., 4.], [4., 6., 6.]])) < 0.001).all()


def test_mul_public_private() -> None:
    apply_scaling = True
    x = ABY3Tensor(np.array([[1.333, 1.333], [1.333, 1.333]])).encode(apply_scaling)
    y = ABY3Tensor(np.array([[2., 2.], [2., 2.]])).encode(apply_scaling)

    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # mul_public_private
        z0, z1, z2 = fedrun((aby3_ACT.mul_public_private, y, sh0),
                            (aby3_PAS.mul_public_private, y, sh1),
                            (aby3_ASS.mul_public_private, y, sh2))

        # reconstruct x_shares * y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 0, 'global'),
                                  (aby3_PAS.reconstruct, z1, 0, 'global'),
                                  (aby3_ASS.reconstruct, z2, 0, 'global'))

        assert (err_ratio(res3.decode(), np.array([[2.666, 2.666, ], [2.666, 2.666]])) < 0.001).all()


def test_mul_trunc1_private_private() -> None:
    # define plaintext x and y
    apply_scaling = True
    x = ABY3Tensor(np.array([[1.333, 1.333], [1.333, 1.333]])).encode(apply_scaling)
    y = ABY3Tensor(np.array([[2., 2.], [2., 2.]])).encode(apply_scaling)

    share_type = 0
    shape = x.shape
    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # mul_trunc1_private_private
        z0, z1, z2 = fedrun((aby3_ACT.mul_private_private, sh0, sh3),
                            (aby3_PAS.mul_private_private, sh1, sh4),
                            (aby3_ASS.mul_private_private, sh2, sh5))

        # reconstruct x_shares * y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 0, 'global'),
                                  (aby3_PAS.reconstruct, z1, 0, 'global'),
                                  (aby3_ASS.reconstruct, z2, 0, 'global'))

        assert (err_ratio(res3.decode(), np.array([[2.666, 2.666, ], [2.666, 2.666]])) < 0.001).all()


def test_mul_trunc2_private_private() -> None:
    # define plaintext x and y
    apply_scaling = True
    x = ABY3Tensor(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])).encode(apply_scaling)
    y = ABY3Tensor(np.array([[2., 2., 2.], [2., 2., 2.]])).encode(apply_scaling)
    share_type = 0
    shape = x.shape
    with Session(roles):
        # share x
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # share y
        sh3, sh4, sh5 = fedrun((aby3_ACT.share, y, share_type),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # assstant(P3) has two key, other role has three
        key0, key1, key2 = fedrun((aby3_ACT.generate_b2a_key, shape, '', Role.assistant),
                                  (aby3_PAS.generate_b2a_key, shape, '', Role.assistant),
                                  (aby3_ASS.generate_b2a_key, shape, '', Role.assistant))

        # active(P1) has two key, other role has three
        key3, key4, key5 = fedrun((aby3_ACT.generate_b2a_key, shape, '', Role.active),
                                  (aby3_PAS.generate_b2a_key, shape, '', Role.active),
                                  (aby3_ASS.generate_b2a_key, shape, '', Role.active))

        # mul_trunc2_private_private
        z0, z1, z2 = fedrun((aby3_ACT.mul_trunc2_private_private, sh0, sh3, key0, key3),
                            (aby3_PAS.mul_trunc2_private_private, sh1, sh4, key1, key4),
                            (aby3_ASS.mul_trunc2_private_private, sh2, sh5, key2, key5))

        # reconstruct x_shares * y_shares result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 0, 'global'),
                                  (aby3_PAS.reconstruct, z1, 0, 'global'),
                                  (aby3_ASS.reconstruct, z2, 0, 'global'))

        assert (err_ratio(res3.decode(), np.array([[2.2, 4.4, 6.6], [8.8, 11.0, 13.2]])) < 0.001).all()
        assert (err_ratio(res4.decode(), np.array([[2.2, 4.4, 6.6], [8.8, 11.0, 13.2]])) < 0.001).all()
        assert (err_ratio(res5.decode(), np.array([[2.2, 4.4, 6.6], [8.8, 11.0, 13.2]])) < 0.001).all()


def test_pow_private() -> None:
    # define plaintext x and y
    apply_scaling = True
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]])).encode(apply_scaling)

    with Session(roles):
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # mul_pow_private
        z0, z1, z2 = fedrun((aby3_ACT.pow_private, sh0, 2),
                            (aby3_PAS.pow_private, sh1, 2),
                            (aby3_ASS.pow_private, sh2, 2))

        # reconstruct x^2 result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 0, 'global'),
                                  (aby3_PAS.reconstruct, z1, 0, 'global'),
                                  (aby3_ASS.reconstruct, z2, 0, 'global'))

        assert (err_ratio(res3.decode(), np.array([[1, 4, 9], [16, 25, 36]])) < 0.001).all()

        z3, z4, z5 = fedrun((aby3_ACT.pow_private, sh0, 3),
                            (aby3_PAS.pow_private, sh1, 3),
                            (aby3_ASS.pow_private, sh2, 3))

        # reconstruct x^3 result
        res6, res7, res8 = fedrun((aby3_ACT.reconstruct, z3, 0, 'global'),
                                  (aby3_PAS.reconstruct, z4, 0, 'global'),
                                  (aby3_ASS.reconstruct, z4, 0, 'global'))

        assert (err_ratio(res7.decode(), np.array([[1, 8, 27], [64, 125, 216]])) < 0.001).all()


def test_polynomial_private() -> None:

    # define plaintext x and y
    apply_scaling = True
    x = ABY3Tensor(np.array([[1, 2, 3], [4, 5, 6]])).encode(apply_scaling)

    with Session(roles):
        sh0, sh1, sh2 = fedrun((aby3_ACT.share, x, 0),
                               (aby3_PAS.get_share, Role.active),
                               (aby3_ASS.get_share, Role.active))

        # y = 1 + 1.2 * x + 3 * (x ** 2) + 0.5 * (x ** 3)
        polynomial_coeffs = [1, 1.2, 3.0, 0.5]
        # mul_pow_private
        z0, z1, z2 = fedrun((aby3_ACT.polynomial_private, sh0, polynomial_coeffs),
                            (aby3_PAS.polynomial_private, sh1, polynomial_coeffs),
                            (aby3_ASS.polynomial_private, sh2, polynomial_coeffs))

        # reconstruct result
        res3, res4, res5 = fedrun((aby3_ACT.reconstruct, z0, 0, 'global'),
                                  (aby3_PAS.reconstruct, z1, 0, 'global'),
                                  (aby3_ASS.reconstruct, z2, 0, 'global'))

        assert (err_ratio(res3.decode(), np.array([[5.7, 19.4, 45.1], [85.8, 144.5, 224.2]])) < 0.001).all()



