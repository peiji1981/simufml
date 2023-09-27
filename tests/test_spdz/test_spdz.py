import numpy as np

from simufml.common.comm.federation import Session
from simufml.common.comm.federation import fedrun
from simufml.utils.util import Role
from simufml.secureprotol.spdz import SpdzUtil
from simufml.secureprotol.spdz import FFTensor
from simufml.secureprotol.spdz import finite_field
from simufml.tests.util import err_ratio

roles = [Role.active, Role.passive]
spdz_ACT = SpdzUtil(Role.active, [Role.passive])
spdz_PAS = SpdzUtil(Role.passive, [Role.active])

# n0: the bit length of ring; n1: the bit length of scale
n0 = finite_field.default_bitlength
n1 = finite_field.default_scale.bit_length() - 1

def test_share():
    fft = FFTensor.random((10,10))
    with Session(roles):
        sh0, sh1 = fedrun((spdz_ACT.share, fft), (spdz_PAS.get_share,))
    
    assert (FFTensor(sh0.fxp+sh1.fxp)==fft).all()


def test_reconstruct():
    secret = FFTensor.random((10,10))
    with Session(roles):
        sh0, sh1 = fedrun((spdz_ACT.share, secret), (spdz_PAS.get_share,))

        # reconstruct on both side
        res0, res1 = fedrun((spdz_ACT.reconstruct, sh0, 'global'), (spdz_PAS.reconstruct, sh1, 'global'))
        assert (res0==secret).all()
        assert (res1==secret).all()
        
        # reconstruct on only ACT
        res0, res1 = fedrun((spdz_ACT.reconstruct, sh0, 'local'), (spdz_PAS.reconstruct, sh1, 'remote'))
        assert (res0==secret).all()
        assert res1 is None


def test_add_sub():
    # (1) ACT holds x, PAS holds y
    # choose a proper range for real numbers, to void the result being out of range.
    real_range = (-(1<<(n0-n1-2)), 1<<(n0-n1-2))
    x = np.random.uniform(*real_range, size=(10,10))
    y = np.random.uniform(*real_range, size=(10,10))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)

    with Session(roles):
        # (2) Every party shares it's secret, so every gets two shares
        X0, X1 = fedrun((spdz_ACT.share, X), (spdz_PAS.get_share,))
        Y0, Y1 = fedrun((spdz_ACT.get_share,), (spdz_PAS.share, Y))

        # (3.1) Secret + Secret: Every party does Xi+Yi, then they reconstruct the result.
        Z, _ = fedrun((spdz_ACT.reconstruct, X0+Y0, 'local'), (spdz_PAS.reconstruct, X1+Y1, 'remote'))
        assert (err_ratio(Z.decode(), x+y)<0.001).all()
        
        # (3.2) Secret - Secret: Every party does Xi-Yi, then they reconstruct the result.
        Z, _ = fedrun((spdz_ACT.reconstruct, X0-Y0, 'local'), (spdz_PAS.reconstruct, X1-Y1, 'remote'))
        assert (err_ratio(Z.decode(), x-y)<0.001).all()
        

def test_secret_op_public():
    # (1) ACT holds x; And y is public.
    # choose a proper range for real numbers, to void the result being out of range.
    real_range = (-(1<<((n0-n1-1)//2-3)), 1<<((n0-n1-1)//2-3))
    x = np.random.uniform(*real_range, size=(10,10))
    y = np.random.uniform(*real_range, size=(10,10))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    
    with Session(roles):
        # (2) ACT shares it's secret X
        X0, X1 = fedrun((spdz_ACT.share, X), (spdz_PAS.get_share,))

        # (3.1) Share + Public: ACT do X0 + Y, then they reconstruct the result.
        Z, _ = fedrun((spdz_ACT.reconstruct, X0 + Y, 'local'), (spdz_PAS.reconstruct, X1, 'remote'))
        assert (err_ratio(Z.decode(), x+y) < 0.01).all()

        # (3.2) Share - Public: ACT do X0 - Y, then they construct the result.
        Z, _ = fedrun((spdz_ACT.reconstruct, X0-Y, 'local'), (spdz_PAS.reconstruct, X1, 'remote'))
        assert (err_ratio(Z.decode(), x-y) < 0.001).all()

        # (3.3) Share * public: Every party do Xi*Y, then they reconstruct the result.
        Z, _ = fedrun((spdz_ACT.reconstruct, X0*Y, 'local'), (spdz_PAS.reconstruct, X1*Y, 'remote'))
        assert (err_ratio(Z.decode(), x*y) < 0.001).all()

        # (3.4) Share @ Public: Every party do Xi@pub, then they reconstruct the result.
        Z, _ = fedrun((spdz_ACT.reconstruct, X0@Y, 'local'), (spdz_PAS.reconstruct, X1@Y, 'remote'))
        assert (err_ratio(Z.decode(), x@y)<0.001).all()

        # (3.5) TODO: share / pub


def test_private_mul():
    # (1) ACT holds x; PAS holds y
    # choose a proper range for real numbers, to void the result being out of range.
    real_range = (-(1<<((n0-n1-1)//2)), 1<<((n0-n1-1)//2))
    x = np.random.uniform(*real_range, size=(10,10))
    y = np.random.uniform(*real_range, size=(10,10))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)

    with Session(roles):
        # (2) ACT and PAS call private_mul coordinately to get shares of Z=X*Y and share it.
        Z0, Z1 = fedrun((spdz_ACT.private_mul, X, None, True), (spdz_PAS.private_mul, None, Y, False))

        # (3) ACT and PAS reconstruct result
        Z, _ = fedrun((spdz_ACT.reconstruct, Z0, 'local'), (spdz_PAS.reconstruct, Z1, 'remote'))
        assert (err_ratio(Z.decode(), x*y)<0.001).all()


def test_private_matmul():
    # (1) ACT holds x; PAS holds y
    # choose a proper range for real numbers, to void the result being out of range.
    real_range = (-(1<<((n0-n1-1)//2-3)), 1<<((n0-n1-1)//2-3))
    x = np.random.uniform(*real_range, size=(10,10))
    y = np.random.uniform(*real_range, size=(10,10))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)

    with Session(roles):
        # (2) ACT and PAS call private_mul coordinately to get shares of Z=X@Y and share it.
        Z0, Z1 = fedrun((spdz_ACT.private_matmul, X, None, True), (spdz_PAS.private_matmul, None, Y, False))

        # (3) ACT and PAS reconstruct result
        Z, _ = fedrun((spdz_ACT.reconstruct, Z0, 'local'), (spdz_PAS.reconstruct, Z1, 'remote'))
        assert (err_ratio(Z.decode(), x@y)<0.002).all()


def test_beaver_triplets_for_mul():
    a_shape = b_shape = (5, 5)
    dot_func = lambda x,y : x*y
    with Session(roles):
        # (1) ACT and PAS generate triplets.
        (A0, B0, C0), (A1, B1, C1) = fedrun(
            (spdz_ACT.beaver_triplets, a_shape, b_shape, dot_func), 
            (spdz_PAS.beaver_triplets, a_shape, b_shape, dot_func)
        )

        # (2) Let's check if they satisfy (A0+A1)*(B0+B1)==(C0+C1) mod ringsize
        A, _ = fedrun((spdz_ACT.reconstruct, A0, 'local'), (spdz_PAS.reconstruct, A1, 'remote'))
        B, _ = fedrun((spdz_ACT.reconstruct, B0, 'local'), (spdz_PAS.reconstruct, B1, 'remote'))
        C, _ = fedrun((spdz_ACT.reconstruct, C0, 'local'), (spdz_PAS.reconstruct, C1, 'remote'))
        assert (err_ratio((A*B).fxp, C.fxp)<0.001).all()


def test_beaver_triplets_for_matmul():
    a_shape = b_shape = (5, 5)
    dot_func = lambda x,y : x@y
    with Session(roles):
        # (1) ACT and PAS generate triplets.
        (A0, B0, C0), (A1, B1, C1) = fedrun(
            (spdz_ACT.beaver_triplets, a_shape, b_shape, dot_func), 
            (spdz_PAS.beaver_triplets, a_shape, b_shape, dot_func)
        )

        # (2) Let's check if they satisfy (A0+A1)@(B0+B1)==(C0+C1) mod ringsize
        A, _ = fedrun((spdz_ACT.reconstruct, A0, 'local'), (spdz_PAS.reconstruct, A1, 'remote'))
        B, _ = fedrun((spdz_ACT.reconstruct, B0, 'local'), (spdz_PAS.reconstruct, B1, 'remote'))
        C, _ = fedrun((spdz_ACT.reconstruct, C0, 'local'), (spdz_PAS.reconstruct, C1, 'remote'))
        assert (err_ratio((A@B).fxp, C.fxp)<0.001).all()


def test_share_mul_share():
    # choose a proper range for real numbers, to void the result being out of range.
    real_range = (-(1<<((n0-n1-1)//2)), 1<<((n0-n1-1)//2))
    x = np.random.uniform(*real_range, size=(5, 5))
    y = np.random.uniform(*real_range, size=(5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)

    with Session(roles):
        # (1) ACT holds (X0, Y0); PAS holds (X1, Y1). (Don't care how they get these shares.)
        X0, X1 = fedrun((spdz_ACT.share, X), (spdz_PAS.get_share,))
        Y0, Y1 = fedrun((spdz_ACT.share, Y), (spdz_PAS.get_share,))

        # (2) Compute [[Z]] = [[X]] * [[Y]]
        Z0, Z1 = fedrun(
            (spdz_ACT.share_mul_share, X0, Y0, True),
            (spdz_PAS.share_mul_share, X1, Y1, False)
        )   

        # (3) check if X*Y==Z
        Z, _ = fedrun((spdz_ACT.reconstruct, Z0, 'local'), (spdz_PAS.reconstruct, Z1, 'remote'))
        assert (err_ratio((X*Y).fxp, Z.fxp)<0.001).all()


def test_share_matmul_share():
    # choose a proper range for real numbers, to void the result being out of range.
    real_range = (-(1<<((n0-n1-1)//2)), 1<<((n0-n1-1)//2))
    x = np.random.uniform(*real_range, size=(5, 5))
    y = np.random.uniform(*real_range, size=(5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)

    with Session(roles):
        # (1) ACT holds (X0, Y0); PAS holds (X1, Y1). (Don't care how they get these shares.)
        X0, X1 = fedrun((spdz_ACT.share, X), (spdz_PAS.get_share,))
        Y0, Y1 = fedrun((spdz_ACT.share, Y), (spdz_PAS.get_share,))

        # (2) Compute [[Z]] = [[X]] @ [[Y]]
        Z0, Z1 = fedrun(
            (spdz_ACT.share_matmul_share, X0, Y0, True),
            (spdz_PAS.share_matmul_share, X1, Y1, False)
        )   

        # (3) check if X@Y==Z
        Z, _ = fedrun((spdz_ACT.reconstruct, Z0, 'local'), (spdz_PAS.reconstruct, Z1, 'remote'))
        assert (err_ratio((X@Y).fxp, Z.fxp)<0.001).all()
