import numpy as np

from simufml.secureprotol.spdz.ff_tensor import FFTensor
from simufml.tests.util import err_ratio
from simufml.secureprotol.spdz import finite_field
from simufml.secureprotol.spdz import FiniteField

# n0: the bit length of ring; n1: the bit length of scale
n0 = finite_field.default_bitlength
n1 = finite_field.default_scale.bit_length() - 1


def test_encode_decode():
    real_range = (-(1<<(n0-n1-1)), 1<<(n0-n1-1))
    x = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    assert (err_ratio(X.decode(), x)<0.001).all()


def test_add():
    real_range = (-(1<<(n0-n1-2)), 1<<(n0-n1-2))
    x = np.random.uniform(*real_range, (5, 5))
    y = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    # FFTensor + FFTensor
    assert (err_ratio((X+Y).decode(), x+y)<0.001).all()
    # FFTensor + ndarray
    assert (err_ratio((X+y).decode(), x+y)<0.001).all()


def test_inplace_add():
    real_range = (-(1<<(n0-n1-2)), 1<<(n0-n1-2))
    x = np.random.uniform(*real_range, (5, 5))
    y = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    # FFTensor + FFTensor
    X += Y
    x += y 
    assert (err_ratio(X.decode(), x)<0.001).all()


def test_sub():
    real_range = (-(1<<(n0-n1-2)), 1<<(n0-n1-2))
    x = np.random.uniform(*real_range, (5, 5))
    y = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    # FFTensor - FFTensor
    assert (err_ratio((X-Y).decode(), x-y)<0.001).all()
    # FFTensor - ndarray
    assert (err_ratio((X-y).decode(), x-y)<0.001).all()


def test_inplace_sub():
    real_range = (-(1<<(n0-n1-2)), 1<<(n0-n1-2))
    x = np.random.uniform(*real_range, (5, 5))
    y = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    # FFTensor + FFTensor
    X -= Y
    x -= y 
    assert (err_ratio(X.decode(), x)<0.001).all()


def test_neg():
    real_range = (-(1<<(n0-n1-1)), 1<<(n0-n1-1))
    x = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    assert (err_ratio((-X).decode(), -x)<0.001).all()


def test_mul():
    real_range = (-(1<<((n0-n1-1)//2)), 1<<((n0-n1-1)//2))
    x = np.random.uniform(*real_range, (5, 5))
    y = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    # FFTensor * FFTensor
    assert (err_ratio((X*Y).decode(), x*y)<0.001).all()
    # FFTensor * ndarray
    assert (err_ratio((X*y).decode(), x*y)<0.001).all()


def test_inplace_mul():
    real_range = (-(1<<((n0-n1-2)//2)), 1<<((n0-n1-2)//2))
    x = np.random.uniform(*real_range, (5, 5))
    y = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    X *= Y
    x *= y
    assert (err_ratio(X.decode(), x)<0.001).all()


def test_div():
    with FiniteField.set_field(8, 10):
        # FFTensor / ndarray
        assert ((FFTensor([100, -100]) / np.array([3, 7])) == FFTensor([33, -14])).all()
        # FFTensor / FFTensor
        assert ((FFTensor([100, -100]) / FFTensor([30, 70])) == FFTensor([33, -14])).all()


def test_reciprocal():
    with FiniteField.set_field(8, 10):
        fft = FFTensor([1, -12])
        assert (fft.reciprocal()==FFTensor([100, -8])).all()


def test_matmul():
    real_range = (-(1<<((n0-n1-1)//2-3)), 1<<((n0-n1-1)//2-3))
    x = np.random.uniform(*real_range, (5, 5))
    y = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    Y = FFTensor.encode(y)
    # FFTensor @ FFTensor
    assert (err_ratio((X@Y).decode(), x@y)<0.001).all()
    # FFTensor @ ndarray
    assert (err_ratio((X@y).decode(), x@y)<0.001).all()


def test_transpose():
    with FiniteField.set_field(8, 10):
        fft = FFTensor([[1,2,3],[4,5,6]])
        assert (fft.transpose(-1, 0) == FFTensor([[1,4],[2,5],[3,6]])).all()


def test_pow():
    real_range = (-(1<<((n0-n1-1)//3)), 1<<((n0-n1-1)//3))
    x = np.random.uniform(*real_range, (5, 5))
    while not (x!=0).all():
        x = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    assert (err_ratio((X**3).decode(), x**3)<0.01).all()


def test_sum():
    real_range = (-(1<<(n0-n1-9)), 1<<(n0-n1-9))
    x = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    assert (err_ratio((X.sum()).decode(), x.sum())<0.001).all()


def test_mean():
    real_range = (-(1<<(n0-n1-2)), 1<<(n0-n1-2))
    x = np.random.uniform(*real_range, (5, 5))
    X = FFTensor.encode(x)
    assert (err_ratio((X.mean()).decode(), x.mean())<0.001).all()