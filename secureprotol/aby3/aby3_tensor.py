#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import numpy as np

from simufml.secureprotol import aby3
from simufml.secureprotol.aby3 import native
from simufml.secureprotol.fate_paillier import PaillierEncryptedNumber


class ABY3Tensor:
    """
    Finite Field Tensor. It's elements are all fixed point numbers in a finite field.
    """

    def __init__(
            self,
            fxp: np.ndarray
    ):
        self.fxp = fxp

    @property
    def fxp(self):
        return self._fxp

    @fxp.setter
    def fxp(self, fxp):
        self._fxp = fxp if ABY3Tensor.encrypted(fxp) else self.wraparound(fxp)

    @staticmethod
    def encrypted(fxp):
        return isinstance(np.array(fxp).reshape(-1)[0], PaillierEncryptedNumber)

    def in_range(self, fxp):
        bounds = aby3.FiniteField.bounds
        fxp = fxp.fxp
        return (bounds[0] <= fxp).all() and (fxp < bounds[1]).all()

    # def wraparound(self, fxp):
    #     fxp = np.round(fxp).astype(np.int64)
    #     fxp = fxp % spdz.FiniteField.ringsize
    #     fxp = fxp - (spdz.FiniteField.bounds[1]<=fxp)*spdz.FiniteField.ringsize
    #     return fxp

    def wraparound(self, fxp):
        return fxp

    @property
    def shape(self):
        return self.fxp.shape

    def random_alike(self, factory=np.int64):
        # np.random.seed(seed)
        # [minval, maxval)
        if factory == np.int64:
            return ABY3Tensor(np.random.randint(*aby3.FiniteField.bounds, self.shape, dtype=factory))
        elif factory == np.bool:
            # rng = np.random.default_rng()
            # return ABY3Tensor(rng.integers(0, 2, self.shape))
            return ABY3Tensor(np.random.randint(0, 2, self.shape, dtype=factory))
        else:
            raise NotImplementedError

    @classmethod
    def random(cls, shape, factory=np.int64):
        fft = cls(np.zeros(shape))
        return fft.random_alike(factory)

    def __str__(self):
        return f"ABY3Tensor: {self.fxp}"

    def __repr__(self):
        return self.__str__()

    # @classmethod
    # def encode(cls, v):
    #     fxp = np.array(v) * aby3.FiniteField.scale
    #     return cls(fxp)

    # def encode(self, apply_scaling: str = True):
    #     if apply_scaling:
    #         scaled = self.fxp * aby3.FiniteField.scale
    #         # return ABY3Tensor((self.fxp * aby3.FiniteField.scale).astype(np.int64))
    #     else:
    #         scaled = self.fxp
    #         # return ABY3Tensor(self.fxp.astype(np.int64))
    #     integers = scaled.astype(np.int64)
    #
    #     return ABY3Tensor(integers)

    def encode(self, apply_scaling: str = True):
        if apply_scaling:
            scaled = self.fxp * aby3.FiniteField.scale
        else:
            scaled = self.fxp

        integers = scaled.astype(np.int64)

        return ABY3Tensor(integers)



    def decode(self, apply_scaling: str = True):
        if apply_scaling:
            return self.fxp / aby3.FiniteField.scale
        else:
            return self.fxp

    # def where(self, condition, x, y):
    #     return ABY3Tensor(np.where(condition, x, y))

    def _get_value(self, other):
        v0 = self.decode()
        v1 = other.decode() if isinstance(other, ABY3Tensor) else other
        return v0, v1

    def _get_fxp(self, other):
        other = other if isinstance(other, ABY3Tensor) else ABY3Tensor.encode(other)
        return self.fxp, other.fxp

    def _wrap(self, v):
        return ABY3Tensor.encode(v)

    def _check_compatibility(self, other):
        if not isinstance(other, ABY3Tensor):
            raise ValueError("Only supports this operation between two ABY3Tensor objects.")

    def __eq__(self, other):
        self._check_compatibility(other)
        return self.fxp == other.fxp

    def __gt__(self, other):
        self._check_compatibility(other)
        return self.fxp > other.fxp

    def __ge__(self, other):
        self._check_compatibility(other)
        return self.fxp >= other.fxp

    def __lt__(self, other):
        self._check_compatibility(other)
        return self.fxp < other.fxp

    def __le__(self, other):
        self._check_compatibility(other)
        return self.fxp <= other.fxp

    def __add__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return ABY3Tensor(fxp0 + fxp1)

    def __xor__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return ABY3Tensor(fxp0 ^ fxp1)

    def __and__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return ABY3Tensor(fxp0 & fxp1)

    def __invert__(self):
        return ABY3Tensor(~ self.fxp)

    def __iadd__(self, other):
        self.fxp = (self + other).fxp
        return self



    # TODO: __radd__, __rsub__, __rmul__, __rtruediv__, __rmatmul__
    # It's not hard to think of an implementation of these methods, take __radd__
    # for example:
    # >>> def __radd__(self, other):
    # ...     retur self + other
    # But there is a problem with this implementation, for example:
    # >>> np.array([1,2]) + ABY3Tensor([0, 0], 8, 1)
    # np.array([ABY3Tensor: [1, 1], ABY3Tensor: [2, 2]], dtype=object)
    #
    # Until we find a way to fix this problem, we use a bypass for each __rxx__ operation:
    # - __radd__: instead of call `ndarray + fftensor`, you use `fftensor + ndarray`
    # - __rsub__: instead of call `ndarray - fftensor`, you use `(-fftensor) + ndarray`
    # - __rmul__: instead of call `ndarray * fftensor`, you use `fftensor * ndarray`
    # - __rtruediv__: instead of call `ndarray / fftensor`, you use `fftensor.reciprocal() * ndarray`
    # - __rmatmul__: instead of call `ndarray @ fftensor`, you use
    #   `(fftensor.transpose(-1, 0) @ ndarray.transpose(-1, 0)).transpose(-1, 0)`

    def __sub__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return ABY3Tensor(fxp0 - fxp1)

    def __isub__(self, other):
        self.fxp = (self - other).fxp
        return self

    def __neg__(self):
        return ABY3Tensor(-self.fxp)

    def truncate(self, amount, base=2):
        if base == 2:
            return ABY3Tensor(self.fxp >> amount)
        factor = base ** amount
        factor_inverse = native.inverse(factor, self.modulus)
        return ABY3Tensor((self.fxp - (self.fxp % factor)) * factor_inverse)

    @property
    def modulus(self) -> int:
        return 1 << aby3.FiniteField.bitlength


    # def __mul__(self, other):
    #     """
    #     Note: It's important to do '/ scale' before '% ringsize'. Because when two encoded
    #     numbers are multiplied, the result is not in field F_{ringsize} any more, but in a
    #     field F_{ringsize*scale}. So we should do `/ scale` to take it back to F_{ringsize},
    #     then we can do `% ringsize` as normal.
    #     The same for __matmul__.
    #     """
    #     fxp0, fxp1 = self._get_fxp(other)
    #     res = self.truncate(fxp0 * fxp1)
    #     return ABY3Tensor(res)
    def __mul__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return ABY3Tensor(fxp0 * fxp1)

    def __imul__(self, other):
        self.fxp = (self * other).fxp
        return self

    def __truediv__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return self._wrap(fxp0 / fxp1)

    def reciprocal(self):
        v0 = self.decode()
        return self._wrap(1 / v0)

    def __matmul__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        res = np.dot(fxp0, fxp1)
        return ABY3Tensor(res)

    def transpose(self, *axes):
        fxp = np.transpose(self.fxp, axes)
        return ABY3Tensor(fxp)

    def __pow__(self, n):
        if not (isinstance(n, int) and n > 0):
            raise ValueError('Only support power to positive integer.')
        fxp = self.fxp
        res = fxp
        for _ in range(n - 1):
            res = self.truncate(res * fxp)
        return ABY3Tensor(res)

    def __lshift__(self, n):
        if not (isinstance(n, int) and n > 0):
            raise ValueError('Only support left shift to positive integer.')
        return ABY3Tensor(self.fxp << n)

    def __rshift__(self, n):
        if not (isinstance(n, int) and n > 0):
            raise ValueError('Only support right shift to positive integer.')
        return ABY3Tensor(self.fxp >> n)

    def sum(self):
        return ABY3Tensor(self.fxp.sum())

    def mean(self):
        return ABY3Tensor(self.fxp.mean())

    def arith_rshift(self, bitlength):
        if bitlength < 0:
            raise ValueError("Unsupported shift steps.")
        if bitlength == 0:
            return self

        return ABY3Tensor(self.fxp >> bitlength)

    def logical_rshift(self, bitlength):
        if bitlength < 0:
            raise ValueError("Unsupported shift steps.")
        if bitlength == 0:
            return self

        mask = ~((-1) << (64 - bitlength))
        x = self.fxp >> bitlength
        x = x & mask
        return ABY3Tensor(x)

    def encrypt(self, cipher):
        return ABY3Tensor(cipher.encrypt(self.fxp))

    def decrypt(self, cipher):
        return ABY3Tensor(cipher.decrypt(self.fxp))
