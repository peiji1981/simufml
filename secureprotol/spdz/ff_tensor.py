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
import torch

from simufml.secureprotol.fate_paillier import PaillierEncryptedNumber
from simufml.secureprotol import spdz


class FFTensor:
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
        self._fxp = fxp if FFTensor.encrypted(fxp) else self.wraparound(fxp)


    @staticmethod
    def encrypted(fxp):
        return isinstance(np.array(fxp).reshape(-1)[0], PaillierEncryptedNumber)


    def in_range(self, fxp):
        bounds = spdz.FiniteField.bounds
        fxp = fxp.fxp
        return (bounds[0]<=fxp).all() and (fxp<bounds[1]).all()


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


    def random_alike(self):
        # np.random.seed(seed)
        fxp = np.random.randint(*spdz.FiniteField.bounds, self.shape)
        return FFTensor(fxp)


    @classmethod
    def random(cls, shape):
        fft = cls(np.zeros(shape))
        return fft.random_alike()


    def __str__(self):
        return f"FFTensor: {self.fxp}"


    def __repr__(self):
        return self.__str__()

    
    @classmethod
    def encode(cls, v):
        fxp = np.array(v) * spdz.FiniteField.scale
        return cls(fxp)


    def decode(self):
        return self.fxp/spdz.FiniteField.scale


    def _get_value(self, other):
        v0 = self.decode()
        v1 = other.decode() if isinstance(other, FFTensor) else other
        return v0, v1


    def _get_fxp(self, other):
        other = other if isinstance(other, FFTensor) else FFTensor.encode(other)
        return self.fxp, other.fxp


    def _wrap(self, v):
        return FFTensor.encode(v)


    def _check_compatibility(self, other):
        if not isinstance(other, FFTensor):
            raise ValueError("Only supports this operation between two FFTensor objects.")
        

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
        return FFTensor(fxp0 + fxp1)

    def __xor__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return FFTensor(fxp0 ^ fxp1)

    def __and__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return FFTensor(fxp0 & fxp1)

    def __iadd__(self, other):
        self.fxp = (self + other).fxp
        return self

    # TODO: __radd__, __rsub__, __rmul__, __rtruediv__, __rmatmul__
    # It's not hard to think of an implementation of these methods, take __radd__
    # for example:
    # >>> def __radd__(self, other): 
    # ...     retur self + other
    # But there is a problem with this implementation, for example:
    # >>> np.array([1,2]) + FFTensor([0, 0], 8, 1)
    # np.array([FFTensor: [1, 1], FFTensor: [2, 2]], dtype=object)
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
        return FFTensor(fxp0-fxp1)


    def __isub__(self, other):
        self.fxp = (self - other).fxp
        return self


    def __neg__(self):
        return FFTensor(-self.fxp)


    def truncate(self, fxp):
        return fxp / spdz.FiniteField.scale


    def __mul__(self, other):
        """
        Note: It's important to do '/ scale' before '% ringsize'. Because when two encoded
        numbers are multiplied, the result is not in field F_{ringsize} any more, but in a
        field F_{ringsize*scale}. So we should do `/ scale` to take it back to F_{ringsize},
        then we can do `% ringsize` as normal.
        The same for __matmul__.
        """
        fxp0, fxp1 = self._get_fxp(other)
        res = self.truncate(fxp0 * fxp1)
        return FFTensor(res)


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
        res = self.truncate(np.dot(fxp0, fxp1))
        return FFTensor(res)


    def transpose(self, *axes):
        fxp = np.transpose(self.fxp, axes)
        return FFTensor(fxp)


    def __pow__(self, n):
        if not (isinstance(n, int) and n>0):
            raise ValueError('Only support power to positive integer.')
        fxp = self.fxp
        res = fxp
        for _ in range(n-1):
            res = self.truncate(res * fxp)
        return FFTensor(res)

    # def __rlshift__(self, n):
    #     if not (isinstance(n, int) and n>0):
    #         raise ValueError('Only support power to positive integer.')
    #     fxp = self.fxp

    def __lshift__(self, other):
        fxp0, fxp1 = self._get_fxp(other)
        return FTensor(fxp0 << fxp1)

    def sum(self):
        return FFTensor(self.fxp.sum())

    def mean(self):
        return FFTensor(self.fxp.mean())

    def arith_rshift(self, bitlength):
        if bitlength < 0:
            raise ValueError("Unsupported shift steps.")
        if bitlength == 0:
            return self

        return FFTensor([x >> bitlength for x in self.fxp])

    def logical_rshift(self, bitlength):
        if bitlength < 0:
            raise ValueError("Unsupported shift steps.")
        if bitlength == 0:
            return self

        mask = ~((-1) << (spdz.FiniteField.bound - bitlength))
        x = self.fxp >> bitlength
        x = x & mask
        return FFTensor(x)

    def encrypt(self, cipher):
        return FFTensor(cipher.encrypt(self.fxp))


    def decrypt(self, cipher):
        return FFTensor(cipher.decrypt(self.fxp))

    # def arith_rshift(self, steps):

