from typing import Tuple
import numpy as np

""" We use a bit_length of 32 based on the following considerations:
- The largest int type in numpy is 'int64'. For values out of the range of 'int64',
the dtype will become 'object'.
- We have two options:
- A: If we choose to use numpy's builtin dtypes, then there are:
    - pros: 
        (1) the computation would be fast
        (2) we can use all numpy's method, like np.round, np.mod, ...
    - cons:
        (1) we can't use numbers out of the range of 'int64', otherwise there will 
        be problem
- B: If we choose to use 'object' type, then there are:
    - pros:
        (1) we can use number as large as we need.
    - cons:
        (1) the computation would be slow
        (2) some of numpy's method will not work
- What we choose is option A, which is to use numpy's builin int64. Because it will bring
more convenience. While the cost is we can't use very large integers.
- In order to avoid the result of multiplication being out of range, we choose bit_length
be 32. """

default_bitlength = 64
default_precision_integral = 10
default_scaling_base = 2
default_precision_fractional = 13
default_scale = default_scaling_base ** default_precision_fractional


class _FiniteField:
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self, bitlength: int = None, scale: int = None):
        self.bitlength = default_bitlength if bitlength is None else bitlength
        self.scale = default_scale if scale is None else scale
        self.base = default_scaling_base
        self.precision_fractional = default_precision_fractional
        self.precision_integral = default_precision_integral

    @property
    def ringsize(self):
        return 1 << self.bitlength

    @property
    def half(self):
        return 1 << (self.bitlength - 1)

    # @property
    # def bounds(self):
    #     # [low, high)
    #     return - 1 << (self.bitlength - 1), (1 << (self.bitlength - 1)) - 1

    @property
    def bounds(self):
        # [low, high)
        return - 1 << (self.bitlength - 1), (1 << (self.bitlength - 1))

    def set_field(self, bitlength, scale):
        class SetField:
            def __enter__(_self):
                _self.bitlength = self.bitlength
                _self.scale = self.scale

                self.bitlength = bitlength
                self.scale = scale

            def __exit__(_self, *args):
                self.bitlength = _self.bitlength
                self.scale = _self.scale

        return SetField()


FiniteField = _FiniteField()
