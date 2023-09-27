#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import random
import numpy as np

BITS = 10
MIXED_RATE = 0.5


class RandomNumberGenerator(object):
    def __init__(self, dtype=np.float32):
        self.lower_bound = -2 ** BITS
        self.upper_bound = 2 ** BITS
        self.dtype = dtype

    @staticmethod
    def get_size_by_shape(shape):
        size = 1
        for dim in shape:
            size *= dim

        return size

    def generate_random_number_1d(self, size, mixed_rate=MIXED_RATE):
        return [
            random.SystemRandom().uniform(
                self.lower_bound,
                self.upper_bound
            ) 
            if np.random.rand() < mixed_rate 
            else 
            np.random.uniform(
                self.lower_bound, 
                self.upper_bound
            ) 
            for _ in range(size)
        ]

    def generate_random_number(self, shape=None, mixed_rate=MIXED_RATE):
        size = self.get_size_by_shape(shape)
        return np.reshape(self.generate_random_number_1d(size, mixed_rate=mixed_rate), shape).astype(self.dtype)
