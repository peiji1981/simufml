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
import os

from simufml.secureprotol.symmetric_encryption.py_aes_core import AESModeOfOperationOFB
from simufml.secureprotol.symmetric_encryption.symmetric_encryption import SymmetricKey
from federatedml.util import conversion


class AESKey(SymmetricKey):
    """
    Note that a key cannot used for both encryption and decryption scenarios
    """
    def __init__(self, key, nonce=None):
        """

        :param key: bytes, must be 16, 24 or 32 bytes long
        :param nonce: bytes, must be 16 bytes long
        """
        super(AESKey, self).__init__()
        if nonce is None:
            self.nonce = os.urandom(16)
            self.key = key
            self.cipher_core = AESModeOfOperationOFB(key=self.key, iv=self.nonce)
        else:
            self.nonce = nonce
            self.key = key
            self.cipher_core = AESModeOfOperationOFB(key=self.key, iv=self.nonce)

    def _renew(self):
        """
        Self renew cipher_core after encryption and decryption
        :return:
        """
        self.cipher_core = AESModeOfOperationOFB(key=self.key, iv=self.nonce)


class AESEncryptKey(AESKey):
    """
    AES encryption scheme
    Note that the ciphertext size is affected only by that of the plaintext, instead of the key length
    """
    def __init__(self, key):
        super(AESEncryptKey, self).__init__(key=key)

    def encrypt(self, plaintext):
        if type(plaintext) is not bytes:
            plaintext = self._all_to_bytes(plaintext)
        elif type(plaintext) is bytes:
            pass
        else:
            raise TypeError("AES encryptor supports bytes/int/float/str")
        ciphertext = self.cipher_core.encrypt(plaintext)
        self._renew()
        return ciphertext

    @staticmethod
    def _all_to_bytes(message):
        """
        Convert an int/float/str to bytes, e.g., 1.65 -> b'1.65', 'hello -> b'hello'
        :param message: int/float/str
        :return: -1 if type error, otherwise str
        """
        if type(message) == int or type(message) == float:
            return conversion.str_to_bytes(str(message))
        elif type(message) == str:
            return conversion.str_to_bytes(message)
        else:
            return -1

    def get_nonce(self):
        return self.nonce


class AESDecryptKey(AESKey):
    """
    AES decryption scheme
    """
    def __init__(self, key, nonce):
        super(AESDecryptKey, self).__init__(key=key, nonce=nonce)

    def decrypt(self, ciphertext):
        """

        :param ciphertext: bytes
        :return: str
        """
        if type(ciphertext) is not bytes:
            raise TypeError("AES decryptor supports bytes only")
        plaintext = conversion.bytes_to_str(self.cipher_core.decrypt(ciphertext))
        self._renew()
        return plaintext
