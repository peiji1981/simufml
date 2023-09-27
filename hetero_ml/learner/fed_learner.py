from typing import Dict, Callable

from simufml.hetero_ml.learner.party_learner import PartyLearner
from simufml.common.comm.federation import Session, fedrun
from simufml.common.comm.communication import Mailman

class FedLearner:
    def __init__(self, parties: PartyLearner):
        self.parties = parties


    def _fedrun(self, func_name, *args):
        func_args = [(getattr(p, func_name), *args) for p in self.parties]
        return fedrun(*func_args)


    def sync_ciphers(self):
        self._fedrun('sync_ciphers')


    def fit(
        self,
        epochs, 
        metrics: Dict[str, Callable]={}, 
        shuffle: bool=None, 
        skip_cipher: bool=None, 
        re_initialize_models: bool=False
    ):
        with Session([p.role for p in self.parties]):
            self.sync_ciphers()
            if re_initialize_models:
                for p in self.parties:
                    p.model.initialize_param()
            
            with self.set_shuffle(shuffle), self.set_cipher(skip_cipher):
                return self._fedrun('fit', epochs, metrics)


    def predict(self, skip_cipher: bool=None):
        with Session([p.role for p in self.parties]):
            self.sync_ciphers()
            with self.set_cipher(skip_cipher):
                return self._fedrun('predict')

    
    def set_shuffle(self, shuffle):
        class SetShuffle:
            def __enter__(_self):
                _self.bk = {}
                for p in self.parties:
                    _self.bk[p] = p.train_dataloader.shuffle
                    p.train_dataloader.shuffle = shuffle if shuffle is not None else _self.bk[p]

            def __exit__(_self, *args):
                for p in self.parties:
                    p.train_dataloader.shuffle = _self.bk[p]

        return SetShuffle()


    def set_cipher(self, skip_cipher):
        class SetCipher:
            def __enter__(_self):
                _self.bk = {}
                for p in self.parties:
                    _self.bk[p] = p.model.skip_cipher
                    p.model.skip_cipher = skip_cipher if skip_cipher is not None else _self.bk[p]

            def __exit__(_self, *args):
                for p in self.parties:
                    p.model.skip_cipher = _self.bk[p]

        return SetCipher()