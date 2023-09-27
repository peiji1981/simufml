

from typing import List, Tuple, Union, Iterator
from numpy import array
from random import shuffle

from simufml.common.comm.communication import Communicator
from simufml.utils.util import Role

class _NoneArray:
    '''
    This class is for consistancy of DataSet's behavier.    
    '''
    def __getitem__(self, i: Union[int, List[int]]) -> None:
        return None


class DataSet:
    def __init__(self, X: array=_NoneArray(), Y: array=_NoneArray()):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, i: Union[int, List[int]]) -> Union[array, Tuple[array]]:
        '''
        Will return (array, array) if self.Y has values; and (array, None) if not.
        '''
        return (self.X[i], self.Y[i])


    def __iter__(self):
        return self._generator()
        

    def _generator(self):
        for i in range(len(self)):
            yield self[i]


class DataLoader(Communicator):
    def __init__(
        self,
        dataset: DataSet=None,
        batch_size: int=None,
        role=None,
        shuffle: bool=True
    ):
        super().__init__(
            role=role,
            variables=[
                ('batch_num', Role.active, Role.broadcast),
                ('batch_ids', Role.active, Role.passive)
            ]
        )

        self.dataset = DataSet() if dataset is None else dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __len__(self) -> int:
        if self.role==Role.active:
            return (len(self.dataset) - 1) // self.batch_size + 1
        else:
            return self.batch_num


    def __iter__(self) -> Iterator:
        return self.batch_generator()


    def batch_generator(self) -> Iterator:
        for ids in self._batch_ids_generator():
            yield self.dataset[ids]


    def _batch_ids_generator(self) -> List:
        if self.role==Role.active:
            idss = list(range(len(self.dataset)))
            if self.shuffle:
                shuffle(idss)
            for i in range(len(self)):
                yield idss[i*self.batch_size : (i + 1)*self.batch_size]
        elif self.role==Role.passive:
            for batch_ids in self.batch_idss:
                yield batch_ids
        else:
            for _ in range(len(self)):
                yield 0


    async def sync_batches(self):
        if self.role==Role.active:
            await self.send('batch_num', len(self))
            for ids in self._batch_ids_generator():
                await self.send('batch_ids', ids)
        else:
            self.batch_num = await self.get('batch_num')

            if self.role==Role.passive:
                batch_idss = []
                for _ in range(len(self)):
                    batch_ids = await self.get('batch_ids')
                    batch_idss.append(batch_ids)
                self.batch_idss = batch_idss


    def no_shuffle(self):
        class NoShuffle:
            def __enter__(_self):
                _self.bk = self.shuffle
                self.shuffle = False

            def __exit__(_self, *args):
                self.shuffle = _self.bk

        return NoShuffle()
    