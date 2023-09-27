
from simufml.hetero_ml.hetero_nn import HeteroNN_PAS
from .utils import PartialEmbedding


class HeteroFNN_PAS(HeteroNN_PAS):
    def __init__(
        self,
        feature_size: int,
        embed_size: int,
        total_fs: int,
        cipher_keylen: int=1024
    ):
        bot = PartialEmbedding(feature_size, embed_size, total_fs)
        super().__init__(
            bottom_model=bot,
            cipher_keylen=cipher_keylen
        )


