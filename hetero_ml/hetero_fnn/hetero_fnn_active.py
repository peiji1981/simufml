
from typing import List
from torch.nn import functional as F

from simufml.hetero_ml.hetero_nn import HeteroNN_ACT
from .utils import PartialEmbedding, MLP

class HeteroFNN_ACT(HeteroNN_ACT):
    def __init__(
        self,
        feature_size: int,
        embed_size: int,
        total_fs: int,
        mlp_arch: List[int],
        lr: float=1e-3,
        cipher_keylen: int=1024
    ):
        bot = PartialEmbedding(feature_size, embed_size, total_fs)
        top = MLP(mlp_arch)

        ACT_fs, PAS_fs = feature_size, total_fs-feature_size
        super().__init__(
            bottom_model=bot,
            top_model=top,
            mix_layer_io_shape=(ACT_fs*embed_size, PAS_fs*embed_size, mlp_arch[0]),
            loss_func=F.binary_cross_entropy_with_logits,
            lr=lr,
            mix_layer_active=F.relu,
            cipher_keylen=cipher_keylen)



