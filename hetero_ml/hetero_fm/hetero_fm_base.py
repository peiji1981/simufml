
from simufml.utils.util import Role
from simufml.common.hetero_model import HeteroModel

class HeteroFMBase(HeteroModel):
    roles = [Role.active, Role.passive, Role.assistant]
    variables = [
        ('en_grad_z',          Role.active,    Role.assistant),
        ('en_aggvx',           Role.active,    Role.passive),
        ('en_ACT_grad_embed',  Role.active,    Role.assistant),
        ('en_yz',              Role.active,    Role.assistant),
        ('en_agg_z',           Role.active,    Role.assistant),
        ('en_PAS_forward',     Role.passive,   Role.active), # [en_host_z, en_host_vx]
        ('en_PAS_grad_embed',  Role.passive,   Role.assistant),
        ('grad_z',             Role.assistant, Role.broadcast),
        ('ACT_grad_embed',     Role.assistant, Role.active),
        ('PAS_grad_embed',     Role.assistant, Role.passive),
        ('loss',               Role.assistant, Role.active),
        ('agg_z',              Role.assistant, Role.active)
    ]

    @property
    def cipher(self):
        return self.dummy_cipher if self.skip_cipher else self.ciphers[Role.assistant]