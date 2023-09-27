
from simufml.utils.util import Role
from simufml.common.hetero_model import HeteroModel

class HeteroNNBase(HeteroModel):
    roles = [Role.active, Role.passive]
    variables = [
        ('en_wAaA_plus_eB',              Role.active,  Role.passive), # [[w_A * α_A + ε_B]]
        ('en_gzAaA_plus_eB',             Role.active,  Role.passive), # [[g_z_A * α_A + ε_B]]
        ('en_gaA',                       Role.active,  Role.passive), # [[g_α_A]],
        ('learning_rate',                Role.active,  Role.passive), # ŋ
        ('en_aA',                        Role.passive, Role.active),  # [[α_A]]
        ('wAaA_plus_eB_plus_aAeAcc',     Role.passive, Role.active),  # w_A * α_A + ε_B + ε_acc * α_A
        ('gzAaA_plus_eB_plus_eA_ov_eta', Role.passive, Role.active),  # g_z_A * α_A + ε_B + ε_A/ŋ 
        ('en_eAcc',                      Role.passive, Role.active)   # [[ε_acc]]
    ]

    @property
    def cipher(self):
        return self.dummy_cipher if self.skip_cipher else self.ciphers[Role.passive]