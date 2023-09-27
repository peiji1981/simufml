from simufml.utils.util import Role
from simufml.common.hetero_sshe_model import HeteroSSHEModel


class HeteroSSHELRBase(HeteroSSHEModel):
    roles = [Role.active, Role.passive]
    variables = [
        ('en_z_a', Role.passive, Role.active),
        ('en_y_a', Role.active, Role.passive),
        ('en_z_b', Role.active, Role.passive),
        ('en_z_b_squ', Role.active, Role.passive),
        ('en_L_a', Role.passive, Role.active)
    ]

    @property
    def peer(self):
        return Role.active if self.role==Role.passive else Role.passive


    @property
    def cipher(self):
        return self.dummy_cipher if self.skip_cipher else self.ciphers[self.role]


    @property
    def peer_cipher(self):
        return self.dummy_cipher if self.skip_cipher else self.ciphers[self.peer]