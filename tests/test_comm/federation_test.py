from simufml.common.comm.communication import Mailman
from simufml.common.comm.federation import fedrun
from simufml.utils.util import Role
from simufml.common.comm.federation import Session


def test_fedrun():
    async def foo(role, peer, v):
        mailman = Mailman.get_instance(role)
        await mailman.send('x', role, peer, v)
        return await mailman.get('x', peer)

    roles = [Role.active, Role.passive]
    with Session(roles):
        pkt0, pkt1 = fedrun((foo, roles[0], roles[1], 7), (foo, roles[1], roles[0], 8))
    assert (pkt0.obj, pkt1.obj) == (8,7)