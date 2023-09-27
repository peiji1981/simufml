from typing import Tuple
import trio
from itertools import permutations
import math

from simufml.common.comm.communication import Mailman


def fedrun(func_arg_0: Tuple, func_arg_1: Tuple, /, *args):
    func_args = [func_arg_0, func_arg_1] + list(args)
    return trio.run(_fedrun, func_args)


async def _fedrun(func_args):
    outputs = [None for _ in func_args]
    async with trio.open_nursery() as nursery:
        for m in Mailman._instances.values():
            nursery.start_soon(m.serve)
        nursery.start_soon(_run_and_cancel, func_args, outputs, nursery.cancel_scope)
    return tuple(outputs)


async def _run_and_cancel(func_args, outputs, cancel_scop):
    async with trio.open_nursery() as nursery:
        for i, func_arg in enumerate(func_args):
            nursery.start_soon(_store_result, func_arg, outputs, i)
    cancel_scop.cancel()


async def _store_result(func_arg, outputs, i):
    outputs[i] = await func_arg[0](*func_arg[1:])


class Session:
    def __init__(self, roles):
        self.roles = roles


    def __enter__(self):
        for r0, r1 in permutations(self.roles, 2):
            m0, m1 = Mailman.create_or_get(r0), Mailman.create_or_get(r1)
            send_ch, receive_ch = trio.open_memory_channel(max_buffer_size=math.inf)
            m0.send_channels[r1] = send_ch
            m1.receive_channels[r0] = receive_ch


    def __exit__(self, *args):
        for m in Mailman._instances.values():
            for ch in m.send_channels.values():
                ch.close()
            for ch in m.receive_channels.values():
                ch.close()
            m.send_channels.clear()
            m.receive_channels.clear()
        Mailman._instances.clear()