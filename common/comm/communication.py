from typing import List, Tuple, Any, Union
from copy import deepcopy
import trio
import math

from simufml.utils.util import Role
from simufml.utils.util import list_wrap_str


class Communicator:
    def __init__(
        self, 
        role: str,
        variables: List[Tuple] = []
    ):
        self.role = role
        self._format_variables(variables)


    @property
    def mailman(self):
        return Mailman.get_instance(self.role)


    def _format_variables(self, variables):
        self.variables = {}
        for tag, srcs, dsts in variables:
            if tag in self.variables:
                raise RuntimeError(f'Duplicate tag: {tag}')
            self.variables[tag] = {'srcs': list_wrap_str(srcs), 'dsts': list_wrap_str(dsts)}
            

    async def send(self, tag: str, obj: Any, dsts: List[str]=None, suffix: str='') -> None:
        srcs = self.variables[tag]['srcs']
        _dsts = self.variables[tag]['dsts']
        dsts = _dsts if dsts is None else list_wrap_str(dsts)
        assert set(dsts).issubset(_dsts) or Role.broadcast in _dsts, \
            f'{dsts} is neither None nor a subset of {_dsts}'
        assert self.role in srcs or Role.broadcast in srcs, \
            f'role {self.role} is not configured to be source of variable {tag}:{srcs}->{dsts}'
        await self.mailman.send(tag+suffix, self.role, dsts, obj)
        

    async def get(self, tag: str, suffix: str='', srcs: Union[str, List[str]]=None) -> Any:
        _srcs = self.variables[tag]['srcs']
        dsts = self.variables[tag]['dsts']
        srcs = _srcs if srcs is None else list_wrap_str(srcs)
        assert Role.broadcast in dsts or self.role in dsts, \
            f'role {self.role} is not configured to be destination of variable {tag}:{srcs}->{dsts}'
        packets = await self.mailman.get(tag+suffix, srcs)
        return [packet.obj for packet in packets] if isinstance(packets, list) else packets.obj


class Packet:
    def __init__(self, tag, src, dst, obj):
        self.tag = tag
        self.src = src
        self.dst = dst
        self.obj = deepcopy(obj)


class Mailman:
    _instances = {}


    @classmethod
    def get_instance(cls, role: str):
        return cls._instances[role]


    @classmethod
    def create_or_get(cls, role: str):
        if role not in cls._instances:
            cls._instances[role] = cls()
        return cls._instances[role]


    def __init__(self):
        self.send_channels = {}
        self.receive_channels = {}
        self.receive_qs = {}
        self.inbox = {}


    async def send(self, tag, src, dsts: Union[List, str], obj):
        dsts = list_wrap_str(dsts)
        async with trio.open_nursery() as nursery:
            for dst, ch in self.send_channels.items():
                if Role.broadcast in dsts or dst in dsts:
                    nursery.start_soon(ch.send, Packet(tag, src, dst, obj))

    
    async def get(self, tag, srcs: Union[List,str]):
        srcs = list_wrap_str(srcs)
        srcs = self.receive_channels.keys() if Role.broadcast in srcs else srcs
        for src in srcs:
            assert src in self.receive_channels, f'{src} is not in receive_channels: {list(self.receive_channels.keys())}.'
        packets = []
        with trio.fail_after(60):
            for src in srcs:
                while tag not in self.inbox[src]:
                    await trio.sleep(1e-3)
                packets.append(await self.inbox[src][tag].get())
        return packets[0] if len(packets)==1 else packets

    
    async def serve(self):
        async def _async_fetch(src, ch):
            async for packet in ch:
                tag = packet.tag
                if tag not in self.inbox[src]:
                    self.inbox[src][tag] = TrioQueue()
                self.inbox[src][tag].put_nowait(packet)

        self.inbox = {}
        async with trio.open_nursery() as nursery:
            for src,ch in self.receive_channels.items():
                self.inbox[src] = {}
                nursery.start_soon(_async_fetch, src, ch)


class TrioQueue:
    def __init__(self, maxlen=math.inf):
        self.put_channel, self.get_channel = trio.open_memory_channel(maxlen)


    async def put(self, value):
        return await self.put_channel.send(value)


    def put_nowait(self, value):
        return self.put_channel.send_nowait(value)


    async def get(self):
        return await self.get_channel.receive()

    
    def get_nowait(self):
        return self.get_channel.receive_nowait()


    def close(self):
        self.put_channel.close()
        self.get_channel.close()