from typing import Union, List
class Role:
    active = 'active'
    passive = 'passive'
    assistant = 'assistant'
    broadcast = 'broadcast'


def get_full_path_name(obj: object) -> str:
    """Return the full path and name of a class
    """
    typ = obj if isinstance(obj, type) else type(obj) 
    return f"{typ.__module__}.{typ.__name__}"


def list_wrap_str(str_list: Union[str, List[str]]):
    return str_list if isinstance(str_list, list) else [str_list]


class RandomSeed:
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self, seed: int=None):
        self.seed = seed


RANDOM_SEED = RandomSeed()
