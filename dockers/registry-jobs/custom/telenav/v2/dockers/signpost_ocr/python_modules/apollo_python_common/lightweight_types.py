from collections import namedtuple

Point = namedtuple('Point', 'x y')


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
