from .scHPF_ import *
from .util import *

# TODO make this and other paths cross-platform
with open(__file__.rsplit('/',2)[0]+ '/VERSION') as f:
    __version__ = f.read().strip()
