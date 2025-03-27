import os
import sys
from ctypes import POINTER, cast, c_char
import atexit

from devito import Operator, switchconfig
from devito.types import Symbol
from devito.types.equation import PetscEq
from devito.petsc.types import Initialize, Finalize, AllocateMemory



dummy = Symbol(name='d')

with switchconfig(language='petsc'):
    op_memory = Operator(
        [PetscEq(dummy, AllocateMemory(dummy))],
        name='kernel_allocate_memory', opt='noop'
    )
