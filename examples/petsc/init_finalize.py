import numpy as np
import ctypes
import sys
from ctypes import POINTER, c_int, c_char

from devito import Operator, switchconfig, configuration
from devito.types import Symbol, Constant
from devito.types.basic import DataSymbol
from devito.types.equation import PetscEq
from devito.petsc.types import Initialize, Finalize
from devito.tools import CustomDtype
configuration['opt'] = 'noop'


# Ensure that PetscInitialize and PetscFinalize are called
# only once per script, rather than for each Operator constructed.

argcc = len(sys.argv)
argv = sys.argv

encoded = [s.encode('utf-8') for s in argv]
argv = (ctypes.POINTER(ctypes.c_char) * len(encoded))()

for i, string in enumerate(encoded):
    argv[i] = ctypes.cast(string, ctypes.POINTER(ctypes.c_char))

dummy = Symbol(name='d')

class argv_symbol(DataSymbol):
    @property
    def _C_ctype(self):
        return ctypes.POINTER(ctypes.POINTER(c_char))

argc_symb = DataSymbol(name='argc', dtype=np.int32)
argv_symb = argv_symbol(name='argv')


with switchconfig(openmp=False, mpi=True):

    op_init = Operator(
        [PetscEq(dummy, Initialize((argc_symb, argv_symb)))],
         name='kernel_init'
    )

    op_finalize = Operator(
        [PetscEq(dummy, Finalize(dummy))],
        name='kernel_finalize'
    )


op_init.apply(argc=argcc, argv=argv)
op_finalize.apply()


# print(op_init.ccode)
# print(op_finalize.ccode)