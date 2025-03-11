import numpy as np
import ctypes
from ctypes import POINTER, c_int, c_char

from devito import Operator, switchconfig, configuration
from devito.types import Symbol, Constant
from devito.types.basic import DataSymbol
from devito.types.equation import PetscEq
from devito.petsc.types import Initialize, Finalize
from devito.tools import CustomDtype
configuration['opt'] = 'noop'

import sys; 


argcc = len(sys.argv)-1
argv = sys.argv
argv = argv[1:]


# b_string1 = argv.encode('utf-8')
encoded_strings = [s.encode('utf-8') for s in argv]
# argtypes = [ctypes.c_char_p] * len(encoded_strings)

argv = (ctypes.POINTER(ctypes.c_char) * len(encoded_strings))()

# Assign each encoded string as a pointer in the array
for i, b_string in enumerate(encoded_strings):
    argv[i] = ctypes.cast(b_string, ctypes.POINTER(ctypes.c_char))


# from IPython import embed; embed()
# Ensure that PetscInitialize and PetscFinalize are called
# only once per script, rather than for each Operator constructed.


dummy = Symbol(name='d')
# argc_symb = Constant(name='argc', dtype=np.int32, is_const=False)


# argv_type = ctypes.POINTER(ctypes.c_char_p)
# argv_type = ctypes.c_char_p
argv_type = ctypes.POINTER(ctypes.POINTER(c_char))

class argv_symbol(DataSymbol):
    @property
    def _C_ctype(self):
        return argv_type

argc_symb = DataSymbol(name='argc', dtype=np.int32)
argv_symb = argv_symbol(name='argv')


# from IPython import embed; embed()

# argv_symb = Constant(name='argv', dtype=ctypes.c_char_p, is_const=False)
# argv_symb = Constant(name='argv', dtype=c_dtype, is_const=False)


with switchconfig(openmp=False, mpi=True):

    op_init = Operator(
        [PetscEq(dummy, Initialize((argc_symb, argv_symb)))],
         name='kernel_init'
    )

    op_finalize = Operator(
        [PetscEq(dummy, Finalize(dummy))],
        name='kernel_finalize'
    )


# from IPython import embed; embed()
# argv = (ctypes.c_char_p * argcc)(*[s.encode('utf-8') for s in argv])
# from IPython import embed; embed()

op_init.apply(argc=argcc, argv=argv)
# op_finalize.apply()

print(op_init.ccode)
print(op_finalize.ccode)