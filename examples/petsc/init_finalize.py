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


argcc = len(sys.argv)
argv = sys.argv

# Ensure that PetscInitialize and PetscFinalize are called
# only once per script, rather than for each Operator constructed.


# argc
# array of strings ()

c_dtype = CustomDtype('char',  modifier=' **')

dummy = Symbol(name='d')
# argc_symb = Constant(name='argc', dtype=np.int32, is_const=False)


# argv_type = ctypes.POINTER(ctypes.c_char_p)
argv_type = ctypes.POINTER(ctypes.POINTER(c_char))




argc_symb = DataSymbol(name='argc', dtype=np.int32)
argv_symb = DataSymbol(name='argv', dtype=argv_type)

# argv_type = ctypes.POINTER(ctypes.POINTER(c_char))

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
op_init.apply(argc=argcc, argv=argv)
# op_finalize.apply()

print(op_init.ccode)
print(op_finalize.ccode)
