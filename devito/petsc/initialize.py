import sys
from ctypes import POINTER, cast, c_char
import atexit

from devito import Operator, switchconfig
from devito.types import Symbol
from devito.types.equation import PetscEq
from devito.petsc.types import Initialize, Finalize

global _petsc_initialized
_petsc_initialized = False


def PetscInitialize():
    global _petsc_initialized
    
    if not _petsc_initialized:
        dummy = Symbol(name='d')

        with switchconfig(openmp=False, mpi=True):

            op_init = Operator(
                [PetscEq(dummy, Initialize(dummy))],
                name='kernel_init', opt='noop'
            )

            op_finalize = Operator(
                [PetscEq(dummy, Finalize(dummy))],
                name='kernel_finalize', opt='noop'
            )

        encoded = [s.encode('utf-8') for s in sys.argv]
        argv = (POINTER(c_char) * len(sys.argv))()
        for i, string in enumerate(encoded):
            argv[i] = cast(string, POINTER(c_char))

        # argv_type = (POINTER(c_char) * len(sys.argv))
        # from IPython import embed; embed()
        op_init.apply(argc=len(sys.argv), argv=argv)

        atexit.register(op_finalize.apply)
        _petsc_initialized = True

