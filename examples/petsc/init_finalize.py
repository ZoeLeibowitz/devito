from devito import Operator, switchconfig, configuration
from devito.types import Symbol
from devito.types.equation import PetscEq
from devito.petsc.types import Initialize, Finalize
configuration['opt'] = 'noop'


# Ensure that PetscInitialize and PetscFinalize are called
# only once per script, rather than for each Operator constructed.

s = Symbol(name='s')
with switchconfig(openmp=False, mpi=True):
    op_init = Operator([PetscEq(s, Initialize(s))], name='kernel_init')
    op_finalize = Operator([PetscEq(s, Finalize(s))], name='kernel_finalize')

# currently this runs with DEVITO_MPI=0
op_init.apply()
op_finalize.apply()

print(op_init.ccode)
print(op_finalize.ccode)
