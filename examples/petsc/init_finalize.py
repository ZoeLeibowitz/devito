from devito import Grid, Function, Eq, Operator, switchconfig
from devito.types import Symbol
from devito.types.equation import PetscEq
from devito.petsc import PETScSolve
from devito.petsc.types import LinearSolveExpr, Initialize
import pandas as pd
from devito import configuration
import numpy as np
import sympy
configuration['opt'] = 'noop'


# Ensure that PetscInitialize and PetscFinalize are called
# only once per script, rather than for each Operator constructed.

init = Symbol(name='petscinit')

op_init = Operator([PetscEq(init, Initialize(init))])

print(op_init.ccode)


# n_values = [11, 13, 15]
n_values = [11]

for n in n_values:
    grid = Grid(shape=(n, n), dtype=np.float64)

    phi = Function(name='phi', grid=grid, space_order=2, dtype=np.float64)

    rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)

    eqn = Eq(rhs, phi.laplace, subdomain=grid.interior)

    rhs.data[:] = np.float64(5.0)

    petsc = PETScSolve(eqn, target=phi)

    with switchconfig(openmp=False):
        op = Operator(petsc)

    # op.apply()

print(op.ccode)