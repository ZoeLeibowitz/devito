from devito import *
from devito.petsc import PETScSolve
import pandas as pd
from devito import configuration
import numpy as np
import sympy
configuration['opt'] = 'noop'


# Ensure that PetscInitialize and PetscFinalize are called
# only once per script, rather than for each Operator constructed.

n_values = [11, 13, 15]

for n in n_values:
    grid = Grid(shape=(n, n), dtype=np.float64)

    phi = Function(name='phi', grid=grid, space_order=2, dtype=np.float64)

    rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)

    eqn = Eq(rhs, phi.laplace, subdomain=grid.interior)

    rhs.data[:] = np.float64(5.0)

    petsc = PETScSolve(eqn, target=phi)

    with switchconfig(openmp=False):
        op = Operator(petsc)

    op.apply()

print(op.ccode)