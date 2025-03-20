import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, configuration, TimeFunction, switchconfig)
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
from devito.data.allocators import ALLOC_PETSC
import devito.data.allocators
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()

nx = 81
ny = 81

grid = Grid(shape=(nx, ny), extent=(2., 2.), dtype=np.float64)

u = Function(name='u', grid=grid, dtype=np.float64, space_order=2, allocator=ALLOC_PETSC)
v = Function(name='v', grid=grid, dtype=np.float64, space_order=2, allocator=ALLOC_PETSC)

v.data[:] = 5.0

eq = Eq(v, u.laplace, subdomain=grid.interior)

petsc = PETScSolve([eq], u)

op = Operator(petsc)
print(op.ccode)
op.apply()
