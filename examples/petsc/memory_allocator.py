import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, configuration)
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
from devito.petsc.allocators import PetscMemoryAllocator
import devito.data.allocators
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

devito.data.allocators.ALLOC_NUMA_ANY = PetscMemoryAllocator()

PetscInitialize()

nx = 81
ny = 81

grid = Grid(shape=(nx, ny), extent=(2., 2.), dtype=np.float64)

u = Function(name='u', grid=grid, dtype=np.float64, space_order=2)
u.data._allocator = PetscMemoryAllocator()
v = Function(name='v', grid=grid, dtype=np.float64, space_order=2)

v.data[:] = 5.0

eq = Eq(v, u.laplace, subdomain=grid.interior)

petsc = PETScSolve([eq], u)

op = Operator(petsc)

op.apply()