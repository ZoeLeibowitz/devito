import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, configuration, TimeFunction, switchconfig)
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
# from devito.petsc.allocators import PetscMemoryAllocator
# import devito.data.allocators
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# devito.data.allocators.ALLOC_NUMA_ANY = PetscMemoryAllocator()

PetscInitialize()

# nx = 81
# ny = 81

# grid = Grid(shape=(nx, ny), extent=(2., 2.), dtype=np.float64)

# u = Function(name='u', grid=grid, dtype=np.float64, space_order=2)
# # u.data._allocator = PetscMemoryAllocator()
# v = Function(name='v', grid=grid, dtype=np.float64, space_order=2)

# v.data[:] = 5.0

# eq = Eq(v, u.laplace, subdomain=grid.interior)

# petsc = PETScSolve([eq], u)

# op = Operator(petsc)
# # print(op.ccode)
# op.apply()



grid = Grid((11, 11))

# Modulo time stepping
u1 = TimeFunction(name='u1', grid=grid, space_order=2)
v1 = Function(name='v1', grid=grid, space_order=2)
eq1 = Eq(v1.laplace, u1)
petsc1 = PETScSolve(eq1, v1)
with switchconfig(openmp=False):
    op1 = Operator(petsc1)
    print(op1.ccode)
op1.apply(time_M=5)