
import numpy as np

from devito import Grid, Function, Eq, solve, Operator, SubDomain, switchconfig
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize

PetscInitialize()

# Test in single precision i.e PETSc must be configured with single precision
# TODO: add warnings for this etc

# Some variable declarations
nx = 81
ny = 81
nt = 100
c = 1.
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
sigma = .2
dt = sigma * dx

grid = Grid(shape=(nx, ny), extent=(2., 2.), dtype=np.float64)

u = Function(name='u', grid=grid, dtype=np.float64, space_order=2)
v = Function(name='v', grid=grid, dtype=np.float64, space_order=2)

eq = Eq(v, u.laplace, subdomain=grid.interior)


petsc = PETScSolve([eq], u)

op = Operator(petsc)
