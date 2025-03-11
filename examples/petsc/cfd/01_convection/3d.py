from examples.cfd import init_hat
import numpy as np

from devito import Grid, TimeFunction, Eq, solve, Operator, SubDomain
from devito.petsc import PETScSolve, EssentialBC

##############################################################################################
# check petsc runs in 3d
# Some variable declarations
nx = 81
ny = 81
nz = 81

nt = 100
c = 1.
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
dz = 2. / (nz - 1)
sigma = .2
dt = sigma * dx


grid = Grid(shape=(nx, ny, nz), extent=(2., 2., 2.))
u = TimeFunction(name='u', grid=grid)

eq = Eq(u.dt + c*u.dxl + c*u.dyl + c*u.dzl, subdomain=grid.interior)

petsc = PETScSolve([eq], u.forward)

op = Operator(petsc)
print(op.ccode)
op(time=nt, dt=dt)
