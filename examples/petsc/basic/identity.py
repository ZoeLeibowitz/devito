from devito import Grid, Function, Eq, Operator
from devito.petsc import PETScSolve
import numpy as np

grid = Grid(shape=(11, 11), dtype=np.float64)

u = Function(name='u', grid=grid, space_order=2, dtype=np.float64)
v = Function(name='v', grid=grid, space_order=2, dtype=np.float64)

# Solving Ax=b where A is the identity matrix
v.data[:] = 5.0
eqn = Eq(u, v)
petsc = PETScSolve(eqn, target=u)

op = Operator(petsc)
print(op.ccode)

# Check the solve function returns the correct output
op.apply()
assert np.allclose(u.data, v.data)