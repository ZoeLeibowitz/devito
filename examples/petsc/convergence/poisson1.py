from devito import *
from devito.petsc import PETScSolve, EssentialBC
import pandas as pd
from devito import configuration
import numpy as np
import sympy
configuration['opt'] = 'noop'
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# Solving pn.laplace = 2x(y - 1)(y - 2x + xy + 2)e^(x-y)
# Constant zero Dirichlet BCs.


# Subdomains to implement BCs
class SubTop(SubDomain):
    name = 'subtop'
    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('right', 1)}
sub1 = SubTop()

class SubBottom(SubDomain):
    name = 'subbottom'
    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('left', 1)}
sub2 = SubBottom()

class SubLeft(SubDomain):
    name = 'subleft'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: y}
sub3 = SubLeft()

class SubRight(SubDomain):
    name = 'subright'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: y}
sub4 = SubRight()


def analytical(x, y):
    tmp = np.float64(x-y)
    return np.float64(np.exp(x-y) * x * (1-x) * y * (1-y))


Lx = np.float64(1.)
Ly = np.float64(1.)

n_values = [13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173]
dx = np.array([Lx/(n-1) for n in n_values])
errors = []


for n in n_values:
    grid = Grid(shape=(n, n), extent=(Lx, Ly), subdomains=(sub1,sub2,sub3,sub4,), dtype=np.float64)

    phi = Function(name='phi', grid=grid, space_order=2, dtype=np.float64)

    rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)

    eqn = Eq(rhs, phi.laplace, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)
    rhs.data[:] = np.float64(2.0*X*(Y-1.0)*(Y - 2.0*X + X*Y + 2.0)) * np.float64(np.exp(X-Y))

    # # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(phi, np.float64(0.), subdomain=sub1)]
    bcs += [EssentialBC(phi, np.float64(0.), subdomain=sub2)]
    bcs += [EssentialBC(phi, np.float64(0.), subdomain=sub3)]
    bcs += [EssentialBC(phi, np.float64(0.), subdomain=sub4)]

    petsc = PETScSolve([eqn]+bcs, target=phi, solver_parameters={'ksp_rtol': 1e-8})

    op = Operator(petsc)

    op.apply()

    phi_analytical = analytical(X, Y)

    error = np.linalg.norm(phi_analytical[1:-1,1:-1] - phi.data[1:-1, 1:-1]) / np.linalg.norm(phi_analytical[1:-1,1:-1])

    errors.append(error)

print(errors)   
slope, _ = np.polyfit(np.log(dx), np.log(errors), 1)
print(slope)

assert slope > 1.9
assert slope < 2.1
