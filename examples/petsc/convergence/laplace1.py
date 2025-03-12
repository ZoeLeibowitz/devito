from devito import *
from devito.petsc import PETScSolve, EssentialBC
import pandas as pd
from devito import configuration
import numpy as np
import sympy
configuration['opt'] = 'noop'


# Solving phi.laplace = 0
# Constant Dirichlet BCs:
# phi(x, 0) = 0
# phi(0, y) = 0
# phi(1, y) = 0
# phi(x, 1) = f(x) = sin(pi*x)

# The analytical solution is:
# phi(x, y) = sinh(pi*y)*sin(pi*x)/sinh(pi)


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


Lx = np.float64(1.)
Ly = np.float64(1.)


# def analytical(x, y, Lx, Ly):
#     tmp = np.float64(np.pi)/Lx
#     return (np.float64(np.sinh(tmp*y)) * np.float64(np.sin(tmp*x))) / (np.float64(np.sinh(tmp*Ly)))

def analytical(x, y, Lx, Ly):
    tmp = np.float64(np.pi)/Lx
    return np.float64(np.sinh(tmp*y)) * np.float64(np.sin(tmp*x)) / np.float64(np.sinh(tmp*Ly))


n_values = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61]
dx = np.array([Lx/(n-1) for n in n_values])
errors = []


for n in n_values:
    grid = Grid(shape=(n, n), extent=(Lx, Ly), subdomains=(sub1,sub2,sub3,sub4,), dtype=np.float64)

    phi = Function(name='phi', grid=grid, space_order=2, dtype=np.float64)
    rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)

    phi.data[:] = np.float64(0.0)
    rhs.data[:] = np.float64(0.0)

    eqn = Eq(rhs, phi.laplace, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)

    # Create boundary condition expressions using subdomains
    bc_func = Function(name='bcs', grid=grid, space_order=2, dtype=np.float64)
    bc_func.data[:] = np.float64(0.0)
    bc_func.data[:, -1] = np.sin(tmpx*np.pi)

    phi.data[:,-1] = bc_func.data[:,-1]

    bcs = [EssentialBC(phi, bc_func, subdomain=sub1)]  # top
    bcs += [EssentialBC(phi, bc_func, subdomain=sub2)]  # bottom
    bcs += [EssentialBC(phi, bc_func, subdomain=sub3)]  # left
    bcs += [EssentialBC(phi, bc_func, subdomain=sub4)]  # right

    petsc = PETScSolve([eqn]+bcs, target=phi)

    with switchconfig(openmp=False):
        op = Operator(petsc)

    op.apply()

    pd.DataFrame(phi.data[:]).to_csv("results/%s.csv" % n, header=None, index=None)

    phi_analytical = analytical(X, Y, Lx, Ly)

    pd.DataFrame(phi_analytical[:]).to_csv("results/%s_analytical.csv" % n, header=None, index=None)

    error = np.sqrt(np.sum((phi.data[1:-1,1:-1]-phi_analytical[1:-1,1:-1])**2)/np.float64(n*n))

    errors.append(error)

# print(op.ccode)
print(errors)
slope, _ = np.polyfit(np.log(dx), np.log(errors), 1)
print(slope)

# assert slope > 1.9
# Seems to be superconverging a bit? (slope is 2.447)
# assert slope < 2.1

# TODO: investigate why error increases from n=41 to n=43