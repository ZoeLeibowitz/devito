from examples.cfd import init_hat
import numpy as np

from devito import Grid, TimeFunction, Eq, solve, Operator, SubDomain
from devito.petsc import PETScSolve, EssentialBC


# Test in single precision i.e PETSc must be configured with single precision
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

# Create field and assign initial conditions
u = np.empty((nx, ny))
init_hat(field=u, dx=dx, dy=dy, value=2.)


for n in range(nt + 1):
    # Copy previous result into a new buffer
    un = u.copy()
    
    # Update the new result with a 3-point stencil
    u[1:, 1:] = (un[1:, 1:] - (c * dt / dy * (un[1:, 1:] - un[1:, :-1])) -
                              (c * dt / dx * (un[1:, 1:] - un[:-1, 1:])))

    # Apply boundary conditions. 
    u[0, :] = 1.  # left
    u[-1, :] = 1. # right
    u[:, 0] = 1.  # bottom
    u[:, -1] = 1. # top


# A small sanity check for auto-testing
assert (u[45:55, 45:55] > 1.8).all()
u_ref = u.copy()

##############################################################################################
# Standard Devito
grid = Grid(shape=(nx, ny), extent=(2., 2.))
u = TimeFunction(name='u', grid=grid)

eq = Eq(u.dt + c*u.dxl + c*u.dyl, subdomain=grid.interior)
stencil = solve(eq, u.forward)


init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
init_hat(field=u.data[1], dx=dx, dy=dy, value=2.)

x, y = grid.dimensions
t = grid.stepping_dim
bc_left = Eq(u[t + 1, 0, y], 1.)
bc_right = Eq(u[t + 1, nx-1, y], 1.)
bc_top = Eq(u[t + 1, x, ny-1], 1.)
bc_bottom = Eq(u[t + 1, x, 0], 1.)

# Now combine the BC expressions with the stencil to form operator
expressions = [Eq(u.forward, stencil)]
expressions += [bc_left, bc_right, bc_top, bc_bottom]
op = Operator(expressions=expressions, opt=None, openmp=False)  # <-- Turn off performance optimisations
op(time=nt, dt=dt)


# Some small sanity checks for the testing framework
assert (u.data[0, 45:55, 45:55] > 1.8).all()
assert np.allclose(u.data[0], u_ref, rtol=3.e-2)


##############################################################################################
# check petsc
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

grid = Grid(shape=(nx, ny), extent=(2., 2.), subdomains=(sub1, sub2, sub3, sub4))
u = TimeFunction(name='u', grid=grid)

eq = Eq(u.dt + c*u.dxl + c*u.dyl, subdomain=grid.interior)

init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
init_hat(field=u.data[1], dx=dx, dy=dy, value=2.)

bcs = [EssentialBC(u.forward, 1., subdomain=sub1)]
bcs += [EssentialBC(u.forward, 1., subdomain=sub2)]
bcs += [EssentialBC(u.forward, 1., subdomain=sub3)]
bcs += [EssentialBC(u.forward, 1., subdomain=sub4)]

petsc = PETScSolve([eq]+bcs, u.forward)

op = Operator(petsc)
print(op.ccode)
op(time=nt, dt=dt)

assert (u.data[0, 45:55, 45:55] > 1.8).all()
assert np.allclose(u.data[0], u_ref, rtol=3.e-2)