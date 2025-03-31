from devito import *
import os
from examples.seismic.source import DGaussSource, TimeAxis
from examples.seismic import plot_image
import numpy as np
from devito.finite_differences.operators import div, grad
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


PetscInitialize()

# Initial grid: 1km x 1km, with spacing 100m
extent = (2000., 2000.)
shape = (81, 81)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1), dtype=np.float64))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1), dtype=np.float64))
grid = Grid(extent=extent, shape=shape, dimensions=(x, z), dtype=np.float64)


# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., 200.
dt = 1e2*(1. / np.sqrt(2.)) / 60.
# print(dt)
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = DGaussSource(name='src', grid=grid, f0=0.01, time_range=time_range, a=0.004)
src.coordinates.data[:] = [1000., 1000.]


# Now we create the velocity and pressure fields
p = TimeFunction(name='p', grid=grid, staggered=NODE, space_order=4)
v_x = TimeFunction(name='v_x', grid=grid, staggered=(x,), space_order=4, time_order=1)
v_z = TimeFunction(name='v_z', grid=grid, staggered=(z,), space_order=4, time_order=1)


# Now we create the velocity and pressure fields
# p = TimeFunction(name='p', grid=grid, staggered=NODE, space_order=2, time_order=1)
# v = VectorTimeFunction(name='v', grid=grid, space_order=2, time_order=1)


# p = TimeFunction(name='p', grid=grid, staggered=NODE, space_order=4, time_order=1)
# v = VectorTimeFunction(name='v', grid=grid, space_order=4, time_order=1)

t = grid.stepping_dim
time = grid.time_dim

# We need some initial conditions
V_p = 4.0
density = 1.

ro = 1/density
l2m = V_p*V_p*density

# The source injection term
src_p = src.inject(field=p.forward, expr=src)

# 2nd order acoustic according to fdelmoc
# u_v_2 = Eq(v.forward, solve(v.dt - ro * grad(p), v.forward))
# u_p_2 = Eq(p.forward, solve(p.dt - l2m * div(v.forward), p.forward))


# print(v.forward)
# from IPython import embed; embed()
# petsc
# 2nd order acoustic according to fdelmoc
# u_v_2_x = Eq(v_x.forward, solve(v_x.dt - ro * p.dx, v_x.forward))


# u_v_2_z = Eq(v_z.forward, solve(v_z.dt - ro * p.dz, v_z.forward))
# u_p_2 = Eq(p.forward, solve(p.dt - l2m * (v_x.forward.dx + v_z.forward.dz), p.forward))


# eqn_petsc = Eq(v_x.dt, ro * p.dx)
petsc1 = PETScSolve(Eq(v_x.dt, ro * p.dx), target=v_x.forward)
petsc2 = PETScSolve(Eq(v_z.dt, ro * p.dz), target=v_z.forward)


# DOUBLE CHECK DIV DOESN'T USE A DIFFERENT DISCRETISATION
petsc3 = PETScSolve(Eq(p.dt, l2m * (v_x.forward.dx + v_z.forward.dz)), target=p.forward, solver_parameters={'ksp_rtol': 1e-7})

# from IPython import embed; embed()


with switchconfig(language='petsc'):
    op_2 = Operator(petsc1 + petsc2 + petsc3 + src_p, opt='noop')
# op_2 = Operator([u_v_2, u_p_2] + src_p)
    op_2(time=src.time_range.num-1, dt=dt)



# op_2 = Operator(u_v_2_x, opt='noop')
# op_2(time=src.time_range.num-1, dt=dt)

# print(op_2.ccode)
norm_p = norm(p)
# print(p.data[:])
# assert np.isclose(norm_p, .35098, atol=1e-4, rtol=0)
print(norm_p)
