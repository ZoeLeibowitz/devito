from functools import singledispatch

import sympy

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions
from devito.symbolics import uxreplace
from devito.petsc.types import PETScArray, LinearSolveExpr, MatVecEq, RHSEq

__all__ = ['PETScSolve']


def PETScSolve(eq, target, bcs=None, solver_parameters=None, **kwargs):

    y_matvec, x_matvec, b_tmp = [
        PETScArray(name=f'{prefix}_{target.name}',
                   dtype=target.dtype,
                   dimensions=target.space_dimensions,
                   shape=target.grid.shape,
                   liveness='eager',
                   halo=[target.halo[d] for d in target.space_dimensions])
        for prefix in ['y_matvec', 'x_matvec', 'b_tmp']]

    b, F_target = separate_eqn(eq, target)

    # Args were updated so need to update target to enable uxreplace on F_target
    new_target = {f for f in retrieve_functions(F_target) if
                  f.function == target.function}
    assert len(new_target) == 1  # Sanity check: only one target expected
    new_target = new_target.pop()

    # TODO: Current assumption is that problem is linear and user has not provided
    # a jacobian. Hence, we can use F_target to form the jac-vec product
    matvecaction = MatVecEq(
        y_matvec, LinearSolveExpr(uxreplace(F_target, {new_target: x_matvec}),
                                  target=target, solver_parameters=solver_parameters),
        subdomain=eq.subdomain)

    # Part of pde that remains constant at each timestep
    rhs = RHSEq(b_tmp, LinearSolveExpr(b, target=target,
                solver_parameters=solver_parameters), subdomain=eq.subdomain)

    if not bcs:
        return [matvecaction, rhs]

    bcs_for_matvec = []
    for bc in bcs:
        # TODO: Insert code to distiguish between essential and natural
        # boundary conditions since these are treated differently within
        # the solver
        # NOTE: May eventually remove the essential bcs from the solve
        # (and move to rhs) but for now, they are included since this
        # is not trivial to implement when using DMDA
        # NOTE: Below is temporary -> Just using this as a palceholder for
        # the actual BC implementation for the matvec callback
        new_rhs = bc.rhs.subs(target, x_matvec)
        bc_rhs = LinearSolveExpr(
            new_rhs, target=target, solver_parameters=solver_parameters
        )
        bcs_for_matvec.append(MatVecEq(y_matvec, bc_rhs, subdomain=bc.subdomain))

    return [matvecaction] + bcs_for_matvec + [rhs]


def separate_eqn(eqn, target):
    """
    Separate the equation into two separate expressions,
    where F(target) = b.
    """
    zeroed_eqn = Eq(eqn.lhs - eqn.rhs, 0)
    tmp = eval_time_derivatives(zeroed_eqn.lhs)
    b, F_target = remove_target(tmp, target)
    return -b, F_target


@singledispatch
def remove_target(expr, target):
    return (0, expr) if expr == target else (expr, 0)


@remove_target.register(sympy.Add)
def _(expr, target):
    if not expr.has(target):
        return (expr, 0)

    args_b, args_F = zip(*(remove_target(a, target) for a in expr.args))
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_target.register(Mul)
def _(expr, target):
    if not expr.has(target):
        return (expr, 0)

    args_b, args_F = zip(*[remove_target(a, target) if a.has(target)
                           else (a, a) for a in expr.args])
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_target.register(Derivative)
def _(expr, target):
    return (0, expr) if expr.has(target) else (expr, 0)


@singledispatch
def centre_stencil(expr, target):
    """
    Extract the centre stencil from an expression. Its coefficient is what
    would appear on the diagonal of the matrix system if the matrix were
    formed explicitly.
    """
    return expr if expr == target else 0


@centre_stencil.register(sympy.Add)
def _(expr, target):
    if not expr.has(target):
        return 0

    args = [centre_stencil(a, target) for a in expr.args]
    return expr.func(*args, evaluate=False)


@centre_stencil.register(Mul)
def _(expr, target):
    if not expr.has(target):
        return 0

    args = []
    for a in expr.args:
        if not a.has(target):
            args.append(a)
        else:
            args.append(centre_stencil(a, target))

    return expr.func(*args, evaluate=False)


@centre_stencil.register(Derivative)
def _(expr, target):
    if not expr.has(target):
        return 0
    args = [centre_stencil(a, target) for a in expr.evaluate.args]
    return expr.evaluate.func(*args)