from functools import singledispatch

from devito.finite_differences.differentiable import Add, Mul, EvalDerivative
from devito.finite_differences.derivative import Derivative
from devito.types import Eq
from devito.operations.solve import eval_time_derivatives
from devito.petsc.types import PETScArray, LinearSolveExpr, MatVecEq, RHSEq

from sympy import simplify


__all__ = ['PETScSolve']


def PETScSolve(eq, target, bcs=None, solver_parameters=None, **kwargs):
    # TODO: Add check for time dimensions and utilise implicit dimensions.

    is_time_dep = any(dim.is_Time for dim in target.dimensions)
    # TODO: Current assumption is rhs is part of pde that remains
    # constant at each timestep. Need to insert function to extract this from eq.
    y_matvec, x_matvec, b_tmp = [
        PETScArray(name=f'{prefix}_{target.name}',
                   dtype=target.dtype,
                   dimensions=target.space_dimensions,
                   shape=target.grid.shape,
                   liveness='eager',
                   halo=target.halo[1:] if is_time_dep else target.halo)
        for prefix in ['y_matvec', 'x_matvec', 'b_tmp']]

    # TODO: Extend to rearrange equation for implicit time stepping.
    matvecaction = MatVecEq(y_matvec, LinearSolveExpr(eq.lhs.subs(target, x_matvec),
                            target=target, solver_parameters=solver_parameters),
                            subdomain=eq.subdomain)

    # Part of pde that remains constant at each timestep
    rhs = RHSEq(b_tmp, LinearSolveExpr(eq.rhs, target=target,
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
    b = remove_target(tmp, target)
    F_target = simplify(tmp - b)

    return -b, F_target


@singledispatch
def remove_target(expr, target):
    return 0 if expr == target else expr


@remove_target.register(Add)
@remove_target.register(EvalDerivative)
def _(expr, target):
    if not expr.has(target):
        return expr

    args = [remove_target(a, target) for a in expr.args]
    return expr.func(*args, evaluate=False)


@remove_target.register(Mul)
def _(expr, target):
    if not expr.has(target):
        return expr

    args = []
    for a in expr.args:
        if not a.has(target):
            args.append(a)
        else:
            a1 = remove_target(a, target)
            args.append(a1)

    return expr.func(*args, evaluate=False)


@remove_target.register(Derivative)
def _(expr, target):
    return 0 if expr.has(target) else expr