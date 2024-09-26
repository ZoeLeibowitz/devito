from devito.ir.iet import Expression, Callback, FixedArgsCallable, Call
from devito.ir.equations import OpInjectSolve


class LinearSolverExpression(Expression):
    """
    Base class for general expressions required by a
    matrix-free linear solve of the form Ax=b.
    """
    pass


class InjectSolveDummy(LinearSolverExpression):
    """
    Placeholder expression to run the iterative solver.
    """
    def __init__(self, expr, pragmas=None, operation=OpInjectSolve):
        super().__init__(expr, pragmas=pragmas, operation=operation)


class PETScCallable(FixedArgsCallable):
    pass


class MatVecCallback(Callback):
    @property
    def callback_form(self):
        param_types_str = ', '.join([str(t) for t in self.param_types])
        return "(%s (*)(%s))%s" % (self.retval, param_types_str, self.name)


class FormFunctionCallback(Callback):
    @property
    def callback_form(self):
        return "%s" % self.name


class PETScCall(Call):
    pass
