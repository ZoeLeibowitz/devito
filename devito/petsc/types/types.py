import sympy

from devito.tools import Reconstructable, sympy_mutex


class LinearSolveExpr(sympy.Function, Reconstructable):

    __rargs__ = ('expr',)
    __rkwargs__ = ('target', 'solver_parameters', 'matvecs',
                   'formfuncs', 'formrhs', 'arrays', 'time_mapper')

    defaults = {
        'ksp_type': 'gmres',
        'pc_type': 'jacobi',
        'ksp_rtol': 1e-7,  # Relative tolerance
        'ksp_atol': 1e-50,  # Absolute tolerance
        'ksp_divtol': 1e4,  # Divergence tolerance
        'ksp_max_it': 10000  # Maximum iterations
    }

    def __new__(cls, expr, target=None, solver_parameters=None,
                matvecs=None, formfuncs=None, formrhs=None,
                arrays=None, time_mapper=None, **kwargs):

        if solver_parameters is None:
            solver_parameters = cls.defaults
        else:
            for key, val in cls.defaults.items():
                solver_parameters[key] = solver_parameters.get(key, val)

        with sympy_mutex:
            obj = sympy.Function.__new__(cls, expr)

        obj._expr = expr
        obj._target = target
        obj._solver_parameters = solver_parameters
        obj._matvecs = matvecs
        obj._formfuncs = formfuncs
        obj._formrhs = formrhs
        obj._arrays = arrays
        obj._time_mapper = time_mapper
        return obj

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.expr)

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    def __hash__(self):
        return hash(self.expr)

    def __eq__(self, other):
        return (isinstance(other, LinearSolveExpr) and
                self.expr == other.expr and
                self.target == other.target)

    @property
    def expr(self):
        return self._expr

    @property
    def target(self):
        return self._target

    @property
    def solver_parameters(self):
        return self._solver_parameters

    @property
    def matvecs(self):
        return self._matvecs

    @property
    def formfuncs(self):
        return self._formfuncs

    @property
    def formrhs(self):
        return self._formrhs

    @property
    def arrays(self):
        return self._arrays

    @property
    def time_mapper(self):
        return self._time_mapper

    @classmethod
    def eval(cls, *args):
        return None

    func = Reconstructable._rebuild
