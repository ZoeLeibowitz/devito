import numpy as np
from sympy.printing.c import C99CodePrinter

from devito.ir import Call, BasePrinter
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.symbolics.extended_dtypes import c_complex, c_double_complex
from devito.petsc.utils import petsc_type_mappings

__all__ = ['CBB', 'CDataManager', 'COrchestrator']


class CBB(LangBB):

    mapper = {
        # Complex
        'includes-complex': 'complex.h',
        # Allocs
        'header-memcpy': 'string.h',
        'host-alloc': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-alloc-pin': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-free': lambda i:
            Call('free', (i,)),
        'host-free-pin': lambda i:
            Call('free', (i,)),
        'alloc-global-symbol': lambda i, j, k:
            Call('memcpy', (i, j, k)),
    }


class CDataManager(DataManager):
    langbb = CBB


class COrchestrator(Orchestrator):
    langbb = CBB


class CPrinter(BasePrinter, C99CodePrinter):

    _default_settings = {**BasePrinter._default_settings,
                         **C99CodePrinter._default_settings}
    _func_literals = {np.float32: 'f', np.complex64: 'f'}
    _func_prefix = {np.float32: 'f', np.float64: 'f',
                    np.complex64: 'c', np.complex128: 'c'}
    _includes = ['stdlib.h', 'math.h', 'sys/time.h']

    # These cannot go through _print_xxx because they are classes not
    # instances
    type_mappings = {**C99CodePrinter.type_mappings,
                     c_complex: 'float _Complex',
                     c_double_complex: 'double _Complex'}

    def _print_ImaginaryUnit(self, expr):
        return '_Complex_I'


class PetscCPrinter(CPrinter):
    _restrict_keyword = ''

    type_mappings = {**CPrinter.type_mappings, **petsc_type_mappings}
