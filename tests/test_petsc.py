from devito import Grid
from devito.ir.iet import Call, ElementalFunction, Definition, DummyExpr
from devito.passes.iet.languages.C import CDataManager
from devito.types import (DM, Mat, Vec, PetscMPIInt, KSP,
                          PC, KSPConvergedReason, PETScFunction)
import numpy as np


def test_petsc_local_object():
    """
    Test C++ support for PETSc LocalObjects.
    """
    lo0 = DM('da')
    lo1 = Mat('A')
    lo2 = Vec('x')
    lo3 = PetscMPIInt('size')
    lo4 = KSP('ksp')
    lo5 = PC('pc')
    lo6 = KSPConvergedReason('reason')

    iet = Call('foo', [lo0, lo1, lo2, lo3, lo4, lo5, lo6])
    iet = ElementalFunction('foo', iet, parameters=())

    dm = CDataManager(sregistry=None)
    iet = CDataManager.place_definitions.__wrapped__(dm, iet)[0]

    assert 'DM da;' in str(iet)
    assert 'Mat A;' in str(iet)
    assert 'Vec x;' in str(iet)
    assert 'PetscMPIInt size;' in str(iet)
    assert 'KSP ksp;' in str(iet)
    assert 'PC pc;' in str(iet)
    assert 'KSPConvergedReason reason;' in str(iet)


def test_petsc_functions():
    """
    Test C++ support for PETScFunctions.
    """
    grid = Grid((2, 2))
    x, y = grid.dimensions

    ptr0 = PETScFunction(name='ptr0', dtype=np.float32, dimensions=grid.dimensions,
                         shape=grid.shape)
    ptr1 = PETScFunction(name='ptr1', dtype=np.float32, grid=grid)

    defn0 = Definition(ptr0)
    defn1 = Definition(ptr1)

    expr = DummyExpr(ptr0.indexed[x, y], ptr1.indexed[x, y] + 1)

    assert str(defn0) == 'PetscScalar**restrict ptr0;'
    assert str(defn1) == 'PetscScalar**restrict ptr1;'
    assert str(expr) == 'ptr0[x][y] = ptr1[x][y] + 1;'