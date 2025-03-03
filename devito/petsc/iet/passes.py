import cgen as c
import numpy as np

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, BlankLine,
                           DummyExpr, CallableBody, List, Call, Callable)
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.types import Symbol, Scalar
from devito.petsc.types import (PetscMPIInt, PetscErrorCode, MultipleFieldData,
                                PointerIS, Mat, LocalVec, GlobalVec, CallbackMat, SNES,
                                DummyArg, PetscInt, PointerDM, PointerMat, MatReuse,
<<<<<<< HEAD
                                CallbackPointerIS, CallbackPointerDM, JacobianStruct,
                                SubMatrixStruct)
=======
                                CallbackPointerIS, CallbackPointerDM, JacobianStruct, SubMatrixStruct)
>>>>>>> 469f4e26a (change names)
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.petsc.utils import core_metadata
from devito.petsc.iet.routines import (CBBuilder, CCBBuilder, BaseObjectBuilder,
                                       CoupledObjectBuilder, BaseSetup, CoupledSetup,
                                       Solver, CoupledSolver, TimeDependent,
                                       NonTimeDependent)
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi


@iet_pass
def lower_petsc(iet, **kwargs):
    # Check if PETScSolve was used
    injectsolve_mapper = MapNodes(Iteration, InjectSolveDummy,
                                  'groupby').visit(iet)

    if not injectsolve_mapper:
        return iet, {}

    unique_grids = {i.expr.rhs.grid for (i,) in injectsolve_mapper.values()}
    # Assumption is that all solves are on the same grid
    if len(unique_grids) > 1:
        raise ValueError("All PETScSolves must use the same Grid, but multiple found.")
    grid = unique_grids.pop()
    objs.update({'grid': grid})

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(objs, **kwargs)

    setup = []
    subs = {}
    efuncs = {}

    for iters, (injectsolve,) in injectsolve_mapper.items():

        builder = Builder(injectsolve, objs, iters, **kwargs)

        setup.extend(builder.solversetup.calls)

        # Transform the spatial iteration loop with the calls to execute the solver
        subs.update(builder.solve.mapper)

        efuncs.update(builder.cbbuilder.efuncs)

    populate_matrix_context(efuncs, objs)

    iet = Transformer(subs).visit(iet)

    init = init_petsc(objs, **kwargs)
    body = core + tuple(setup) + (BlankLine,) + iet.body.body
    body = iet.body._rebuild(
        init=init, body=body,
        frees=(petsc_call('PetscFinalize', []),)
    )
    iet = iet._rebuild(body=body)
    metadata = core_metadata()
    metadata.update({'efuncs': tuple(efuncs.values())})
    return iet, metadata


def init_petsc(objs, **kwargs):
    # Initialize PETSc -> for now, assuming all solver options have to be
    # specified via the parameters dict in PETScSolve
    # TODO: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options
    Null = objs['Null']
    initialize = petsc_call('PetscInitialize', [Null, Null, Null, Null])

    return objs['begin_user'], initialize


def make_core_petsc_calls(objs, **kwargs):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [objs['comm'], Byref(objs['size'])])

    return call_mpi, BlankLine


<<<<<<< HEAD
=======
def build_core_objects(grid, **kwargs):
    """
    Returns a dict containing shared symbols and objects that are not
    unique to each PETScSolve.

    Many of these objects are used as arguments in callback functions to make
    the C code cleaner and more modular. This is also a step toward leveraging
    Devito's `reuse_efuncs` functionality, allowing reuse of efuncs when
    they are semantically identical.

    TODO: Further refinement is needed to make use of `reuse_efuncs`. Also,
    add docs for the objects inside the dict.
    """
    if kwargs['options']['mpi']:
        # TODO: Devito MPI + PETSc testing -> communicator = grid.distributor._obj_comm
        # communicator = 'PETSC_COMM_WORLD'
        communicator = grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    subdms = PointerDM(name='subdms')
    fields = PointerIS(name='fields')
    submats = PointerMat(name='submats')
    rows = PointerIS(name='rows')
    cols = PointerIS(name='cols')

    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': grid,
        'block': CallbackMat('block'),
        'submat_arr': PointerMat(name='submat_arr'),
        'subblockrows': PetscInt('subblockrows'),
        'subblockcols': PetscInt('subblockcols'),
        'rowidx': PetscInt('rowidx'),
        'colidx': PetscInt('colidx'),
        'J': Mat('J'),
        'X': GlobalVec('X'),
        'xloc': LocalVec('xloc'),
        'Y': GlobalVec('Y'),
        'yloc': LocalVec('yloc'),
        'F': GlobalVec('F'),
        'floc': LocalVec('floc'),
        'B': GlobalVec('B'),
        'nfields': PetscInt('nfields'),
        'irow': PointerIS(name='irow'),
        'icol': PointerIS(name='icol'),
        'nsubmats': Scalar('nsubmats', dtype=np.int32),
        'matreuse': MatReuse('scall'),
        'snes': SNES('snes'),
        'rows': rows,
        'cols': cols,
        'Subdms': subdms,
        'LocalSubdms': CallbackPointerDM(name='subdms'),
        'Fields': fields,
        'LocalFields': CallbackPointerIS(name='fields'),
        'Submats': submats,
        'ljacctx': JacobianStruct(
            fields=[subdms, fields, submats], modifier=' *'
        ),
        'subctx': SubMatrixStruct(fields=[rows, cols]),
        'Null': Macro('NULL'),
        'dummyctx': Symbol('lctx'),
        'dummyptr': DummyArg('dummy'),
        'dummyefunc': Symbol('dummyefunc'),
        'dof': PetscInt('dof'),
        'begin_user': c.Line('PetscFunctionBeginUser;'),
    }


>>>>>>> f08c644c4 (fix pytest)
class Builder:
    """
    This class is designed to support future extensions, enabling
    different combinations of solver types, preconditioning methods,
    and other functionalities as needed.
    The class will be extended to accommodate different solver types by
    returning subclasses of the objects initialised in __init__,
    depending on the properties of `injectsolve`.
    """
    def __init__(self, injectsolve, objs, iters, **kwargs):
        self.injectsolve = injectsolve
        self.objs = objs
        self.iters = iters
        self.kwargs = kwargs

        self.coupled = isinstance(injectsolve.expr.rhs.fielddata, MultipleFieldData)
        self.args = {
            'injectsolve': self.injectsolve,
            'objs': self.objs,
            'iters': self.iters,
            **self.kwargs
        }

        self.objbuilder = self._object_builder()
        self.args['solver_objs'] = self.objbuilder.solver_objs

        self.timedep = self._time_dependency()
        self.args['timedep'] = self.timedep

        self.cbbuilder = self._callback_builder()
        self.args['cbbuilder'] = self.cbbuilder

        self.solversetup = self._setup()
        self.solve = self._solver_execution()

    def _object_builder(self):
        return (
            CoupledObjectBuilder(**self.args)
            if self.coupled else
            BaseObjectBuilder(**self.args)
        )

    def _time_dependency(self):
        time_mapper = self.injectsolve.expr.rhs.time_mapper
        timedep_class = TimeDependent if time_mapper else NonTimeDependent
        return timedep_class(**self.args)

    def _callback_builder(self):
        return (CCBBuilder(**self.args) if self.coupled else CBBuilder(**self.args))

    def _setup(self):
        return (CoupledSetup(**self.args) if self.coupled else BaseSetup(**self.args))

    def _solver_execution(self):
        return (CoupledSolver(**self.args) if self.coupled else Solver(**self.args))


def populate_matrix_context(efuncs, objs):
    name = 'PopulateMatContext'

    if name not in efuncs:
        return

    subdms_expr = DummyExpr(
        FieldFromPointer(objs['Subdms']._C_symbol, objs['ljacctx']),
        objs['Subdms']._C_symbol
    )
    fields_expr = DummyExpr(
        FieldFromPointer(objs['Fields']._C_symbol, objs['ljacctx']),
        objs['Fields']._C_symbol
    )
    body = CallableBody(
        List(body=[subdms_expr, fields_expr]),
        init=(objs['begin_user'],),
        retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
    )
    efuncs[name] = Callable(
        name, body, objs['err'],
        parameters=[objs['ljacctx'], objs['Subdms'], objs['Fields']]
    )


# TODO: Devito MPI + PETSc testing
# if kwargs['options']['mpi'] -> communicator = grid.distributor._obj_comm
communicator = 'PETSC_COMM_WORLD'
subdms = PointerDM(name='subdms')
fields = PointerIS(name='fields')
submats = PointerMat(name='submats')
rows = PointerIS(name='rows')
cols = PointerIS(name='cols')


# A static dict containing shared symbols and objects that are not
# unique to each PETScSolve.
# Many of these objects are used as arguments in callback functions to make
# the C code cleaner and more modular. This is also a step toward leveraging
# Devito's `reuse_efuncs` functionality, allowing reuse of efuncs when
# they are semantically identical.
objs = {
    'size': PetscMPIInt(name='size'),
    'comm': communicator,
    'err': PetscErrorCode(name='err'),
    'block': CallbackMat('block'),
    'submat_arr': PointerMat(name='submat_arr'),
    'subblockrows': PetscInt('subblockrows'),
    'subblockcols': PetscInt('subblockcols'),
    'rowidx': PetscInt('rowidx'),
    'colidx': PetscInt('colidx'),
    'J': Mat('J'),
    'X': GlobalVec('X'),
    'xloc': LocalVec('xloc'),
    'Y': GlobalVec('Y'),
    'yloc': LocalVec('yloc'),
    'F': GlobalVec('F'),
    'floc': LocalVec('floc'),
    'B': GlobalVec('B'),
    'nfields': PetscInt('nfields'),
    'irow': PointerIS(name='irow'),
    'icol': PointerIS(name='icol'),
    'nsubmats': Scalar('nsubmats', dtype=np.int32),
    'matreuse': MatReuse('scall'),
    'snes': SNES('snes'),
    'rows': rows,
    'cols': cols,
    'Subdms': subdms,
    'LocalSubdms': CallbackPointerDM(name='subdms'),
    'Fields': fields,
    'LocalFields': CallbackPointerIS(name='fields'),
    'Submats': submats,
    'ljacctx': JacobianStruct(
        fields=[subdms, fields, submats], modifier=' *'
    ),
    'subctx': SubMatrixStruct(fields=[rows, cols]),
    'Null': Macro('NULL'),
    'dummyctx': Symbol('lctx'),
    'dummyptr': DummyArg('dummy'),
    'dummyefunc': Symbol('dummyefunc'),
    'dof': PetscInt('dof'),
    'begin_user': c.Line('PetscFunctionBeginUser;'),
}
