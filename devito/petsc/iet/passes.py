import cgen as c
import numpy as np

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, BlankLine,
                           DummyExpr, CallableBody, List, Call, Callable)
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.types import Symbol, Scalar
from devito.petsc.types import (PetscMPIInt, PetscErrorCode, MultipleFieldData,
                                IS, PETScStruct, CallbackDM, Mat, LocalVec, GlobalVec,
                                LocalMat, SNES, DummyArg, PetscInt, SubDM, SubMats,
                                MatReuse, LocalIS, LocalSubDMs)
from devito.petsc.iet.nodes import InjectSolveDummy
from devito.petsc.utils import core_metadata
from devito.petsc.iet.routines import (CBBuilder, CCBBuilder, BaseObjectBuilder,
                                       CoupledObjectBuilder, BaseSetup, CoupledSetup,
                                       Solver, CoupledSolver, TimeDependent,
                                       NonTimeDependent)
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi, petsc_struct


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
    objs = build_core_objects(grid, **kwargs)

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

    return objs['petsc_func_begin_user'], initialize


def make_core_petsc_calls(objs, **kwargs):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [objs['comm'], Byref(objs['size'])])

    return call_mpi, BlankLine


def build_core_objects(grid, **kwargs):
    """
    Returns a dict containing shared symbols and objects that are not
    unique to each PETScSolve.

    Many of these objects are used as arguments in callback functions to make
    the C code cleaner and more modular. This is also a step toward leveraging
    Devito's `reuse_efuncs` functionality, allowing reuse of efuncs when
    they are semantically identical.

    TODO: Further refinement is needed to make use of `reuse_efuncs`.
    """
    if kwargs['options']['mpi']:
        communicator = grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    subdms = SubDM(name='subdms')
    fields = IS(name='fields')
    submats = SubMats(name='submats')

    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': grid,

        'Null': Macro('NULL'),
        'dummyctx': Symbol('lctx'),
        'dummyptr': DummyArg('dummy'),
        'dummyefunc': Symbol('dummyefunc'),
        'dof': PetscInt('dof'),

        # Matrices & Vectors
        'block': LocalMat('block'),
        'submat_arr': SubMats(name='submat_arr'),
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

        # Callback & Contexts
        'cbdm': CallbackDM('dm', liveness='eager'),
        'nfields': PetscInt('nfields'),

        # Index Sets (IS)
        'irow': IS(name='irow', nindices=1),
        'icol': IS(name='icol', nindices=1),
        'nsubmats': Scalar('nsubmats', dtype=np.int32),
        'matreuse': MatReuse('scall'),

        # SNES Solver
        'snes': SNES('snes'),

        # SubMatrixCtx struct members
        'rows': IS(name='rows', nindices=1),
        'cols': IS(name='cols', nindices=1),

        # JacMatrixCtx struct members
        'Subdms': subdms,
        'LocalSubdms': LocalSubDMs(name='subdms', nindices=1),
        'Fields': fields,
        'LocalFields': LocalIS(name='fields', nindices=1),
        'Submats': submats,

        # Jacobian Context
        'ljacctx': petsc_struct(
            name='jctx',
            pname='JacobianCtx',
            fields=[subdms, fields, submats],
            liveness='lazy',
            modifier=' *'
        ),

        'jctx': PETScStruct(
            name='jctx', pname='JacobianCtx',
            fields=[subdms, fields], liveness='lazy'
        ),

        # PETSc Function Begin
        'petsc_func_begin_user': c.Line('PetscFunctionBeginUser;'),
    }


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

        # Determine the time dependency class
        time_mapper = injectsolve.expr.rhs.time_mapper
        timedep = TimeDependent if time_mapper else NonTimeDependent
        self.timedep = timedep(injectsolve, iters, **kwargs)

        # TODO: obvs improve this
        if isinstance(injectsolve.expr.rhs.fielddata, MultipleFieldData):
            coupled = True
        else:
            coupled = False

        # Objects
        if coupled:
            self.objbuilder = CoupledObjectBuilder(injectsolve, objs, **kwargs)
        else:
            self.objbuilder = BaseObjectBuilder(injectsolve, objs, **kwargs)
        self.solver_objs = self.objbuilder.solver_objs

        # Callbacks
        if coupled:
            self.cbbuilder = CCBBuilder(
                injectsolve, objs, self.solver_objs, timedep=self.timedep,
                **kwargs
            )
        else:
            self.cbbuilder = CBBuilder(
                injectsolve, objs, self.solver_objs, timedep=self.timedep,
                **kwargs
            )

        if coupled:
            # Solver setup
            self.solversetup = CoupledSetup(
                self.solver_objs, objs, injectsolve, self.cbbuilder
            )
        else:
            self.solversetup = BaseSetup(
                self.solver_objs, objs, injectsolve, self.cbbuilder
            )

        # NOTE: might not acc need a separate coupled class for this->rethink
        # just addding one for the purposes of debugging and figuring
        # out the coupled abstraction
        if coupled:
            # Execute the solver
            self.solve = CoupledSolver(
                self.solver_objs, objs, injectsolve, iters,
                self.cbbuilder, timedep=self.timedep
            )
        else:
            # Execute the solver
            self.solve = Solver(
                self.solver_objs, objs, injectsolve, iters,
                self.cbbuilder, timedep=self.timedep
            )


def populate_matrix_context(efuncs, objs):
    name = 'PopulateMatContext'

    try:
        efuncs[name]
    except KeyError:
        return

    subdms_expr = DummyExpr(
        FieldFromPointer(objs['Subdms']._C_symbol, objs['jctx']), objs['Subdms']._C_symbol
    )
    fields_expr = DummyExpr(
        FieldFromPointer(objs['Fields']._C_symbol, objs['jctx']), objs['Fields']._C_symbol
    )
    body = CallableBody(
        List(body=[subdms_expr, fields_expr]),
        init=(objs['petsc_func_begin_user'],),
        retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
    )
    efuncs[name] = Callable(
        name, body, objs['err'],
        parameters=[objs['jctx'], objs['Subdms'], objs['Fields']]
    )
