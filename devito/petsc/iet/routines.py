from collections import OrderedDict

import cgen as c
import numpy as np

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine, Callable, FindNodes,
                           retrieve_iteration_tree, filter_iterations, Iteration,
                           PointerCast, Definition)
from devito.symbolics import (Byref, FieldFromPointer, Macro, cast_mapper, VOIDP, Cast,
                              FieldFromComposite, IntDiv, Modulo, Deref, IndexedPointer)
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import Temp, Symbol, CustomDimension, Dimension, Scalar
from devito.tools import filter_ordered

from devito.petsc.types import PETScArray, PETScStruct
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatShellSetOp, InjectSolveDummy)
from devito.petsc.iet.utils import petsc_call, petsc_struct
from devito.petsc.utils import solver_mapper
from devito.petsc.types import (DM, CallbackDM, Mat, LocalVec, GlobalVec, KSP, PC, LocalMat,
                                SNES, DummyArg, PetscInt, StartPtr, SingleIS, IS, SubDM, SubMats,
                                MatReuse, VecScatter, DMCast)


class CBBuilder:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __init__(self, injectsolve, objs, solver_objs,
                 rcompile=None, sregistry=None, timedep=None, **kwargs):

        self.rcompile = rcompile
        self.sregistry = sregistry
        self.timedep = timedep
        self.objs = objs
        self.solver_objs = solver_objs
        self.injectsolve = injectsolve

        self._efuncs = OrderedDict()
        self._struct_params = []

        self._main_matvec_callback = None
        self._main_formfunc_callback = None
        self._user_struct_callback = None
        self._matvecs = []
        self._formfuncs = []
        self._formrhss = []

        self._make_core()
        self._make_user_struct_callback()
        self._efuncs = self._uxreplace_efuncs()

    @property
    def efuncs(self):
        return self._efuncs

    @property
    def struct_params(self):
        return self._struct_params

    @property
    def filtered_struct_params(self):
        return filter_ordered(self.struct_params)

    @property
    def main_matvec_callback(self):
        """
        This is the matvec callback associated with the whole Jacobian i.e
        is set in the main kernel via
        `PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MyMatShellMult));`
        """
        return self._matvecs.pop()

    @property
    def main_formfunc_callback(self):
        return self._formfuncs.pop()

    @property
    def matvecs(self):
        return self._matvecs

    @property
    def formfuncs(self):
        return self._formfuncs

    @property
    def formrhss(self):
        return self._formrhss

    @property
    def user_struct_callback(self):
        return self._user_struct_callback

    def _make_core(self):
        fielddata = self.injectsolve.expr.rhs.fielddata
        self._make_matvec(fielddata)
        self._make_formfunc(fielddata)
        self._make_formrhs(fielddata)

    def _make_matvec(self, fielddata):
        # Compile matvec `eqns` into an IET via recursive compilation
        matvecs = fielddata.matvecs
        sobjs = self.solver_objs
        irs_matvec, _ = self.rcompile(matvecs,
                                      options={'mpi': False}, sregistry=self.sregistry)
        body_matvec = self._create_matvec_body(List(body=irs_matvec.uiet.body),
                                               fielddata)

        matvec_callback = PETScCallable(
            self.sregistry.make_name(prefix='MyMatShellMult'), body_matvec,
            retval=self.objs['err'],
            parameters=(
                sobjs['Jac'], sobjs['X_global'], sobjs['Y_global']
            )
        )
        self._matvecs.append(matvec_callback)
        self._efuncs[matvec_callback.name] = matvec_callback

    def _create_matvec_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)

        # TODO: maybe this shouldn't be attached to the fielddata -> think about this
        # currently it's attched to both i think
        y_matvec = fielddata.arrays['y_matvec']
        x_matvec = fielddata.arrays['x_matvec']

        mat_get_dm = petsc_call('MatGetDM', [sobjs['Jac'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, sobjs['X_global'],
                                     'INSERT_VALUES', sobjs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, sobjs['X_global'], 'INSERT_VALUES', sobjs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['Y_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [sobjs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [sobjs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [sobjs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [sobjs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, sobjs['Y_local'], 'INSERT_VALUES', sobjs['Y_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['Y_local'], 'INSERT_VALUES', sobjs['Y_global']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['Y_local'])]
        )

        # TODO: Some of the calls are placed in the `stacks` argument of the
        # `CallableBody` to ensure that they precede the `cast` statements. The
        # 'casts' depend on the calls, so this order is necessary. By doing this,
        # you avoid having to manually construct the `casts` and can allow
        # Devito to handle their construction. This is a temporary solution and
        # should be revisited

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end,
             dm_restore_local_xvec,
             dm_restore_local_yvec)
        )

        stacks = (
            mat_get_dm,
            dm_get_app_context,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for i in fields}
        matvec_body = Uxreplace(subs).visit(matvec_body)

        self._struct_params.extend(fields)
        return matvec_body

    def _make_formfunc(self, fielddata):
        formfuncs = fielddata.formfuncs
        sobjs = self.solver_objs
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(
            formfuncs,
            options={'mpi': False}, sregistry=self.sregistry
        )
        body_formfunc = self._create_formfunc_body(
            List(body=irs_formfunc.uiet.body), fielddata
        )
        cb = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction'), body_formfunc,
            retval=self.objs['err'],
            parameters=(sobjs['snes'], sobjs['X_global'],
                        sobjs['F_global'], dummyptr)
        )
        self._formfuncs.append(cb)
        self._efuncs[cb.name] = cb

    def _create_formfunc_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)

        f_formfunc = fielddata.arrays['f_formfunc']
        x_formfunc = fielddata.arrays['x_formfunc']

        dm_cast = DummyExpr(dmda, DMCast(dummyptr), init=True)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, sobjs['X_global'],
                                     'INSERT_VALUES', sobjs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, sobjs['X_global'], 'INSERT_VALUES', sobjs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['F_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [sobjs['F_local'], Byref(f_formfunc._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [sobjs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [sobjs['F_local'], Byref(f_formfunc._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [sobjs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, sobjs['F_local'], 'INSERT_VALUES', sobjs['F_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['F_local'], 'INSERT_VALUES', sobjs['F_global']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['F_local'])]
        )

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end,
             dm_restore_local_xvec,
             dm_restore_local_yvec)
        )

        stacks = (
            dm_cast,
            dm_get_app_context,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for i in fields}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        self._struct_params.extend(fields)
        return formfunc_body

    def _make_formrhs(self, fielddata):
        formrhs = fielddata.formrhs
        sobjs = self.solver_objs

        # Compile formrhs `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(
            formrhs, options={'mpi': False}, sregistry=self.sregistry
        )
        body_formrhs = self._create_form_rhs_body(
            List(body=irs_formrhs.uiet.body), fielddata
        )

        cb = PETScCallable(
            self.sregistry.make_name(prefix='FormRHS'), body_formrhs, retval=self.objs['err'],
            parameters=(
                sobjs['callbackdm'], sobjs['b_global'],
            )
        )
        self._formrhss.append(cb)
        self._efuncs[cb.name] = cb

    def _create_form_rhs_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        sobjs = self.solver_objs
        target = fielddata.target

        dmda = sobjs['callbackdm']
        # TODO: when moving to coupled...perhaps the DMDA should be an argument to this function _create_form_rhs_body

        dm_get_local = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['b_local'])]
        )

        dm_global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, sobjs['b_global'],
                                     'INSERT_VALUES', sobjs['b_local']]
        )

        dm_global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, sobjs['b_global'], 'INSERT_VALUES',
            sobjs['b_local']
        ])

        b_arr = fielddata.arrays['b_tmp']

        vec_get_array = petsc_call(
            'VecGetArray', [sobjs['b_local'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, sobjs['b_local'], 'INSERT_VALUES',
            sobjs['b_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['b_local'], 'INSERT_VALUES',
            sobjs['b_global']
        ])

        vec_restore_array = petsc_call(
            'VecRestoreArray', [sobjs['b_local'], Byref(b_arr._C_symbol)]
        )

        body = body._rebuild(body=body.body + (
            dm_local_to_global_begin, dm_local_to_global_end, vec_restore_array
        ))

        stacks = (
            dm_get_local,
            dm_global_to_local_begin,
            dm_global_to_local_end,
            vec_get_array,
            dm_get_app_context,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for
                i in fields if not isinstance(i.function, AbstractFunction)}
        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        self._struct_params.extend(fields)
        return formrhs_body

    def _local_struct(self):
        """
        This is the struct used within callback functions,
        usually accessed via DMGetApplicationContext.
        """
        self.solver_objs['luserctx'] = petsc_struct(
            self.solver_objs['userctx'].name,
            self.filtered_struct_params,
            self.solver_objs['userctx'].pname,
            liveness='eager',
            modifier=' *'
        )

    def _make_user_struct_callback(self):
        """
        This is the struct initialised inside the main kernel and
        attached to the DM via DMSetApplicationContext.
        # TODO: this could be common between all PETScSolves instead? 
        """
        mainctx = self.solver_objs['userctx'] = petsc_struct(
            self.sregistry.make_name(prefix='ctx'),
            self.filtered_struct_params,
            self.sregistry.make_name(prefix='UserCtx'),
        )
        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, mainctx), i._C_symbol)
            for i in mainctx.callback_fields
        ]
        struct_callback_body = CallableBody(
            List(body=body), init=(petsc_func_begin_user,),
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
        )
        cb = Callable(
            self.sregistry.make_name(prefix='PopulateUserContext'),
            struct_callback_body, self.objs['err'],
            parameters=[mainctx]
        )
        self._efuncs[cb.name] = cb
        self._user_struct_callback = cb

    def _dummy_fields(self, iet):
        # Place all context data required by the shell routines into a struct
        fields = [f.function for f in FindSymbols('basics').visit(iet)]
        fields = [f for f in fields if not isinstance(f.function, (PETScArray, Temp))]
        fields = [
            f for f in fields if not (f.is_Dimension and not (f.is_Time or f.is_Modulo))
        ]
        return fields

    def _uxreplace_efuncs(self):
        self._local_struct()
        mapper = {}
        visitor = Uxreplace({dummyctx: self.solver_objs['luserctx']})
        for k, v in self._efuncs.items():
            mapper.update({k: visitor.visit(v)})
        return mapper


class CCBBuilder(CBBuilder):

    def __init__(self, injectsolve, objs, solver_objs, **kwargs):
        # TODO: probs move this after the super init?
        self._submatrices_callback = None
        super().__init__(injectsolve, objs, solver_objs, **kwargs)
        self._make_coupled_ctx()
        self._make_local_coupled_ctx()
        self._make_whole_matvec()
        self._make_whole_formfunc()
        self._create_submatrices()
        self._efuncs['PopulateMatContext'] = Symbol('dummy')

    @property
    def submatrices_callback(self):
        return self._submatrices_callback

    @property
    def main_matvec_callback(self):
        """
        This is the matvec callback associated with the whole Jacobian i.e
        is set in the main kernel via
        `PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MyMatShellMult));`
        """
        return self._main_matvec_callback

    @property
    def main_formfunc_callback(self):
        return self._main_formfunc_callback

    def _make_core(self):
        # let's just start by generating the diagonal sub matrices, then will extend to off diags
        injectsolve = self.injectsolve
        targets = injectsolve.expr.rhs.fielddata.targets
        all_fielddata = injectsolve.expr.rhs.fielddata 

        for t in targets:
            data = all_fielddata.get_field_data(t)
            self._make_matvec(data)
            self._make_formfunc(data)
            self._make_formrhs(data)

    def _make_whole_matvec(self):
        sobjs = self.solver_objs

        # obvs improve name
        body = self._create_whole_matvec_callback_body()

        cb = PETScCallable(
            self.sregistry.make_name(prefix='WholeMatMult'), List(body=body),
            retval=self.objs['err'],
            parameters=(
                sobjs['Jac'], sobjs['X_global'], sobjs['Y_global']
            )
        )
        self._main_matvec_callback = cb
        self._efuncs[cb.name] = cb

    def _create_whole_matvec_callback_body(self):
        sobjs = self.solver_objs

        mat_get_ctx_main = petsc_call('MatShellGetContext', [sobjs['Jac'], Byref(sobjs['ljacctx'])])

        # J00
        deref_j00 = DummyExpr(sobjs['J00'], FieldFromPointer(sobjs['submats'].indexed[0], sobjs['ljacctx']))
        mat_get_ctx_j00 = petsc_call('MatShellGetContext', [sobjs['J00'], Byref(sobjs['j00ctx'])])
        vec_get_x_j00 = petsc_call('VecGetSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j00ctx'])), Byref(sobjs['j00X'])])
        vec_get_y_j00 = petsc_call('VecGetSubVector', [sobjs['Y_global'], Deref(FieldFromPointer(rows.base, sobjs['j00ctx'])), Byref(sobjs['j00Y'])])
        mat_mult_j00 = petsc_call('MatMult', [sobjs['J00'], sobjs['j00X'], sobjs['j00Y']])
        vec_restore_x_j00 = petsc_call('VecRestoreSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j00ctx'])), Byref(sobjs['j00X'])])
        vec_restore_y_j00 = petsc_call('VecRestoreSubVector', [sobjs['Y_global'], Deref(FieldFromPointer(rows.base, sobjs['j00ctx'])), Byref(sobjs['j00Y'])])

        # J11
        deref_j11 = DummyExpr(sobjs['J11'], FieldFromPointer(sobjs['submats'].indexed[3], sobjs['ljacctx']))
        mat_get_ctx_j11 = petsc_call('MatShellGetContext', [sobjs['J11'], Byref(sobjs['j11ctx'])])
        vec_get_x_j11 = petsc_call('VecGetSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j11ctx'])), Byref(sobjs['j11X'])])
        vec_get_y_j11 = petsc_call('VecGetSubVector', [sobjs['Y_global'], Deref(FieldFromPointer(rows.base, sobjs['j11ctx'])), Byref(sobjs['j11Y'])])
        mat_mult_j11 = petsc_call('MatMult', [sobjs['J11'], sobjs['j11X'], sobjs['j11Y']])
        vec_restore_x_j11 = petsc_call('VecRestoreSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j11ctx'])), Byref(sobjs['j11X'])])
        vec_restore_y_j11 = petsc_call('VecRestoreSubVector', [sobjs['Y_global'], Deref(FieldFromPointer(rows.base, sobjs['j11ctx'])), Byref(sobjs['j11Y'])])

        body = [mat_get_ctx_main, BlankLine, deref_j00, mat_get_ctx_j00, vec_get_x_j00, vec_get_y_j00, mat_mult_j00, vec_restore_x_j00, vec_restore_y_j00,  BlankLine, deref_j11, mat_get_ctx_j11, vec_get_x_j11, vec_get_y_j11, mat_mult_j11, vec_restore_x_j11, vec_restore_y_j11]
        body = CallableBody(
            List(body=tuple(body)),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))
        return body

    def _make_whole_formfunc(self):
        sobjs = self.solver_objs

        # obvs improve name
        body = self._create_whole_formfunc_callback_body()

        main_formfunc_callback = PETScCallable(
            self.sregistry.make_name(prefix='WholeFormFunc'), List(body=body),
            retval=self.objs['err'],
            parameters=(sobjs['snes'], sobjs['X_global'],
                        sobjs['F_global'], dummyptr)
        )
        self._main_formfunc_callback = main_formfunc_callback
        # from IPython import embed; embed()
        self._efuncs[main_formfunc_callback.name] = main_formfunc_callback

    def _create_whole_formfunc_callback_body(self):
        sobjs = self.solver_objs

        tmp = c.Line("struct JacobianCtx * jctx0 = (struct JacobianCtx *)dummy;")

        # J00
        deref_j00 = DummyExpr(sobjs['J00'], FieldFromPointer(sobjs['submats'].indexed[0], sobjs['jacctx']))
        mat_get_dm_j00 = petsc_call('MatGetDM', [sobjs['J00'], Byref(sobjs['dmpn1'])])
        mat_get_ctx_j00 = petsc_call('MatShellGetContext', [sobjs['J00'], Byref(sobjs['j00ctx'])])
        vec_get_x_j00 = petsc_call('VecGetSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j00ctx'])), Byref(sobjs['j00X'])])
        vec_get_y_j00 = petsc_call('VecGetSubVector', [sobjs['F_global'], Deref(FieldFromPointer(rows.base, sobjs['j00ctx'])), Byref(sobjs['j00F'])])
        call_first_formfunc = petsc_call(self.formfuncs[0].name, [sobjs['snes'], sobjs['j00X'], sobjs['j00F'], VOIDP(sobjs['dmpn1'])])
        vec_restore_x_j00 = petsc_call('VecRestoreSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j00ctx'])), Byref(sobjs['j00X'])])
        vec_restore_y_j00 = petsc_call('VecRestoreSubVector', [sobjs['F_global'], Deref(FieldFromPointer(rows.base, sobjs['j00ctx'])), Byref(sobjs['j00F'])])

        # J11
        deref_j11 = DummyExpr(sobjs['J11'], FieldFromPointer(sobjs['submats'].indexed[3], sobjs['jacctx']))
        mat_get_dm_j11 = petsc_call('MatGetDM', [sobjs['J11'], Byref(sobjs['dmpn2'])])
        mat_get_ctx_j11 = petsc_call('MatShellGetContext', [sobjs['J11'], Byref(sobjs['j11ctx'])])
        vec_get_x_j11 = petsc_call('VecGetSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j11ctx'])), Byref(sobjs['j11X'])])
        vec_get_y_j11 = petsc_call('VecGetSubVector', [sobjs['F_global'], Deref(FieldFromPointer(rows.base, sobjs['j11ctx'])), Byref(sobjs['j11F'])])
        call_second_formfunc = petsc_call(self.formfuncs[1].name, [sobjs['snes'], sobjs['j11X'], sobjs['j11F'], VOIDP(sobjs['dmpn2'])])
        vec_restore_x_j11 = petsc_call('VecRestoreSubVector', [sobjs['X_global'], Deref(FieldFromPointer(cols.base, sobjs['j11ctx'])), Byref(sobjs['j11X'])])
        vec_restore_y_j11 = petsc_call('VecRestoreSubVector', [sobjs['F_global'], Deref(FieldFromPointer(rows.base, sobjs['j11ctx'])), Byref(sobjs['j11F'])])

        body = [tmp, BlankLine, deref_j00, mat_get_dm_j00, mat_get_ctx_j00, vec_get_x_j00, vec_get_y_j00, call_first_formfunc, vec_restore_x_j00, vec_restore_y_j00,  BlankLine, deref_j11, mat_get_dm_j11, mat_get_ctx_j11, vec_get_x_j11, vec_get_y_j11, call_second_formfunc, vec_restore_x_j11, vec_restore_y_j11]

        body = CallableBody(
            List(body=tuple(body)),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))
        return body

    def _create_submatrices(self):
        body = self._create_submat_callback_body()
        sobjs = self.solver_objs

        params = (
            sobjs['Jac'],
            sobjs['n_fields'],
            sobjs['all_IS_rows'],
            sobjs['all_IS_cols'],
            sobjs['matreuse'],
            sobjs['submats']
        )

        cb = PETScCallable(
            self.sregistry.make_name(prefix='MatCreateSubMatrices'), body,
            retval=self.objs['err'],
            parameters=params
        )
        self._submatrices_callback = cb
        self._efuncs[cb.name] = cb

    # TODO: obvs improve these names
    def _create_submat_callback_body(self):
        sobjs = self.solver_objs

        n_submats = DummyExpr(sobjs['n_submats'], Mul(sobjs['n_fields'], sobjs['n_fields']))

        malloc_submats = petsc_call('PetscCalloc1', [sobjs['n_submats'], sobjs['submats']])

        mat_get_dm = petsc_call('MatGetDM', [sobjs['Jac'], Byref(sobjs['callbackdm'])])

        dm_get_app = petsc_call('DMGetApplicationContext', [sobjs['callbackdm'], Byref(sobjs['luserctx'])])

        shell_get_ctx = petsc_call('MatShellGetContext', [sobjs['Jac'], Byref(sobjs['ljacctx'])])

        dm_get_info = petsc_call('DMDAGetInfo', [sobjs['callbackdm'], Null, Byref(sobjs['M']), Byref(sobjs['N']), Null, Null, Null, Null, Byref(sobjs['dof']), Null, Null, Null, Null, Null])

        subblock_rows = DummyExpr(sobjs['subblockrows'], Mul(sobjs['M'], sobjs['N']))
        subblock_cols = DummyExpr(sobjs['subblockcols'], Mul(sobjs['M'], sobjs['N']))

        ptr = DummyExpr(sobjs['submat_arr']._C_symbol, Deref(sobjs['submats']), init=True)

        mat_create = petsc_call('MatCreate', [self.objs['comm'], Byref(sobjs['block'])])
        mat_set_sizes = petsc_call('MatSetSizes', [sobjs['block'], 'PETSC_DECIDE', 'PETSC_DECIDE', sobjs['subblockrows'], sobjs['subblockcols']])
        mat_set_type = petsc_call('MatSetType', [sobjs['block'], 'MATSHELL'])

        malloc = petsc_call('PetscMalloc1', [1, Byref(sobjs['submatctx'])])
        i = Dimension(name='i')

        row_idx = DummyExpr(sobjs['row_idx'], IntDiv(i, sobjs['dof']))
        col_idx = DummyExpr(sobjs['col_idx'], Modulo(i, sobjs['dof']))

        deref_subdm = Dereference(Subdms, sobjs['ljacctx'])

        # fix:todo: the SUBMAT_CTX doesn't appear in the ccode because it's not an argument to any function -> fix this in the cgen structure code
        set_rows = DummyExpr(FieldFromPointer(rows.base, sobjs['submatctx']), Byref(sobjs['all_IS_rows'].indexed[sobjs['row_idx']]))
        set_cols = DummyExpr(FieldFromPointer(cols.base, sobjs['submatctx']), Byref(sobjs['all_IS_cols'].indexed[sobjs['col_idx']]))

        dm_set_app_ctx = petsc_call('DMSetApplicationContext', [Subdms.indexed[sobjs['row_idx']], sobjs['luserctx']])

        matset_dm = petsc_call('MatSetDM', [sobjs['block'], Subdms.indexed[sobjs['row_idx']]])

        set_ctx = petsc_call('MatShellSetContext', [sobjs['block'], sobjs['submatctx']])

        mat_setup = petsc_call('MatSetUp', [sobjs['block']])

        assign_block = DummyExpr(sobjs['submat_arr'].indexed[i], sobjs['block'])

        iteration = Iteration(List(body=[mat_create, mat_set_sizes, mat_set_type, malloc, row_idx, col_idx, set_rows, set_cols, dm_set_app_ctx, matset_dm, set_ctx, mat_setup, assign_block]), i, 3)

        # J00
        j00_op = petsc_call(
            'MatShellSetOperation',
            [sobjs['submat_arr'].indexed[0], 'MATOP_MULT',
            MatShellSetOp(self.matvecs[0].name, void, void)]
        )

        # J11
        j11_op = petsc_call(
            'MatShellSetOperation',
            [sobjs['submat_arr'].indexed[3], 'MATOP_MULT',
            MatShellSetOp(self.matvecs[1].name, void, void)]
        )

        body = [
            n_submats,
            malloc_submats,
            mat_get_dm,
            dm_get_app,
            dm_get_info,
            subblock_rows,
            subblock_cols,
            ptr,
            BlankLine,
            iteration,
            j00_op,
            j11_op
        ]
        body = CallableBody(
            List(body=tuple(body)),
            init=(petsc_func_begin_user,),
            stacks=(shell_get_ctx, deref_subdm),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))
        return body

    def _make_coupled_ctx(self):
        objs = self.solver_objs
        fields = [Subdms, Fields, objs['submats']]
        self.solver_objs['jacctx'] = petsc_struct(
            name=self.sregistry.make_name(prefix='jctx'), pname='JacobianCtx',
            fields=fields, liveness='lazy'
        )

    def _make_local_coupled_ctx(self):
        objs = self.solver_objs
        # TODO: can this struct just be combined with the user ctx? i.e I don't think i need two separate ones
        fields = objs['jacctx'].fields
        self.solver_objs['ljacctx'] = petsc_struct(
            name=objs['jacctx'].name, pname=objs['jacctx'].pname,
            fields=fields, liveness='lazy',
            modifier=' *'
        )


class BaseObjectBuilder:
    """
    A base class for constructing objects needed for a PETSc solver.
    Designed to be extended by subclasses, which can override the `_extend_build`
    method to support specific use cases.
    """

    def __init__(self, injectsolve, sregistry=None, **kwargs):
        self.sregistry = sregistry
        self.fielddata = injectsolve.expr.rhs.fielddata
        self.solver_objs = self._build(injectsolve)

    def _build(self, injectsolve):
        """
        Constructs the core dictionary of solver objects and allows
        subclasses to extend or modify it via `_extend_build`.
        Returns:
            dict: A dictionary containing the following objects:
                - 'Jac' (Mat): A matrix representing the jacobian.
                - 'x_global' (GlobalVec): The global solution vector.
                - 'x_local' (LocalVec): The local solution vector.
                - 'b_global': (GlobalVec) Global RHS vector `b`, where `F(x) = b`.
                - 'b_local': (LocalVec) Local RHS vector `b`, where `F(x) = b`.
                - 'ksp': (KSP) Krylov solver object that manages the linear solver.
                - 'pc': (PC) Preconditioner object.
                - 'snes': (SNES) Nonlinear solver object.
                - 'F_global': (GlobalVec) Global residual vector `F`, where `F(x) = b`.
                - 'F_local': (LocalVec) Local residual vector `F`, where `F(x) = b`.
                - 'Y_global': (GlobalVector) The output vector populated by the
                   matrix-free `MyMatShellMult` callback function.
                - 'Y_local': (LocalVector) The output vector populated by the matrix-free
                   `MyMatShellMult` callback function.
                - 'X_global': (GlobalVec) Current guess for the solution,
                   required by the FormFunction callback.
                - 'X_local': (LocalVec) Current guess for the solution,
                   required by the FormFunction callback.
                - 'localsize' (PetscInt): The local length of the solution vector.
                - 'start_ptr' (StartPtr): A pointer to the beginning of the solution array
                   that will be updated at each time step.
                - 'dmda' (DM): The DMDA object associated with this solve, linked to
                   the SNES object via `SNESSetDM`.
                - 'callbackdm' (CallbackDM): The DM object accessed within callback
                   functions via `SNESGetDM`.
        """
        sreg = self.sregistry
        targets = self.fielddata.targets
        # TODO: for any of the Vec objects used in callback funcs, I don't think need to use symbol registry for them..?
        base_dict = {
            'Jac': Mat(sreg.make_name(prefix='J')),
            'x_global': GlobalVec(sreg.make_name(prefix='xglobal')),
            'x_local': LocalVec(sreg.make_name(prefix='xlocal'), liveness='eager'),
            'b_global': GlobalVec(sreg.make_name(prefix='bglobal')),
            'b_local': LocalVec(sreg.make_name(prefix='blocal')),
            'ksp': KSP(sreg.make_name(prefix='ksp')),
            'pc': PC(sreg.make_name(prefix='pc')),
            'snes': SNES(sreg.make_name(prefix='snes')),
            'F_global': GlobalVec(sreg.make_name(prefix='Fglobal')),
            'F_local': LocalVec(sreg.make_name(prefix='Flocal'), liveness='eager'),
            'Y_global': GlobalVec(sreg.make_name(prefix='Yglobal')),
            'Y_local': LocalVec(sreg.make_name(prefix='Ylocal'), liveness='eager'),
            'X_global': GlobalVec(sreg.make_name(prefix='Xglobal')),
            'X_local': LocalVec(sreg.make_name(prefix='Xlocal'), liveness='eager'),
            'localsize': PetscInt(sreg.make_name(prefix='localsize')),
            'dmda': DM(sreg.make_name(prefix='da'), liveness='eager',
                       stencil_width=self.fielddata.space_order, dofs=len(targets)),
            'callbackdm': CallbackDM(
                sreg.make_name(prefix='dm'), liveness='eager',
                stencil_width=self.fielddata.space_order
            ),
        }
        base_dict = self._target_dependent(base_dict)
        return self._extend_build(base_dict, injectsolve)

    def _target_dependent(self, base_dict):
        sreg = self.sregistry
        targets = self.fielddata.targets
        for target in targets:
            base_dict[target.name+'_ptr'] = StartPtr(
                sreg.make_name(prefix='%s_ptr' % target.name), target.dtype
            )
        base_dict = self._extend_target_dependent(base_dict)
        return base_dict

    def _extend_build(self, base_dict, injectsolve):
        """
        Subclasses can override this method to extend or modify the
        base dictionary of solver objects.
        """
        return base_dict

    def _extend_target_dependent(self, base_dict):
        """
        Subclasses can override this method to extend or modify the
        base dictionary of target-dependent solver objects.
        """
        return base_dict


class CoupledObjectBuilder(BaseObjectBuilder):
    def _extend_build(self, base_dict, injectsolve):
        sreg = self.sregistry
        # TODO: add a no_of_targets attribute to the FieldData object
        no_targets = len(self.fielddata.targets)

        base_dict['fields'] = IS(
            name=sreg.make_name(prefix='fields'), nindices=no_targets
            )
        base_dict['subdms'] = SubDM(
            name=sreg.make_name(prefix='subdms'), nindices=no_targets
            )
        # CHANGE THIS TO PETSCINT
        base_dict['n_submats'] = Scalar(sreg.make_name(prefix='nsubmats'), dtype=np.int32)

        base_dict['submat_arr'] = SubMats(name=sreg.make_name(prefix='submat_arr'),
                                          nindices=no_targets*no_targets)

        base_dict['submats'] = SubMats(name=sreg.make_name(prefix='submats'),
                                       nindices=no_targets*no_targets)

        base_dict['n_fields'] = PetscInt(sreg.make_name(prefix='nfields'))

        # global submatrix sizes
        base_dict['M'] = PetscInt(sreg.make_name(prefix='M'))
        base_dict['N'] = PetscInt(sreg.make_name(prefix='N'))
        # from IPython import embed; embed()
        base_dict['dof'] = PetscInt(sreg.make_name(prefix='dof'))
        base_dict['block'] = LocalMat(sreg.make_name(prefix='block'))
        base_dict['subblockrows'] = PetscInt(sreg.make_name(prefix='subblockrows'))
        base_dict['subblockcols'] = PetscInt(sreg.make_name(prefix='subblockcols'))

        # these don't need to be used -> think can just use fields?
        base_dict['all_IS_rows'] = IS(name=sreg.make_name(prefix='allrows'), nindices=1)
        base_dict['all_IS_cols'] = IS(name=sreg.make_name(prefix='allcols'), nindices=1)

        base_dict['J00'] = Mat(name='J00')
        base_dict['J11'] = Mat(name='J11')

        # probably can just use the existing 'callbackdm'
        base_dict['subdm'] = DM(sreg.make_name(prefix='subdm'), liveness='eager')

        pname = 'SubMatrixCtx'
        fields = [rows, cols]

        base_dict['submatctx'] = petsc_struct(
            name=sreg.make_name(prefix='submatctx'),
            pname=pname, fields=fields,
            modifier=' *', liveness='eager'
        )

        base_dict['j00ctx'] = petsc_struct(
            name='j00ctx', pname=pname,
            fields=fields, modifier=' *', liveness='eager'
        )

        base_dict['j11ctx'] = petsc_struct(
            name='j11ctx', pname=pname,
            fields=fields, modifier=' *', liveness='eager'
        )

        base_dict['row_idx'] = PetscInt(sreg.make_name(prefix='rowidx'))
        base_dict['col_idx'] = PetscInt(self.sregistry.make_name(prefix='colidx'))

        # not sure if it should be global or local yet
        base_dict['j00X'] = LocalVec(sreg.make_name(prefix='j00X'))
        base_dict['j00Y'] = LocalVec(sreg.make_name(prefix='j00Y'))
        base_dict['j00F'] = LocalVec(sreg.make_name(prefix='j00F'))

        base_dict['j11X'] = LocalVec(sreg.make_name(prefix='j11X'))
        base_dict['j11Y'] = LocalVec(sreg.make_name(prefix='j11Y'))
        base_dict['j11F'] = LocalVec(sreg.make_name(prefix='j11F'))

        base_dict['matreuse'] = MatReuse(sreg.make_name(prefix='scall'))

        # obvs rethink -> probs don't need?
        base_dict['dmpn1'] = CallbackDM(sreg.make_name(prefix='dmpn1'), liveness='eager')
        base_dict['dmpn2'] = CallbackDM(sreg.make_name(prefix='dmpn2'), liveness='eager')

        base_dict['scatterpn1'] = VecScatter(sreg.make_name(prefix='scatterpn1'))
        base_dict['scatterpn2'] = VecScatter(sreg.make_name(prefix='scatterpn2'))

        return base_dict

    def _extend_target_dependent(self, base_dict):
        sreg = self.sregistry
        targets = self.fielddata.targets
        for target in targets:
            base_dict['xlocal'+target.name] = LocalVec(
                sreg.make_name(prefix='xlocal%s' % target.name), liveness='eager'
            )
            # should defo be moved since this is only needed for coupled solves?
            base_dict['xglobal'+target.name] = GlobalVec(
                sreg.make_name(prefix='xglobal%s' % target.name)
            )
            base_dict['blocal'+target.name] = LocalVec(
                sreg.make_name(prefix='blocal%s' % target.name), liveness='eager'
            )
            base_dict['bglobal'+target.name] = GlobalVec(
                sreg.make_name(prefix='bglobal%s' % target.name)
            )
            # TODO: these aren't being destroyed? because they are set to subdms[] etc
            base_dict['da'+target.name] = DM(
                sreg.make_name(prefix='da%s' % target.name), liveness='eager',
                stencil_width=self.fielddata.space_order
            )
        return base_dict


class BaseSetup:
    def __init__(self, solver_objs, objs, injectsolve, cbbuilder):
        self.solver_objs = solver_objs
        self.objs = objs
        self.injectsolve = injectsolve
        self.cbbuilder = cbbuilder
        self.calls = self._setup()

    @property
    def snes_ctx(self):
        """
        The [optional] context for private data for the function evaluation routine.
        https://petsc.org/main/manualpages/SNES/SNESSetFunction/
        """
        return VOIDP(self.solver_objs['dmda'])

    def _setup(self):
        sobjs = self.solver_objs

        target = self.injectsolve.expr.rhs.fielddata.target

        dmda = sobjs['dmda']

        solver_params = self.injectsolve.expr.rhs.solver_parameters

        snes_create = petsc_call('SNESCreate', [self.objs['comm'], Byref(sobjs['snes'])])

        snes_set_dm = petsc_call('SNESSetDM', [sobjs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(sobjs['Jac'])])

        # NOTE: Assuming all solves are linear for now.
        snes_set_type = petsc_call('SNESSetType', [sobjs['snes'], 'SNESKSPONLY'])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [sobjs['snes'], sobjs['Jac'],
                                sobjs['Jac'], 'MatMFFDComputeJacobian', Null]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['x_global'])])

        global_b = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['b_global'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [sobjs['snes'], Byref(sobjs['ksp'])])

        ksp_set_tols = petsc_call(
            'KSPSetTolerances', [sobjs['ksp'], solver_params['ksp_rtol'],
                                 solver_params['ksp_atol'], solver_params['ksp_divtol'],
                                 solver_params['ksp_max_it']]
        )

        ksp_set_type = petsc_call(
            'KSPSetType', [sobjs['ksp'], solver_mapper[solver_params['ksp_type']]]
        )

        ksp_get_pc = petsc_call(
            'KSPGetPC', [sobjs['ksp'], Byref(sobjs['pc'])]
        )

        # Even though the default will be jacobi, set to PCNONE for now
        pc_set_type = petsc_call('PCSetType', [sobjs['pc'], 'PCNONE'])

        ksp_set_from_ops = petsc_call('KSPSetFromOptions', [sobjs['ksp']])

        matvec_operation = petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_MULT',
             MatShellSetOp(self.cbbuilder.main_matvec_callback.name, void, void)]
        )

        snes_ctx = self.snes_ctx

        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [sobjs['snes'], Null,
             FormFunctionCallback(self.cbbuilder.main_formfunc_callback.name, void, void), snes_ctx]
        )

        dmda_calls = self._create_dmda_calls(dmda)

        mainctx = sobjs['userctx']

        call_struct_callback = petsc_call(
            self.cbbuilder.user_struct_callback.name, [Byref(mainctx)]
        )

        # TODO: check - maybe I don't need to explictly set this
        mat_set_dm = petsc_call('MatSetDM', [sobjs['Jac'], dmda])

        calls_set_app_ctx = petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])

        base_setup = dmda_calls + (
            snes_create,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            snes_set_type,
            global_x,
            global_b,
            snes_get_ksp,
            ksp_set_tols,
            ksp_set_type,
            ksp_get_pc,
            pc_set_type,
            ksp_set_from_ops,
            matvec_operation,
            formfunc_operation,
            call_struct_callback,
            mat_set_dm,
            calls_set_app_ctx,
            BlankLine
        )
        extended_setup = self._extend_setup()
        return base_setup + tuple(extended_setup)

    def _extend_setup(self):
        """
        Hook for subclasses to add additional setup calls.
        """
        return []

    def _create_dmda_calls(self, dmda):
        dmda_create = self._create_dmda(dmda)
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])
        return dmda_create, dm_setup, dm_mat_type

    def _create_dmda(self, dmda):
        grid = self.objs['grid']

        nspace_dims = len(grid.dimensions)

        # MPI communicator
        args = [self.objs['comm']]

        # Type of ghost nodes
        args.extend(['DM_BOUNDARY_GHOSTED' for _ in range(nspace_dims)])

        # Stencil type
        if nspace_dims > 1:
            args.append('DMDA_STENCIL_BOX')

        # Global dimensions
        args.extend(list(grid.shape)[::-1])
        # No.of processors in each dimension
        if nspace_dims > 1:
            args.extend(list(grid.distributor.topology)[::-1])

        # Number of degrees of freedom per node
        args.append(dmda.dofs)
        # "Stencil width" -> size of overlap
        args.append(dmda.stencil_width)
        args.extend([Null]*nspace_dims)

        # The distributed array object
        args.append(Byref(dmda))

        # The PETSc call used to create the DMDA
        dmda = petsc_call('DMDACreate%sd' % nspace_dims, args)

        return dmda


class CoupledSetup(BaseSetup):

    @property
    def snes_ctx(self):
        return Byref(self.solver_objs['jacctx'])

    def _extend_setup(self):
        sobjs = self.solver_objs

        dmda = sobjs['dmda']
        create_field_decomp = petsc_call(
            'DMCreateFieldDecomposition',
            [dmda, Byref(sobjs['n_fields']), Null, Byref(sobjs['fields']), Byref(sobjs['subdms'])]
            )
        matop_create_submats_op = petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_CREATE_SUBMATRICES',
             MatShellSetOp(self.cbbuilder.submatrices_callback.name, void, void)]
        )

        ffps = [DummyExpr(FieldFromComposite(i._C_symbol, sobjs['jacctx']), i._C_symbol) for i in sobjs['jacctx'].fields]

        call_coupled_struct_callback = petsc_call(
            'PopulateMatContext', [Byref(sobjs['jacctx']), sobjs['subdms'], sobjs['fields']]
        )

        shell_set_ctx = petsc_call('MatShellSetContext', [sobjs['Jac'], Byref(sobjs['jacctx']._C_symbol)])

        create_submats = petsc_call(
            'MatCreateSubMatrices', [sobjs['Jac'], sobjs['n_fields'],
            sobjs['fields'], sobjs['fields'], 'MAT_INITIAL_MATRIX',
            Byref(FieldFromComposite(sobjs['submats'].base, sobjs['jacctx']))]
        )

        # probs shouldn't be here but..
        targets = self.injectsolve.expr.rhs.fielddata.targets

        deref_dms = [
            DummyExpr(sobjs['da%s' % t.name], sobjs['subdms'].indexed[i])
            for i, t in enumerate(targets)
        ]

        xglobals = [
            petsc_call('DMCreateGlobalVector', 
                    [sobjs['da%s' % t.name], Byref(sobjs['xglobal%s' % t.name])]) 
            for t in targets
        ]

        bglobals = [
            petsc_call('DMCreateGlobalVector', 
                    [sobjs['da%s' % t.name], Byref(sobjs['bglobal%s' % t.name])]) 
            for t in targets
        ]

        return [create_field_decomp, matop_create_submats_op] + [call_coupled_struct_callback, shell_set_ctx, create_submats] + deref_dms + xglobals + bglobals


class Solver:
    def __init__(self, solver_objs, objs, injectsolve, iters, cbbuilder,
                 timedep=None, **kwargs):
        self.solver_objs = solver_objs
        self.objs = objs
        self.injectsolve = injectsolve
        self.iters = iters
        self.cbbuilder = cbbuilder
        self.timedep = timedep

        self.calls = self._execute_solve()
        self.spatial_body = self._spatial_loop_nest()

        space_iter, = self.spatial_body
        self.mapper = {space_iter: self.calls}

    def _execute_solve(self):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        sobjs = self.solver_objs

        target = self.injectsolve.expr.rhs.fielddata.target

        struct_assignment = self.timedep.assign_time_iters(sobjs['userctx'])

        rhs_callback = self.cbbuilder.formrhss.pop()

        dmda = sobjs['dmda']

        rhs_call = petsc_call(rhs_callback.name, [sobjs['dmda'], sobjs['b_global']])

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(sobjs['x_local'])])

        vec_replace_array = self.timedep.replace_array(sobjs)

        dm_local_to_global_x = petsc_call(
            'DMLocalToGlobal', [dmda, sobjs['x_local'], 'INSERT_VALUES',
                                sobjs['x_global']]
        )

        snes_solve = petsc_call('SNESSolve', [
            sobjs['snes'], sobjs['b_global'], sobjs['x_global']]
        )

        dm_global_to_local_x = petsc_call('DMGlobalToLocal', [
            dmda, sobjs['x_global'], 'INSERT_VALUES', sobjs['x_local']]
        )

        run_solver_calls = (struct_assignment,) + (
            rhs_call,
            local_x
        ) + vec_replace_array + (
            dm_local_to_global_x,
            snes_solve,
            dm_global_to_local_x,
            BlankLine,
        )
        return List(body=run_solver_calls)

    def _spatial_loop_nest(self):
        spatial_body = []
        # TODO: remove the iters[0]
        for tree in retrieve_iteration_tree(self.iters[0]):
            root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
            if self.injectsolve in FindNodes(InjectSolveDummy).visit(root):
                spatial_body.append(root)
        return spatial_body


class CoupledSolver(Solver):
    # NOTE: note this is obvs for debugging, shouldn't acc need to override this whole function
    def _execute_solve(self):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        sobjs = self.solver_objs

        struct_assignment = self.timedep.assign_time_iters(sobjs['userctx'])

        rhs_callbacks = self.cbbuilder.formrhss

        dmda = sobjs['dmda']

        targets = self.injectsolve.expr.rhs.fielddata.targets

        rhs_call_pn1 = petsc_call(rhs_callbacks[0].name, [sobjs['da%s'%targets[0].name], sobjs['bglobal'+targets[0].name]])

        rhs_call_pn2 = petsc_call(rhs_callbacks[1].name, [sobjs['da%s'%targets[1].name], sobjs['bglobal'+targets[1].name]])

        local_x_target1 = petsc_call('DMCreateLocalVector',
                                    [sobjs['da%s'%targets[0].name], Byref(sobjs['xlocal'+targets[0].name])])

        local_x_target2 = petsc_call('DMCreateLocalVector',
                                    [sobjs['da%s'%targets[1].name], Byref(sobjs['xlocal'+targets[1].name])])

        vec_replace_array = self.timedep.replace_array(sobjs)

        dm_local_to_global_xpn1 = petsc_call(
            'DMLocalToGlobal', [sobjs['da%s'%targets[0].name], sobjs['xlocal'+targets[0].name], 'INSERT_VALUES',
                                sobjs['xglobal'+targets[0].name]]
        )

        dm_local_to_global_xpn2 = petsc_call(
            'DMLocalToGlobal', [sobjs['da%s'%targets[1].name], sobjs['xlocal'+targets[1].name], 'INSERT_VALUES',
                                sobjs['xglobal'+targets[1].name]]
        )

        scatterpn1_create = petsc_call('VecScatterCreate', [sobjs['x_global'], sobjs['fields'].indexed[0], sobjs['xglobal'+targets[0].name], Null, Byref(sobjs['scatterpn1'])])
        scatterpn2_create = petsc_call('VecScatterCreate', [sobjs['x_global'], sobjs['fields'].indexed[1], sobjs['xglobal'+targets[1].name], Null, Byref(sobjs['scatterpn2'])])

        vec_scatter_begin_pn1 = petsc_call('VecScatterBegin', [sobjs['scatterpn1'], sobjs['xglobal'+targets[0].name], sobjs['x_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])
        vec_scatter_end_pn1 = petsc_call('VecScatterEnd', [sobjs['scatterpn1'], sobjs['xglobal'+targets[0].name], sobjs['x_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])

        vec_scatter_begin_pn2 = petsc_call('VecScatterBegin', [sobjs['scatterpn2'], sobjs['xglobal'+targets[1].name], sobjs['x_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])
        vec_scatter_end_pn2 = petsc_call('VecScatterEnd', [sobjs['scatterpn2'], sobjs['xglobal'+targets[1].name], sobjs['x_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])


        vec_scatter_begin_b_pn1 = petsc_call('VecScatterBegin', [sobjs['scatterpn1'], sobjs['bglobal'+targets[0].name], sobjs['b_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])
        vec_scatter_end__bpn1 = petsc_call('VecScatterEnd', [sobjs['scatterpn1'], sobjs['bglobal'+targets[0].name], sobjs['b_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])

        vec_scatter_begin_b_pn2 = petsc_call('VecScatterBegin', [sobjs['scatterpn2'], sobjs['bglobal'+targets[1].name], sobjs['b_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])
        vec_scatter_end_b_pn2 = petsc_call('VecScatterEnd', [sobjs['scatterpn2'], sobjs['bglobal'+targets[1].name], sobjs['b_global'], 'INSERT_VALUES', 'SCATTER_REVERSE'])

        snes_solve = petsc_call('SNESSolve', [
            sobjs['snes'], sobjs['b_global'], sobjs['x_global']]
        )

        vec_scatter_begin_forward_pn1 = petsc_call('VecScatterBegin', [sobjs['scatterpn1'], sobjs['x_global'], sobjs['xglobal'+targets[0].name], 'INSERT_VALUES', 'SCATTER_FORWARD'])
        vec_scatter_end_forward_pn1 = petsc_call('VecScatterEnd', [sobjs['scatterpn1'], sobjs['x_global'], sobjs['xglobal'+targets[0].name], 'INSERT_VALUES', 'SCATTER_FORWARD'])

        vec_scatter_begin_forward_pn2 = petsc_call('VecScatterBegin', [sobjs['scatterpn2'], sobjs['x_global'], sobjs['xglobal'+targets[1].name], 'INSERT_VALUES', 'SCATTER_FORWARD'])
        vec_scatter_end_forward_pn2 = petsc_call('VecScatterEnd', [sobjs['scatterpn2'], sobjs['x_global'], sobjs['xglobal'+targets[1].name], 'INSERT_VALUES', 'SCATTER_FORWARD'])


        dm_global_to_local_xpn1 = petsc_call('DMGlobalToLocal', [
            sobjs['da%s'%targets[0].name], sobjs['xglobal'+targets[0].name], 'INSERT_VALUES', sobjs['xlocal'+targets[0].name]]
        )

        dm_global_to_local_xpn2 = petsc_call('DMGlobalToLocal', [
            sobjs['da%s'%targets[1].name], sobjs['xglobal'+targets[1].name], 'INSERT_VALUES', sobjs['xlocal'+targets[1].name]]
        )

        run_solver_calls = (struct_assignment,) + (
            rhs_call_pn1,
            rhs_call_pn2,
            local_x_target1,
            local_x_target2,
        ) + vec_replace_array + (
            dm_local_to_global_xpn1,
            dm_local_to_global_xpn2,
            scatterpn1_create,
            scatterpn2_create,
            vec_scatter_begin_pn1,
            vec_scatter_end_pn1,
            vec_scatter_begin_pn2,
            vec_scatter_end_pn2,
            vec_scatter_begin_b_pn1,
            vec_scatter_end__bpn1,
            vec_scatter_begin_b_pn2,
            vec_scatter_end_b_pn2,
            snes_solve,
            vec_scatter_begin_forward_pn1,
            vec_scatter_end_forward_pn1,
            vec_scatter_begin_forward_pn2,
            vec_scatter_end_forward_pn2,
            dm_global_to_local_xpn1,
            dm_global_to_local_xpn2,
            BlankLine,
        )
        return List(body=run_solver_calls)


class NonTimeDependent:
    def __init__(self, injectsolve, iters, **kwargs):
        self.injectsolve = injectsolve
        self.iters = iters
        self.kwargs = kwargs
        self.origin_to_moddim = self._origin_to_moddim_mapper(iters)
        self.time_idx_to_symb = injectsolve.expr.rhs.time_mapper

    @property
    # TODO: for coupled solves, could have a case where one function is a TimeFunction
    # but the other is a Function, but they both depend on time.
    def targets(self):
        return self.injectsolve.expr.rhs.fielddata.targets

    def _origin_to_moddim_mapper(self, iters):
        return {}

    def uxreplace_time(self, body):
        return body

    def replace_array(self, solver_objs):
        """
        VecReplaceArray() is a PETSc function that allows replacing the array
        of a `Vec` with a user provided array.
        https://petsc.org/release/manualpages/Vec/VecReplaceArray/

        This function is used to replace the array of the PETSc solution `Vec`
        with the array from the `Function` object representing the target.

        Examples
        --------
        >>> self.target
        f1(x, y)
        >>> call = replace_array(solver_objs)
        >>> print(call)
        PetscCall(VecReplaceArray(x_local_0,f1_vec->data));
        """
        to_replace = []
        for target in self.targets:
            field_from_ptr = FieldFromPointer(
                target.function._C_field_data, target.function._C_symbol
            )
            try:
                xlocal = solver_objs['xlocal'+target.name]
            except KeyError:
                xlocal = solver_objs['x_local']

            vec_replace_array = (petsc_call(
                'VecReplaceArray', [xlocal, field_from_ptr]
            ),)
            to_replace.extend(vec_replace_array)
        return tuple(to_replace)

    def assign_time_iters(self, struct):
        return []


class TimeDependent(NonTimeDependent):
    """
    A class for managing time-dependent solvers.

    This includes scenarios where the target is not directly a `TimeFunction`,
    but depends on other functions that are.

    Outline of time loop abstraction with PETSc:

    - At PETScSolve, time indices are replaced with temporary `Symbol` objects
      via a mapper (e.g., {t: tau0, t + dt: tau1}) to prevent the time loop
      from being generated in the callback functions. These callbacks, needed
      for each `SNESSolve` at every time step, don't require the time loop, but
      may still need access to data from other time steps.
    - All `Function` objects are passed through the initial lowering via the
      `LinearSolveExpr` object, ensuring the correct time loop is generated
      in the main kernel.
    - Another mapper is created based on the modulo dimensions
      generated by the `LinearSolveExpr` object in the main kernel
      (e.g., {time: time, t: t0, t + 1: t1}).
    - These two mappers are used to generate a final mapper `symb_to_moddim`
      (e.g. {tau0: t0, tau1: t1}) which is used at the IET level to
      replace the temporary `Symbol` objects in the callback functions with
      the correct modulo dimensions.
    - Modulo dimensions are updated in the matrix context struct at each time
      step and can be accessed in the callback functions where needed.
    """
    # TODO: move these funcs/properties around

    def is_target_time(self, target):
        return any(i.is_Time for i in target.dimensions)

    @property
    def time_spacing(self):
        return self.injectsolve.expr.rhs.grid.stepping_dim.spacing

    def target_time(self, target):
        target_time = [
            i for i, d in zip(target.indices, target.dimensions)
            if d.is_Time
        ]
        assert len(target_time) == 1
        target_time = target_time.pop()
        return target_time

    @property
    def symb_to_moddim(self):
        """
        Maps temporary `Symbol` objects created during `PETScSolve` to their
        corresponding modulo dimensions (e.g. creates {tau0: t0, tau1: t1}).
        """
        mapper = {
            v: k.xreplace({self.time_spacing: 1, -self.time_spacing: -1})
            for k, v in self.time_idx_to_symb.items()
        }
        return {symb: self.origin_to_moddim[mapper[symb]] for symb in mapper}

    def uxreplace_time(self, body):
        return Uxreplace(self.symb_to_moddim).visit(body)

    def _origin_to_moddim_mapper(self, iters):
        """
        Creates a mapper of the origin of the time dimensions to their corresponding
        modulo dimensions from a list of `Iteration` objects.

        Examples
        --------
        >>> iters
        (<WithProperties[affine,sequential]::Iteration time[t0,t1]; (time_m, time_M, 1)>,
         <WithProperties[affine,parallel,parallel=]::Iteration x; (x_m, x_M, 1)>)
        >>> _origin_to_moddim_mapper(iters)
        {time: time, t: t0, t + 1: t1}
        """
        time_iter = [i for i in iters if any(d.is_Time for d in i.dimensions)]
        mapper = {}

        if not time_iter:
            return mapper

        for i in time_iter:
            for d in i.dimensions:
                if d.is_Modulo:
                    mapper[d.origin] = d
                elif d.is_Time:
                    mapper[d] = d
        return mapper

    def replace_array(self, solver_objs):
        """
        In the case that the actual target is time-dependent e.g a `TimeFunction`,
        a pointer to the first element in the array that will be updated during
        the time step is passed to VecReplaceArray().

        Examples
        --------
        >>> self.target
        f1(time + dt, x, y)
        >>> calls = replace_array(solver_objs)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(x_local_0,&(localsize_0)));
        float * start_ptr_0 = (time + 1)*localsize_0 + (float*)(f1_vec->data);
        PetscCall(VecReplaceArray(x_local_0,start_ptr_0));

        >>> self.target
        f1(t + dt, x, y)
        >>> calls = replace_array(solver_objs)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(x_local_0,&(localsize_0)));
        float * start_ptr_0 = t1*localsize_0 + (float*)(f1_vec->data);
        """
        # TODO: improve this
        to_replace = []
        for target in self.targets:
            if self.is_target_time(target):
                mapper = {self.time_spacing: 1, -self.time_spacing: -1}
                target_time = self.target_time(target).xreplace(mapper)

                try:
                    target_time = self.origin_to_moddim[target_time]
                except KeyError:
                    pass
                
                # TODO: improve this logic, shouldn't need try and except
                try:
                    xlocal = solver_objs['xlocal'+target.name]
                except KeyError:
                    xlocal = solver_objs['x_local']

                start_ptr = solver_objs[target.name+'_ptr']

                vec_get_size = petsc_call(
                    'VecGetSize', [xlocal, Byref(solver_objs['localsize'])]
                )

                field_from_ptr = FieldFromPointer(
                    target.function._C_field_data, target.function._C_symbol
                )

                expr = DummyExpr(
                    start_ptr, cast_mapper[(target.dtype, '*')](field_from_ptr) +
                    Mul(target_time, solver_objs['localsize']), init=True
                )

                vec_replace_array = petsc_call(
                    'VecReplaceArray', [xlocal, start_ptr]
                )
                to_replace.extend([vec_get_size, expr, vec_replace_array])
            else:
                tmp = super().replace_array(solver_objs)
                to_replace.extend(tmp)
        return tuple(to_replace)

    def assign_time_iters(self, struct):
        """
        Assign required time iterators to the struct.
        These iterators are updated at each timestep in the main kernel
        for use in callback functions.

        Examples
        --------
        >>> struct
        ctx
        >>> struct.fields
        [h_x, x_M, x_m, f1(t, x), t0, t1]
        >>> assigned = assign_time_iters(struct)
        >>> print(assigned[0])
        ctx.t0 = t0;
        >>> print(assigned[1])
        ctx.t1 = t1;
        """
        to_assign = [
            f for f in struct.fields if (f.is_Dimension and (f.is_Time or f.is_Modulo))
        ]
        time_iter_assignments = [
            DummyExpr(FieldFromComposite(field, struct), field)
            for field in to_assign
        ]
        return time_iter_assignments


Null = Macro('NULL')
void = 'void'
dummyctx = Symbol('lctx')
dummyptr = DummyArg('dummy')

# SubMatrixCtx struct members
rows = IS(name='rows', nindices=1)
cols = IS(name='cols', ninidces=1)

# JacMatrixCtx struct members
Subdms = SubDM(name='subdms', nindices=1)
Fields = IS(name='fields', nindices=1)


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
