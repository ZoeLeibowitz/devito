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
                                MatReuse, VecScatter, DMCast, LocalIS, LocalSubDMs)


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
        self._matvecs = {}
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
        return next(iter(self._matvecs.values()))

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
        self._make_matvec(fielddata, fielddata.matvecs)
        self._make_formfunc(fielddata)
        self._make_formrhs(fielddata)

    # TODO: probs don't need to pass in fielddata?
    def _make_matvec(self, fielddata, matvecs, prefix='MatMult'):
        # Compile matvec `eqns` into an IET via recursive compilation
        sobjs = self.solver_objs
        irs_matvec, _ = self.rcompile(matvecs,
                                      options={'mpi': False}, sregistry=self.sregistry)
        body_matvec = self._create_matvec_body(List(body=irs_matvec.uiet.body),
                                               fielddata)

        matvec_callback = PETScCallable(
            self.sregistry.make_name(prefix=prefix), body_matvec,
            retval=self.objs['err'],
            parameters=(
                J, X, Y
            )
        )
        self._matvecs[prefix] = matvec_callback
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

        mat_get_dm = petsc_call('MatGetDM', [J, Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(xloc)]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, X,
                                     'INSERT_VALUES', xloc]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, X, 'INSERT_VALUES', xloc
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(yloc)]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [yloc, Byref(y_matvec._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [xloc, Byref(x_matvec._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [yloc, Byref(y_matvec._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [xloc, Byref(x_matvec._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, yloc, 'INSERT_VALUES', Y
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, yloc, 'INSERT_VALUES', Y
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(xloc)]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(yloc)]
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
            parameters=(snes, X,
                        F, dummyptr)
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
            'DMGetLocalVector', [dmda, Byref(xloc)]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, X,
                                     'INSERT_VALUES', xloc]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, X, 'INSERT_VALUES', xloc
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(floc)]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [floc, Byref(f_formfunc._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [xloc, Byref(x_formfunc._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [floc, Byref(f_formfunc._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [xloc, Byref(x_formfunc._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, floc, 'INSERT_VALUES', F
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, floc, 'INSERT_VALUES', F
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(xloc)]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(floc)]
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
                sobjs['callbackdm'], B,
            )
        )
        self._formrhss.append(cb)
        self._efuncs[cb.name] = cb

    def _create_form_rhs_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        sobjs = self.solver_objs
        target = fielddata.target

        dmda = sobjs['callbackdm']
        # TODO: when moving to coupled...perhaps the DMDA should be
        # an argument to this function _create_form_rhs_body

        dm_get_local = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['b_local'])]
        )

        dm_global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, B,
                                     'INSERT_VALUES', sobjs['b_local']]
        )

        dm_global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, B, 'INSERT_VALUES',
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
            B
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['b_local'], 'INSERT_VALUES',
            B
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
        # self._make_local_coupled_ctx()
        self._make_whole_matvec()
        self._make_whole_formfunc()
        self._create_submatrices()
        self._efuncs['PopulateMatContext'] = Symbol('dummy')

    @property
    def submatrices_callback(self):
        return self._submatrices_callback

    @property
    def submatrices(self):
        return self.injectsolve.expr.rhs.fielddata.submatrices

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
            self._make_formfunc(data)
            self._make_formrhs(data)

            row_matvecs = all_fielddata.submatrices.submatrices[t]
            for submat, mtvs in row_matvecs.items():
                if mtvs['matvecs']:
                    self._make_matvec(data, mtvs['matvecs'], prefix='%s_MatMult' % submat)

    def _make_whole_matvec(self):
        sobjs = self.solver_objs

        # obvs improve name
        body = self._create_whole_matvec_callback_body()

        cb = PETScCallable(
            self.sregistry.make_name(prefix='WholeMatMult'), List(body=body),
            retval=self.objs['err'],
            parameters=(J, X, Y)
        )
        self._main_matvec_callback = cb
        self._efuncs[cb.name] = cb

    def _create_whole_matvec_callback_body(self):
        sobjs = self.solver_objs

        ctx_main = petsc_call('MatShellGetContext', [J, Byref(ljacctx)])

        nonzero_submats = self.submatrices.nonzero_submatrix_keys

        deref_mat = []
        mat_get_ctx = []
        vec_get_x = []
        vec_get_y = []
        mat_mult = []
        vec_restore_x = []
        vec_restore_y = []

        for sm in nonzero_submats:
            idx = self.submatrices.submat_to_index[sm]
            ctx = sobjs[sm+'ctx']

            deref_mat.append(
                DummyExpr(sobjs[sm], FieldFromPointer(Submats.indexed[idx], ljacctx))
            )
            mat_get_ctx.append(
                petsc_call('MatShellGetContext', [sobjs[sm], Byref(ctx)])
            )
            vec_get_x.append(
                petsc_call(
                    'VecGetSubVector', [X,
                    Deref(FieldFromPointer(cols.base, sobjs[sm+'ctx'])),
                    Byref(sobjs[sm+'X'])]
                )
            )
            vec_get_y.append(
                petsc_call(
                    'VecGetSubVector', [Y,
                    Deref(FieldFromPointer(rows.base, sobjs[sm+'ctx'])),
                    Byref(sobjs[sm+'Y'])]
                )
            )
            mat_mult.append(
                petsc_call('MatMult', [sobjs[sm], sobjs[sm+'X'], sobjs[sm+'Y']])
            )
            vec_restore_x.append(
                petsc_call(
                    'VecRestoreSubVector', [X,
                    Deref(FieldFromPointer(cols.base, sobjs[sm+'ctx'])),
                    Byref(sobjs[sm+'X'])]
                )
            )
            vec_restore_y.append(
                petsc_call(
                    'VecRestoreSubVector', [Y,
                    Deref(FieldFromPointer(rows.base, sobjs[sm+'ctx'])),
                    Byref(sobjs[sm+'Y'])]
                )
            )

        body = (
            [ctx_main, BlankLine]
            + deref_mat
            + mat_get_ctx
            + vec_get_x
            + vec_get_y
            + mat_mult
            + vec_restore_x
            + vec_restore_y
        )

        body = CallableBody(
            List(body=body),
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
            parameters=(snes, X,
                        F, dummyptr)
        )
        self._main_formfunc_callback = main_formfunc_callback
        self._efuncs[main_formfunc_callback.name] = main_formfunc_callback

    def _create_whole_formfunc_callback_body(self):
        sobjs = self.solver_objs

        # TODO: replace obvs
        # ljacctx = sobjs['ljacctx']
        struct_cast = DummyExpr(ljacctx, StructCast(dummyptr))

        targets = self.injectsolve.expr.rhs.fielddata.targets

        deref_subdms = Dereference(LocalSubdms, ljacctx)
        deref_fields = Dereference(LocalFields, ljacctx)

        vec_get_x = []
        vec_get_f = []
        call_formfunc = []
        vec_restore_x = []
        vec_restore_f = []

        # X = sobjs['X_global']
        # F = sobjs['F_global']

        for i, t in enumerate(targets):
            field_ptr = FieldFromPointer(LocalFields.indexed[i], ljacctx)
            x_name = f'Xglobal{t.name}'
            f_name = f'Fglobal{t.name}'

            # Get sub vectors
            vec_get_x.append(
                petsc_call('VecGetSubVector', [X, field_ptr, Byref(sobjs[x_name])])
            )
            vec_get_f.append(
                petsc_call('VecGetSubVector', [F, field_ptr, Byref(sobjs[f_name])])
            )

            # Call form function
            call_formfunc.append(
                petsc_call(self.formfuncs[i].name, [snes, sobjs[x_name],
                sobjs[f_name], VOIDP(LocalSubdms.indexed[i])])
            )

            # Restore sub vectors
            vec_restore_x.append(
                petsc_call('VecRestoreSubVector', [X, field_ptr, Byref(sobjs[x_name])])
            )
            vec_restore_f.append(
                petsc_call('VecRestoreSubVector', [F, field_ptr, Byref(sobjs[f_name])])
            )

        body = (
            vec_get_x
            + vec_get_f
            + call_formfunc
            + vec_restore_x
            + vec_restore_f
            + [BlankLine]
        )
        body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=(struct_cast, deref_subdms, deref_fields),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))
        return body

    def _create_submatrices(self):
        body = self._create_submat_callback_body()
        sobjs = self.solver_objs

        params = (
            J,
            nfields,
            irow,
            icol,
            matreuse,
            Submats,
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

        n_submats = DummyExpr(
            nsubmats, Mul(nfields, nfields)
        )

        malloc_submats = petsc_call('PetscCalloc1', [nsubmats, Submats])

        mat_get_dm = petsc_call('MatGetDM', [J, Byref(sobjs['callbackdm'])])

        dm_get_app = petsc_call(
            'DMGetApplicationContext', [sobjs['callbackdm'], Byref(sobjs['luserctx'])]
        )

        shell_get_ctx = petsc_call('MatShellGetContext', [J, Byref(ljacctx)])

        # TODO: Not sure if I should use global or local dimensions yet
        dm_get_info = petsc_call(
            'DMDAGetInfo', [sobjs['callbackdm'], Null, Byref(sobjs['M']),
            Byref(sobjs['N']), Null, Null, Null, Null, Byref(dof),
            Null, Null, Null, Null, Null]
        )
        subblock_rows = DummyExpr(subblockrows, Mul(sobjs['M'], sobjs['N']))
        subblock_cols = DummyExpr(subblockcols, Mul(sobjs['M'], sobjs['N']))

        ptr = DummyExpr(submat_arr._C_symbol, Deref(Submats), init=True)

        mat_create = petsc_call('MatCreate', [self.objs['comm'], Byref(block)])
        mat_set_sizes = petsc_call(
            'MatSetSizes', [block, 'PETSC_DECIDE', 'PETSC_DECIDE',
            subblockrows, subblockcols]
        )

        mat_set_type = petsc_call('MatSetType', [block, 'MATSHELL'])

        malloc = petsc_call('PetscMalloc1', [1, Byref(sobjs['submatctx'])])
        i = Dimension(name='i')

        row_idx = DummyExpr(rowidx, IntDiv(i, dof))
        col_idx = DummyExpr(colidx, Modulo(i, dof))

        deref_subdm = Dereference(Subdms, ljacctx)

        # fix:todo: the SUBMAT_CTX doesn't appear in the ccode because it's not
        # an argument to any function -> fix this in the cgen structure code
        set_rows = DummyExpr(
            FieldFromPointer(rows.base, sobjs['submatctx']),
            Byref(irow.indexed[rowidx])
        )
        set_cols = DummyExpr(
            FieldFromPointer(cols.base, sobjs['submatctx']),
            Byref(icol.indexed[colidx])
        )

        dm_set_app_ctx = petsc_call(
            'DMSetApplicationContext', [Subdms.indexed[rowidx], sobjs['luserctx']]
        )

        matset_dm = petsc_call('MatSetDM', [block, Subdms.indexed[rowidx]])

        set_ctx = petsc_call('MatShellSetContext', [block, sobjs['submatctx']])

        mat_setup = petsc_call('MatSetUp', [block])

        assign_block = DummyExpr(submat_arr.indexed[i], block)

        iter_body = (
            mat_create,
            mat_set_sizes,
            mat_set_type,
            malloc,
            row_idx,
            col_idx,
            set_rows,
            set_cols,
            dm_set_app_ctx,
            matset_dm,
            set_ctx,
            mat_setup,
            assign_block
        )

        upper_bound = nsubmats - 1
        iteration = Iteration(List(body=iter_body), i, upper_bound)

        nonzero_submats = self.submatrices.nonzero_submatrix_keys

        matmult_op = []
        for sb in nonzero_submats:
            idx = self.submatrices.submat_to_index[sb]
            matvec = self.matvecs[sb+'_MatMult']
            matmult_op.append(
                petsc_call('MatShellSetOperation', [submat_arr.indexed[idx],
                'MATOP_MULT', MatShellSetOp(matvec.name, void, void)])
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
        ] + matmult_op

        return CallableBody(
            List(body=tuple(body)),
            init=(petsc_func_begin_user,),
            stacks=(shell_get_ctx, deref_subdm),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

    def _make_coupled_ctx(self):
        objs = self.solver_objs
        fields = [Subdms, Fields, Submats]
        self.solver_objs['jacctx'] = petsc_struct(
            name=self.sregistry.make_name(prefix='jctx'), pname='JacobianCtx',
            fields=fields, liveness='lazy'
        )

    # def _make_local_coupled_ctx(self):
    #     objs = self.solver_objs
    #     # TODO: can this struct just be combined with the user ctx? i.e
    #     # I don't think i need two separate ones
    #     fields = objs['jacctx'].fields
    #     self.solver_objs['ljacctx'] = petsc_struct(
    #         name=objs['jacctx'].name, pname=objs['jacctx'].pname,
    #         fields=fields, liveness='lazy',
    #         modifier=' *'
    #     )


class BaseObjectBuilder:
    """
    A base class for constructing objects needed for a PETSc solver.
    Designed to be extended by subclasses, which can override the `_extend_build`
    method to support specific use cases.
    """

    def __init__(self, injectsolve, sregistry=None, **kwargs):
        self.injectsolve = injectsolve
        self.sregistry = sregistry
        self.fielddata = injectsolve.expr.rhs.fielddata
        self.solver_objs = self._build()

    def _build(self):
        """
        # TODO: update docts to reflect new/changes objs
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
        # TODO: for any of the Vec objects used in callback funcs, I don't
        # think need to use symbol registry for them..?
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
            'dmda': DM(sreg.make_name(prefix='da'), liveness='eager', dofs=len(targets)),
            'callbackdm': CallbackDM(sreg.make_name(prefix='dm'), liveness='eager'),
        }
        base_dict = self._target_dependent(base_dict)
        return self._extend_build(base_dict)

    def _target_dependent(self, base_dict):
        sreg = self.sregistry
        targets = self.fielddata.targets
        for target in targets:
            base_dict[target.name+'_ptr'] = StartPtr(
                sreg.make_name(prefix='%s_ptr' % target.name), target.dtype
            )
        base_dict = self._extend_target_dependent(base_dict)
        return base_dict

    def _extend_build(self, base_dict):
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
    def _extend_build(self, base_dict):
        injectsolve = self.injectsolve
        sreg = self.sregistry
        # TODO: add a no_of_targets attribute to the FieldData object
        targets = self.fielddata.targets
        no_targets = len(targets)

        base_dict['fields'] = IS(
            name=sreg.make_name(prefix='fields'), nindices=no_targets
            )
        base_dict['subdms'] = SubDM(
            name=sreg.make_name(prefix='subdms'), nindices=no_targets
            )
        # CHANGE THIS TO PETSCINT
        # base_dict['n_submats'] = Scalar(sreg.make_name(prefix='nsubmats'), dtype=np.int32)

        base_dict['n_fields'] = PetscInt(sreg.make_name(prefix='nfields'))

        # global submatrix sizes
        space_dims = len(self.fielddata.grid.dimensions)

        dim_labels = ["M", "N", "P"]  # Extendable for higher dimensions if needed
        base_dict.update({
            dim_labels[i]: PetscInt(dim_labels[i]) for i in range(space_dims)
        })

        # these don't need to be used -> think can just use fields?
        # base_dict['all_IS_rows'] = IS(name=sreg.make_name(prefix='allrows'), nindices=1)
        # base_dict['all_IS_cols'] = IS(name=sreg.make_name(prefix='allcols'), nindices=1)

        submatrices = injectsolve.expr.rhs.fielddata.submatrices
        submatrix_keys = submatrices.submatrix_keys

        pname = 'SubMatrixCtx'
        fields = [rows, cols]

        base_dict['submatctx'] = petsc_struct(
            name=sreg.make_name(prefix='submatctx'),
            pname=pname, fields=fields,
            modifier=' *', liveness='eager'
        )

        for key in submatrix_keys:
            base_dict[key] = Mat(name=key)

            base_dict[key+'ctx'] = petsc_struct(
                name=key+'ctx', pname=pname,
                fields=fields, modifier=' *', liveness='eager'
            )

            # not sure if it should be global or local yet
            base_dict[key+'X'] = LocalVec(key+'X')
            base_dict[key+'Y'] = LocalVec(key+'Y')
            base_dict[key+'F'] = LocalVec(key+'F')

        # base_dict['matreuse'] = MatReuse(sreg.make_name(prefix='scall'))

        # obvs rethink -> probs don't need?
        for t in targets:
            base_dict['dm%s'%t.name] = CallbackDM(sreg.make_name(prefix='dm%s'%t.name), liveness='eager')
            base_dict['scatter%s'%t.name] = VecScatter(sreg.make_name(prefix='scatter%s'%t.name))

        return base_dict

    def _extend_target_dependent(self, base_dict):
        sreg = self.sregistry
        targets = self.fielddata.targets
        for target in targets:
            base_dict['xlocal'+target.name] = LocalVec(
                sreg.make_name(prefix='xlocal%s' % target.name), liveness='eager'
            )
            base_dict['Fglobal'+target.name] = LocalVec(
                sreg.make_name(prefix='Fglobal%s' % target.name), liveness='eager'
            )
            base_dict['Xglobal'+target.name] = LocalVec(
                sreg.make_name(prefix='Xglobal%s' % target.name)
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
                sreg.make_name(prefix='da%s' % target.name), liveness='eager'
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
        stencil_width = self.injectsolve.expr.rhs.fielddata.space_order
        args.append(stencil_width)
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
            Byref(FieldFromComposite(Submats.base, sobjs['jacctx']))]
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

        rhs_callback = next(iter(self.cbbuilder.formrhss.values()))

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
        xglobal = sobjs['x_global']
        bglobal = sobjs['b_global']

        targets = self.injectsolve.expr.rhs.fielddata.targets

        rhs_calls = ()
        local_x_vecs = ()
        loc_to_glob = ()
        scatter_create = ()
        scatter_reverse_x = ()
        scatter_reverse_b = ()
        scatter_forward_x = ()
        glob_to_loc = ()

        for i, (c, t) in enumerate(zip(rhs_callbacks, targets)):
            dm = sobjs['da%s'%t.name]
            target_xlocal = sobjs['xlocal%s' % t.name]
            target_xglobal = sobjs['xglobal%s' % t.name]
            target_bglobal = sobjs['bglobal%s' % t.name]
            field = sobjs['fields'].indexed[i]
            scatter = sobjs['scatter%s' % t.name]

            rhs_calls += (petsc_call(c.name, [dm, target_bglobal]),)

            local_x_vecs += (petsc_call(
                'DMCreateLocalVector', [dm, Byref(target_xlocal)]
            ),)
            loc_to_glob += (petsc_call(
                'DMLocalToGlobal', [dm, target_xlocal, 'INSERT_VALUES', target_xglobal]
            ),)
            scatter_create += (petsc_call(
                'VecScatterCreate', [xglobal, field, target_xglobal, Null, Byref(scatter)]
            ),)
            scatter_reverse_x += (petsc_call(
                'VecScatterBegin', [scatter, target_xglobal, xglobal, 'INSERT_VALUES', 'SCATTER_REVERSE']
            ),)
            scatter_reverse_x += (petsc_call(
                'VecScatterEnd', [scatter, target_xglobal, xglobal, 'INSERT_VALUES', 'SCATTER_REVERSE']
            ),)
            scatter_reverse_b += (petsc_call(
                'VecScatterBegin', [scatter, target_bglobal, bglobal, 'INSERT_VALUES', 'SCATTER_REVERSE']
            ),)
            scatter_reverse_b += (petsc_call(
                'VecScatterEnd', [scatter, target_bglobal, bglobal, 'INSERT_VALUES', 'SCATTER_REVERSE']
            ),)
            scatter_forward_x += (petsc_call(
                'VecScatterBegin', [scatter, xglobal, target_xglobal, 'INSERT_VALUES', 'SCATTER_FORWARD']
            ),)
            scatter_forward_x += (petsc_call(
                'VecScatterEnd', [scatter, xglobal, target_xglobal, 'INSERT_VALUES', 'SCATTER_FORWARD']
            ),)
            glob_to_loc += (petsc_call(
                'DMGlobalToLocal', [dm, target_xglobal, 'INSERT_VALUES', target_xlocal]
            ),)
    
        vec_replace_array = self.timedep.replace_array(sobjs)

        snes_solve = (petsc_call('SNESSolve', [sobjs['snes'], bglobal, xglobal]),)

        return List(
            body=(
                (struct_assignment,)
                + rhs_calls
                + local_x_vecs
                + vec_replace_array
                + loc_to_glob
                + scatter_create
                + scatter_reverse_x
                + scatter_reverse_b
                + snes_solve
                + scatter_forward_x
                + glob_to_loc
                + (BlankLine,)
            )
        )



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


################ Symbols/Objects that are not unique to each PETScSolve ################ 
Null = Macro('NULL')
void = 'void'
dummyctx = Symbol('lctx')
dummyptr = DummyArg('dummy')
dof = PetscInt('dof')
block = LocalMat('block')
submat_arr = SubMats(name='submat_arr', nindices=1)
subblockrows = PetscInt('subblockrows')
subblockcols = PetscInt('subblockcols')
rowidx = PetscInt('rowidx')
colidx = PetscInt('colidx')
J = Mat('J')
X = GlobalVec('X')
xloc = LocalVec('xloc')
Y = GlobalVec('Y')
yloc = LocalVec('yloc')
F = GlobalVec('F')
floc = LocalVec('floc')
B = GlobalVec('B')
cbdm = CallbackDM('dm', liveness='eager')
nfields = PetscInt('nfields')
irow = IS(name='irow', nindices=1)
icol = IS(name='icol', nindices=1)
nsubmats = Scalar('nsubmats', dtype=np.int32)
matreuse = MatReuse('scall')
snes = SNES('snes')


# TODO: obvs generalise and improve..should probs use caststar?
class StructCast(Cast):
    _base_typ = 'struct JacobianCtx *'

# SubMatrixCtx struct members
rows = IS(name='rows', nindices=1)
cols = IS(name='cols', ninidces=1)

# JacMatrixCtx struct members
Subdms = SubDM(name='subdms', nindices=1)
LocalSubdms = LocalSubDMs(name='subdms', nindices=1)
Fields = IS(name='fields', nindices=1)
LocalFields = LocalIS(name='fields', nindices=1)
Submats = SubMats(name='submats', nindices=1)

ljacctx = petsc_struct(
    name='jctx',
    pname='JacobianCtx',
    fields=[Subdms, Fields, Submats],
    liveness='lazy',
    modifier=' *'
)

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
