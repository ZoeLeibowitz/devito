from collections import OrderedDict

from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle)
from devito.tools import timed_pass
from devito.petsc.types import MetaData
from devito.petsc.iet.utils import petsc_iet_mapper

__all__ = ['iet_build']


@timed_pass(name='build')
def iet_build(stree):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            return List(body=queues.pop(i))

        elif i.is_Exprs:
            exprs = []
            for e in i.exprs:
                if e.is_Increment:
                    exprs.append(Increment(e))
                elif isinstance(e.rhs, MetaData):
                    exprs.append(petsc_iet_mapper[e.operation](e, operation=e.operation))
                else:
                    exprs.append(Expression(e, operation=e.operation))
            body = ExpressionBundle(i.ispace, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            body = Conditional(i.guard, queues.pop(i))

        elif i.is_Iteration:
            if i.dim.is_Virtual:
                body = List(body=queues.pop(i))
            else:
                body = Iteration(queues.pop(i), i.dim, i.limits,
                                 direction=i.direction, properties=i.properties,
                                 uindices=i.sub_iterators)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            try:
                body = HaloSpot(queues.pop(i), i.halo_scheme)
            except KeyError:
                body = HaloSpot(None, i.halo_scheme)

        elif i.is_Sync:
            body = SyncSpot(i.sync_ops, body=queues.pop(i, None))

        queues.setdefault(i.parent, []).append(body)

    assert False
