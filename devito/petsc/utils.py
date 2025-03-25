import os

from pathlib import Path
from devito.tools import memoized_func


solver_mapper = {
    'gmres': 'KSPGMRES',
    'jacobi': 'PCJACOBI',
    None: 'PCNONE'
}


@memoized_func
def get_petsc_dir():
    # *** First try: via commonly used environment variables
    for i in ['PETSC_DIR']:
        petsc_dir = os.environ.get(i)
        if petsc_dir:
            return petsc_dir
    # TODO: Raise error if PETSC_DIR is not set
    return None


@memoized_func
def get_petsc_arch():
    # *** First try: via commonly used environment variables
    for i in ['PETSC_ARCH']:
        petsc_arch = os.environ.get(i)
        if petsc_arch:
            return petsc_arch
    # TODO: Raise error if PETSC_ARCH is not set
    return None


def core_metadata():
    petsc_dir = get_petsc_dir()
    petsc_arch = get_petsc_arch()

    # Include directories
    global_include = os.path.join(petsc_dir, 'include')
    config_specific_include = os.path.join(petsc_dir, f'{petsc_arch}', 'include')
    include_dirs = (global_include, config_specific_include)

    # Lib directories
    lib_dir = os.path.join(petsc_dir, f'{petsc_arch}', 'lib')

    return {
        'includes': ('petscsnes.h', 'petscdmda.h'),
        'include_dirs': include_dirs,
        'libs': ('petsc'),
        'lib_dirs': lib_dir,
        'ldflags': ('-Wl,-rpath,%s' % lib_dir)
    }


@memoized_func
def get_petsc_variables():
    """
    Taken from https://www.firedrakeproject.org/_modules/firedrake/petsc.html
    Get a dict of PETSc environment variables from the file:
    $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables
    """
    petsc_dir = get_petsc_dir()
    petsc_arch = get_petsc_arch()
    path = [petsc_dir, petsc_arch, 'lib', 'petsc', 'conf', 'petscvariables']
    variables_path = Path(*path)

    with open(variables_path) as fh:
        # Split lines on first '=' (assignment)
        splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
    return {k.strip(): v.strip() for k, v in splitlines}


@memoized_func
def get_petsc_precision():
    """
    Get the PETSc precision.
    """
    petsc_variables = get_petsc_variables()
    return petsc_variables['PETSC_PRECISION']
