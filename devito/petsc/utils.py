import os

from pathlib import Path
from devito.tools import memoized_func


class PetscOSError(OSError):
    pass


solver_mapper = {
    'gmres': 'KSPGMRES',
    'jacobi': 'PCJACOBI',
    None: 'PCNONE'
}


@memoized_func
def get_petsc_dir():
    petsc_dir = os.environ.get('PETSC_DIR')
    if petsc_dir is None:
        raise PetscOSError("PETSC_DIR environment variable not set")
    return petsc_dir


@memoized_func
def get_petsc_arch():
    # Note: users don't have to explicitly set PETSC_ARCH
    # if they add it to the PETSC_DIR path
    return os.environ.get('PETSC_ARCH')


@memoized_func
def core_metadata():
    petsc_dir = get_petsc_dir()
    petsc_arch = get_petsc_arch()

    petsc_include = (os.path.join(petsc_dir, 'include'),)
    petsc_lib = (os.path.join(petsc_dir, 'lib'),)
    if petsc_arch:
        petsc_include += (os.path.join(petsc_dir, petsc_arch, 'include'),)
        petsc_lib += (os.path.join(petsc_dir, petsc_arch, 'lib'),)

    return {
        'includes': ('petscsnes.h', 'petscdmda.h'),
        'include_dirs': petsc_include,
        'libs': ('petsc'),
        'lib_dirs': petsc_lib,
        'ldflags': tuple([f"-Wl,-rpath,{lib}" for lib in petsc_lib])
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
