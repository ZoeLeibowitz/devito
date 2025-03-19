import os

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
    # Note: users don't have to set PETSC_ARCH
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
