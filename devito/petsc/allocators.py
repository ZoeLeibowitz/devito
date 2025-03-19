from devito.data.allocators import MemoryAllocator
from pathlib import Path
from devito.petsc.utils import core_metadata


# class PetscMemoryAllocator(MemoryAllocator):
#     """
#     """
#     @classmethod
#     def initialize(cls):
#         metadata = core_metadata()
#         lib_dir = Path(metadata['lib_dirs'][-1])

#         try:
#             cls.lib = ctypes.CDLL(lib_dir/'libpetsc.so')
#         except OSError:
#             cls.lib = None

#     def _alloc_C_libcall(self, size, ctype):
#         c_bytesize = ctypes.c_ulong(size * ctypes.sizeof(ctype))
#         c_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.c_void_p)
#         ret = self.lib.PetscMalloc(size, ctypes.byref(c_pointer))

#         if ret == 0:
#             return c_pointer, (c_pointer, )
#         else:
#             return None, None

#     def free(self, c_pointer):
#         self.lib.PetscFree(c_pointer)

# PETSC_ALLOC = PetscMemoryAllocator()
