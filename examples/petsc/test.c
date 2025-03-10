
static char help[] = "";

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>


extern PetscErrorCode PetscInit();
extern PetscErrorCode PetscFinal();

int main()
{
  PetscInit();
  PetscPrintf(PETSC_COMM_WORLD, "Hello World!\n");
  PetscFinalize();

  return 0;
}


PetscErrorCode PetscInit()
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(NULL,NULL,NULL,NULL));
    PetscFunctionReturn(0);
}


PetscErrorCode PetscFinal()
{
    PetscFunctionBeginUser;
    PetscCall(PetscFinalize());
    PetscFunctionReturn(0);
}
