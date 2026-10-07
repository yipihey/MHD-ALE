#include "Problemdata.hpp"

namespace mfem
{

ProblemData *GetProblemData(Testcase p, int dim, real_t Bmag)
{
   switch (p)
   {
   case PERIODIC_SHEAR:
      return GetProblemPeriodicShear(dim);
   case PERIODIC_BOX:
      return GetProblemPeriodicBox(dim);
   case TAYLOR_GREEN:
      return GetProblemTaylorGreen(dim);
      break;
   case MHD_ROTOR:
      return GetProblemMHDrotor();
      break;
   case MHD_BLAST:
      return GetProblemMHDblast(Bmag);
      break;
   case BrioWushocktube:
      return GetProblemBrioWushocktube(dim);
      break;
   case STATIC3D:
      return GetProblemStatic3D();
      break;
   case SuperFast:
      return GetProblemSuperFast(dim);
      break;
   case MHDshocktube3:
      return GetProblemMHDshocktube3(dim);
      break;
   case MHDshocktube4:
      return GetProblemMHDshocktube4(dim);
      break;
   default:
      mfem_error("Unknown Problem type");
      break;
   }
}

}
