#ifndef VECGF_ERROR_HPP
#define VECGF_ERROR_HPP

#include "mfem.hpp"

namespace mfem
{

real_t ComputeVectorGradError(ParGridFunction &sol,
                                VectorCoefficient &exgrad,
                                const IntegrationRule *irs[] = NULL);
                                
real_t ComputeVectorH1Error(ParGridFunction &sol,
                                VectorCoefficient &exsol, 
                                VectorCoefficient &exgrad,
                                const IntegrationRule *irs[] = NULL);

}

#endif