#ifndef TOOLS_HPP
#define TOOLS_HPP


#define GETHERE \
{ \
    if (Mpi::Root()) { \
        printf("Get here: %s, line %d \n", __FILE__, __LINE__); \
    } \
}

#include "mfem.hpp"

enum class BoundPreservingType
{
    NONE,
    POSITIVE,
    LOCAL,
    GLOBAL
};

namespace mfem
{

// Compute the curl of a scalar field
void ComputeCurlOfScalar(const ParGridFunction &E_gf, ParGridFunction &curl_gf);

// Compute the curl of a vector field
void ComputeCurl3D(const ParGridFunction &E_gf, ParGridFunction &curl_gf);

// Compute the curl of a scalar/vector field
void ComputeCurl(ParGridFunction &A_gf, ParGridFunction &B_gf);

// Compute inner product of two grid functions
real_t GFInnerProduct(const ParGridFunction &gf_1, const ParGridFunction &gf_2);

// Compute Maximum norm value of grid functions
real_t GFMaxNorm(const ParGridFunction &gf);

// Compute minimum and maximum values of a grid function
void GFMinMax(const ParGridFunction &gf, real_t &min_val, real_t &max_val, const IntegrationRule *ir=nullptr);

// Compute the divergence error of a div-free grid function (RT)
real_t GFDivError(ParGridFunction &gf);

real_t GetMeshSize(ParMesh *pmesh);

void ProjectDensity(ParGridFunction &rho_gf, Coefficient &rho_coeff, const IntegrationRule &ir, bool bound_preserving = false, BoundPreservingType bp_type = BoundPreservingType::NONE);

void DivFreeProject(ParGridFunction &gf, ParFiniteElementSpace *W_space, VectorCoefficient &fcoeff);


}

#endif