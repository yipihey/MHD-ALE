#include "Problemdata.hpp"

namespace mfem
{

static real_t Bmag = 1.0;
static real_t mu0 = 4.0*M_PI;

static real_t rho0(const Vector &x)
{
    return 1.0;
}

static real_t gamma_func(const Vector &x)
{
    return 5.0 / 3.0;
}

static void v0(const Vector &x, Vector &v)
{
    v(0) = 0.0;
    v(1) = 0.0;
    v(2) = 0.0;
}

static void A0(const Vector &x, Vector &A)
{
    A(0) = 0.0;
    A(1) = 0.0;
    A(2) = Bmag*sqrt(4.0/3.0*mu0)*x(1);
}

static void B0(const Vector &x, Vector &B)
{
    B(0) = Bmag*sqrt(4.0/3.0*mu0);
    B(1) = 0.0;
    B(2) = 0.0;
}

static void H0(const Vector &x, Vector &H)
{
    B0(x, H);
    H /= mu0; 
}


static real_t e0(const Vector &x)
{
    MFEM_ABORT("e0 function is not defined for MHD blast problem!");
    return 0.0;
}

ProblemData *GetProblemMHDblast(real_t Bmag_)
{
    Bmag = Bmag_;
    ProblemData *pd = new ProblemData(e0, rho0, gamma_func, v0, A0, B0);
    pd->SetViscosity(true); 
    pd->SetVorticity(false); // no vorticity
    pd->SetMu(mu0); // magnetic permeability
    pd->SetDimension(3);
    
    Array<int> ess_bdr_x({0,0,1,0,1,0});
    Array<int> ess_bdr_y({0,1,0,1,0,0});
    Array<int> ess_bdr_z({1,0,0,0,0,1});
    pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y, ess_bdr_z);
    
    Array<int> ess_bdr_H({1,1,1,1,1,1});
    pd->SetHBoundaryCondition(ess_bdr_H, H0);
    
    return pd;
}

}