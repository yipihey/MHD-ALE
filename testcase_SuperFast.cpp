#include "Problemdata.hpp"

namespace mfem
{

static real_t mid_x = 0.0;
static real_t size_yz = 0.025;
static real_t mu0 = 1.0;
static real_t gamma = 5.0 / 3.0;

static real_t rhoL = 1.0;
static real_t uL = -3.1;
static real_t vL = 0.0;
static real_t pL = 0.45;
static real_t BxL = 0.0;
static real_t ByL = 0.5;

static real_t rhoR = 1.0;
static real_t uR = 3.1;
static real_t vR = 0.0;
static real_t pR = 0.45;
static real_t BxR = 0.0;
static real_t ByR = 0.5;

static real_t B0x = 0.0;
static real_t B0y = 0.0;

static real_t rho0(const Vector &x)
{
    return x(0)<=mid_x? rhoL: rhoR;
}

static real_t gamma_func(const Vector &x)
{
    return gamma;
}

static void v0(const Vector &x, Vector &v)
{
    v(0) = x(0)<=mid_x? uL: uR;
    v(1) = x(0)<=mid_x? vL: vR;
    if(x.Size()==3) v(2) = 0.0;
}

static void A0(const Vector &x, Vector &A)
{
    real_t Az = x(0)<=mid_x? BxL*x(1)-ByL*x(0): BxR*x(1)-ByR*x(0);
    if(x.Size()==3)
    {
        A(0) = 0.0;
        A(1) = 0.0;
        A(2) = Az;
    }
    else
    {
        A(0) = Az;
    }
}

static void B0(const Vector &x, Vector &B)
{
    B(0) = x(0)<=mid_x? BxL: BxR;
    B(1) = x(0)<=mid_x? ByL: ByR;
    if(x.Size()==3) B(2) = 0.0;
}

static void PeriodicB0(const Vector &x, Vector &B)
{
    B(0) = B0x;
    B(1) = B0y;
    if(x.Size()==3) B(2) = 0.0;
}

static void H0(const Vector &x, Vector &H)
{
    B0(x, H);
    H /= mu0; 
}


static real_t e0(const Vector &x)
{
    real_t p = x(0)<=mid_x? pL: pR;
    return p / (gamma_func(x) - 1.0) / rho0(x);
}

ProblemData *GetProblemSuperFast(int dim)
{
    ProblemData *pd = new ProblemData(e0, rho0, gamma_func, v0, A0, B0);
    pd->SetViscosity(true); 
    pd->SetVorticity(true); 
    pd->SetMu(mu0); 
    pd->SetDimension(dim);
    
    if(dim == 3)
    {
        Array<int> ess_bdr_x({0,0,1,0,1,0});
        Array<int> ess_bdr_y({0,0,1,0,1,0});
        Array<int> ess_bdr_z({0,0,1,0,1,0});
        Array<real_t> bdr_val_x({0.0,0.0,uR,0.0,uL,0.0});
        Array<real_t> bdr_val_y({0.0,0.0,vR,0.0,vL,0.0});
        Array<real_t> bdr_val_z({0.0,0.0,0.0,0.0,0.0,0.0});
        pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y, ess_bdr_z, bdr_val_x, bdr_val_y, bdr_val_z);
        
        Array<int> ess_bdr_H({0,0,1,0,1,0});
        pd->SetHBoundaryCondition(ess_bdr_H, H0);
        
        pd->SetPeriodic(-1.0, size_yz, size_yz, new VectorFunctionCoefficient(3, PeriodicB0));
    }
    else if(dim == 2)
    {
        Array<int> ess_bdr_x({0,1,0,1});
        Array<int> ess_bdr_y({0,1,0,1});
        Array<real_t> bdr_val_x({0.0,uR,0.0,uL});
        Array<real_t> bdr_val_y({0.0,vR,0.0,vL});
        pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y, bdr_val_x, bdr_val_y);
        
        Array<int> ess_bdr_H({0,1,0,1});
        pd->SetHBoundaryCondition(ess_bdr_H, H0);
        
        pd->SetPeriodic(-1.0, size_yz, -1.0, new VectorFunctionCoefficient(2, PeriodicB0));
    }
    
    return pd;
}

}