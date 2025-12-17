#include "Problemdata.hpp"

namespace mfem
{
    
static real_t r00 = 0.1; 
static real_t r11 = 0.115;
static real_t u00 = 2.0;
static real_t mu0 = 4.0*M_PI; 

static inline real_t ff(real_t r)
{
    return (r11 - r) / (r11 - r00);
}

static inline real_t rr(const Vector &x)
{
    return sqrt(x(0)*x(0) + x(1)*x(1));
}

static real_t rho0(const Vector &x)
{
    real_t r = rr(x);
    real_t rho = -1.0;
    if (r <= r00)
    {
        rho = 10.0;
    }
    else if (r >= r11)
    {
        rho = 1.0;
    }
    else
    {
        rho = 1.0 + 9.0 * ff(r);
    }
    if(rho < 0.0)
    {
        mfem::out << "Negative density at rho0" << std::endl;
        MFEM_ABORT("Negative density");
    }
    return rho;
}

static real_t gamma_func(const Vector &x)
{
    return 1.4;
}

static void v0(const Vector &x, Vector &v)
{
    real_t r = rr(x);
    if(r >= r11)
    {
        v(0) = 0.0;
        v(1) = 0.0;
    }
    else
    {
        real_t factor = ff(r) * u00 / fmax(r, r00);
        v(0) = factor * (-x(1));
        v(1) = factor * (x(0));
    }
}

static void A0(const Vector &x, Vector &A)
{
    A(0) = 5.0/2.0*sqrt(mu0/M_PI)*x(1);
}

static void B0(const Vector &x, Vector &B)
{
    B(0) = 5.0/2.0*sqrt(mu0/M_PI);
    B(1) = 0.0;
}

static void H0(const Vector &x, Vector &H)
{
    B0(x, H);
    H /= mu0; // H = B/mu
}

// p = (gamma-1)\rho e
// e = p/(gamma-1)/rho
static real_t e0(const Vector &x)
{
    real_t p = 1.0; // constant pressure
    real_t rho = rho0(x);
    return p / (gamma_func(x) - 1.0) / rho;
}

ProblemData *GetProblemMHDrotor()
{
    ProblemData *pd = new ProblemData(e0, rho0, gamma_func, v0, A0, B0);
    pd->SetViscosity(true); 
    pd->SetVorticity(true); 
    pd->SetMu(mu0); // magnetic permeability
    pd->SetDimension(2);
    Array<int> ess_bdr_x({0,1,0,1});
    Array<int> ess_bdr_y({1,0,1,0});
    pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y);
    Array<int> ess_bdr_H({1,1,1,1});
    pd->SetHBoundaryCondition(ess_bdr_H, H0);
    return pd;
}

}