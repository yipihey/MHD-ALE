#include "Problemdata.hpp"

namespace mfem
{

static real_t beta = 0.5; // const for Taylor-Green vortex
static real_t mu0 = 4.0*M_PI;

static real_t rho_exact(const Vector &x, real_t t)
{
    return 1.0;
}

static real_t rho0(const Vector &x)
{
    return rho_exact(x, 0.0);
}

static real_t gamma_func(const Vector &x)
{
    return 5.0 / 3.0;
}

static void v_exact(const Vector &x, real_t t, Vector &v)
{
    v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
    v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
    if (x.Size() == 3)
    {
    v(2) = 0.0;
    }
}

static void v0(const Vector &x, Vector &v)
{
    v_exact(x, 0.0, v);
}

static void A_exact(const Vector &x, real_t t, Vector &A)
{
    if(x.Size()==2)
    {
    A(0) = 1.0/M_PI*sin(M_PI*x(0))*sin(M_PI*x(1))*sqrt(4.0*M_PI)*beta;
    }
    else if(x.Size()==3)
    {
    A(0) = 0.0;
    A(1) = 0.0;
    A(2) = 1.0/M_PI*sin(M_PI*x(0))*sin(M_PI*x(1))*sqrt(4.0*M_PI)*beta;
    }
}

static void A0(const Vector &x, Vector &A)
{
    A_exact(x, 0.0, A);
}

static void B_exact(const Vector &x, real_t t, Vector &B)
{
    B(0) = sin(M_PI*x(0))*cos(M_PI*x(1))*sqrt(4.0*M_PI)*beta;
    B(1) = -cos(M_PI*x(0))*sin(M_PI*x(1))*sqrt(4.0*M_PI)*beta;
    if(x.Size()==3)
    {
    B(2) = 0.0;
    }
}


static void B0(const Vector &x, Vector &B)
{
    B_exact(x, 0.0, B);
}

static void H0(const Vector &x, Vector &H)
{
    B0(x, H);
    H /= mu0; 
}

static real_t e_exact(const Vector &x, real_t t)
{
    const real_t denom = 2.0 / 3.0;  // (5/3 - 1) * density.
    real_t val;
    val = 1.0 
    + (1 - beta*beta)/4*(cos(2*M_PI*x(0))+cos(2*M_PI*x(1))) 
    - beta * beta / 2.0 * (sin(M_PI*x(0))*sin(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(1)) + cos(M_PI*x(0))*cos(M_PI*x(0))*sin(M_PI*x(1))*sin(M_PI*x(1)));
    return val/denom;
}


static real_t e0(const Vector &x)
{
    return e_exact(x, 0.0);
}


static void Lorentz_force(const Vector &xi, real_t t, Vector &f)
{
    real_t x,y;
    x = xi(0);
    y = xi(1);
    f[0] = M_PI*(beta*beta)*cos(M_PI*x)*sin(M_PI*x)*pow(sin(M_PI*y),2.0)*2.0;
    f[1] = M_PI*(beta*beta)*cos(M_PI*y)*pow(sin(M_PI*x),2.0)*sin(M_PI*y)*2.0;
    if(xi.Size() == 3)
    {
        f[2] = 0.0;
    }
}


// TaylorCoefficient used in the 2D Taylor-Green problem.
class TaylorCoefficient : public Coefficient
{
public:
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(2);
      T.Transform(ip, x);
      return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                                  cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
   }
};


ProblemData *GetProblemTaylorGreen(int dim)
{
    ProblemData *pd = new ProblemData(e0, rho0, gamma_func, v0, A0, B0);
    pd->SetSource(nullptr, new TaylorCoefficient);
    pd->SetViscosity(false); // no viscosity
    pd->SetVorticity(false); // no vorticity
    pd->SetMu(mu0); // magnetic permeability
    pd->SetDimension(dim);
    if(dim == 2)
    {
        Array<int> ess_bdr_x({0,1,0,1});
        Array<int> ess_bdr_y({1,0,1,0});
        pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y);
        
        Array<int> ess_bdr_H({1,1,1,1});
        pd->SetHBoundaryCondition(ess_bdr_H, H0);
    }
    else if (dim==3)
    {
        Array<int> ess_bdr_x({0,0,1,0,1,0});
        Array<int> ess_bdr_y({0,1,0,1,0,0});
        Array<int> ess_bdr_z({1,0,0,0,0,1});
        pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y, ess_bdr_z);
        
        Array<int> ess_bdr_H({1,1,1,1,1,1});
        pd->SetHBoundaryCondition(ess_bdr_H, H0);
    }
    
    pd->SetExactSolution(rho_exact, v_exact, e_exact, A_exact, B_exact);
    
    return pd;
}

}