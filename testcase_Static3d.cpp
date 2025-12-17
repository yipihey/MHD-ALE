#include "Problemdata.hpp"

namespace mfem
{

static real_t beta_ = 0.5;
static real_t UMAG = 1024.0;
static const real_t mu0 = beta_*beta_;

static void phi_exact(const Vector &xi, real_t t, Vector &phi)
{
    real_t x = xi(0);
    real_t y = xi(1);
    real_t z = xi(2);
        
    phi[0] = UMAG*(x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0);
    phi[1] = -UMAG*(x*x*x)*(y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0);
    phi[2] = UMAG*(x*x)*(y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0);

}

static void curlphi_exact(const Vector &xi, real_t t, Vector &A)
{
    real_t x = xi(0);
    real_t y = xi(1);
    real_t z = xi(2);
    
    A[0] = UMAG*(x*x)*y*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*2.0+UMAG*(x*x*x)*(y*y)*z*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*2.0+UMAG*(x*x)*(y*y)*(z*z)*(y*2.0-2.0)*pow(x-1.0,2.0)*pow(z-1.0,2.0)+UMAG*(x*x*x)*(y*y)*(z*z)*(z*2.0-2.0)*pow(x-1.0,2.0)*pow(y-1.0,2.0);
    A[1] = UMAG*x*(y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*-2.0+UMAG*(x*x)*(y*y*y)*z*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*2.0-UMAG*(x*x)*(y*y)*(z*z)*(x*2.0-2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)+UMAG*(x*x)*(y*y*y)*(z*z)*(z*2.0-2.0)*pow(x-1.0,2.0)*pow(y-1.0,2.0);
    A[2] = -UMAG*(x*x)*(y*y*y)*(z*z)*(y*2.0-2.0)*pow(x-1.0,2.0)*pow(z-1.0,2.0)-UMAG*(x*x*x)*(y*y)*(z*z)*(x*2.0-2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)-UMAG*(x*x)*(y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*6.0;

}

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
    curlphi_exact(x, t, v);
}

static void v0(const Vector &x, Vector &v)
{
    v_exact(x, 0.0, v);
}

static void A_exact(const Vector &xi, real_t t, Vector &A)
{
    phi_exact(xi, 0.0, A);
    A[0] *= beta_;
    A[1] *= beta_;
    A[2] *= beta_;
}

static void A0(const Vector &xi, Vector &A)
{
    A_exact(xi, 0.0, A);
}

static void B_exact(const Vector &xi, real_t t, Vector &B)
{
    curlphi_exact(xi, t, B);
    B[0] *= beta_;
    B[1] *= beta_;
    B[2] *= beta_;
}

static void B0(const Vector &xi, Vector &B)
{
    B_exact(xi, 0.0, B);
}

static void H0(const Vector &x, Vector &H)
{
    B0(x, H);
    H /= mu0; 
}


static real_t e_exact(const Vector &xi, real_t t)
{
    real_t x = xi(0);
    real_t y = xi(1);
    real_t z = xi(2);
    
    real_t e = (UMAG*UMAG)*(x*x*x*x)*(y*y*y*y)*(z*z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,4.0)*pow(x*4.0+y*4.0-x*y*5.0-3.0,2.0)*-3.0-(UMAG*UMAG)*(x*x)*(y*y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,4.0)*pow(z-1.0,2.0)*pow(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0,2.0)*3.0-(UMAG*UMAG)*(x*x*x*x)*(y*y)*(z*z)*pow(x-1.0,4.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*pow(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0,2.0)*3.0+1.5E+1;

    return e;
}


static real_t e0(const Vector &xi)
{
    return e_exact(xi, 0.0);
}

static real_t e_source(const Vector &xi, real_t t)
{
    real_t x = xi(0);
    real_t y = xi(1);
    real_t z = xi(2);
    return UMAG*(x*x)*(y*y)*(z*z)*(x-1.0)*(y-1.0)*pow(z-1.0,2.0)*((UMAG*UMAG)*(x*x*x*x)*(y*y*y*y)*(z*z*z)*pow(x-1.0,2.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z*-3.0+(z*z)*2.0+1.0)*pow(x*4.0+y*4.0-x*y*5.0-3.0,2.0)*-1.2E+1+(UMAG*UMAG)*(x*x)*(y*y*y*y)*z*pow(x-1.0,2.0)*pow(y-1.0,4.0)*(z-1.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*(z*2.0-x*y-x*z*4.0+(x*x)*y+x*(z*z)*1.2E+1-x*(z*z*z)*8.0-(z*z)*6.0+(z*z*z)*4.0-x*y*(z*z)*6.0-(x*x)*y*z*6.0+(x*x)*y*(z*z)*6.0+x*y*z*6.0)*6.0+(UMAG*UMAG)*(x*x*x*x)*(y*y)*z*pow(x-1.0,4.0)*pow(y-1.0,2.0)*(z-1.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*(z*2.0+x*y-y*z*4.0-x*(y*y)+y*(z*z)*1.2E+1-y*(z*z*z)*8.0-(z*z)*6.0+(z*z*z)*4.0+x*y*(z*z)*6.0+x*(y*y)*z*6.0-x*(y*y)*(z*z)*6.0-x*y*z*6.0)*6.0)*(x*4.0+y*4.0-x*y*5.0-3.0)*2.0+UMAG*(x*x)*y*z*pow(x-1.0,2.0)*(y-1.0)*(z-1.0)*((UMAG*UMAG)*x*(y*y*y*y)*(z*z)*(x-1.0)*pow(y-1.0,4.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*(z-(x*x)*(z*z)*6.0-x*y*2.0-x*z*6.0+(x*x)*y*6.0-(x*x*x)*y*4.0+x*(z*z)*6.0+(x*x)*z*6.0-z*z-(x*x)*y*z*1.2E+1+(x*x*x)*y*z*8.0+x*y*z*4.0)*6.0+(UMAG*UMAG)*(x*x*x)*(y*y)*(z*z)*pow(x-1.0,3.0)*pow(y-1.0,2.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*(z*2.0+(x*x)*(y*y)*5.0+x*y*3.0-x*z*4.0-y*z*4.0-x*(y*y)*3.0-(x*x)*y*5.0+x*(z*z)*4.0+y*(z*z)*4.0-(z*z)*2.0-x*y*(z*z)*8.0+x*(y*y)*z*6.0+(x*x)*y*z*1.0E+1-(x*x)*(y*y)*z*1.0E+1+x*y*z*2.0)*6.0+(UMAG*UMAG)*(x*x*x)*(y*y*y*y)*(z*z*z*z)*(x-1.0)*pow(y-1.0,2.0)*pow(z-1.0,4.0)*(x*4.0+y*4.0-x*y*5.0-3.0)*(x*2.1E+1+y*8.0-x*y*2.7E+1+(x*x)*y*2.0E+1-(x*x)*1.6E+1-6.0)*6.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*2.0-UMAG*x*(y*y)*z*(x-1.0)*pow(y-1.0,2.0)*(z-1.0)*((UMAG*UMAG)*(x*x*x*x)*y*(z*z)*pow(x-1.0,4.0)*(y-1.0)*pow(z-1.0,2.0)*(z+x*y-y*z*2.0-x*(y*y)+y*(z*z)*2.0-z*z+x*(y*y)*z*2.0-x*y*z*2.0)*(z-(y*y)*(z*z)*6.0+x*y*2.0-y*z*6.0-x*(y*y)*6.0+x*(y*y*y)*4.0+y*(z*z)*6.0+(y*y)*z*6.0-z*z+x*(y*y)*z*1.2E+1-x*(y*y*y)*z*8.0-x*y*z*4.0)*6.0-(UMAG*UMAG)*(x*x)*(y*y*y)*(z*z)*pow(x-1.0,2.0)*pow(y-1.0,3.0)*pow(z-1.0,2.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*(z*-2.0+(x*x)*(y*y)*5.0+x*y*3.0+x*z*4.0+y*z*4.0-x*(y*y)*5.0-(x*x)*y*3.0-x*(z*z)*4.0-y*(z*z)*4.0+(z*z)*2.0+x*y*(z*z)*8.0+x*(y*y)*z*1.0E+1+(x*x)*y*z*6.0-(x*x)*(y*y)*z*1.0E+1-x*y*z*1.4E+1)*6.0+(UMAG*UMAG)*(x*x*x*x)*(y*y*y)*(z*z*z*z)*pow(x-1.0,2.0)*(y-1.0)*pow(z-1.0,4.0)*(x*4.0+y*4.0-x*y*5.0-3.0)*(x*8.0+y*2.1E+1-x*y*2.7E+1+x*(y*y)*2.0E+1-(y*y)*1.6E+1-6.0)*6.0)*(z-x*y-x*z*2.0+(x*x)*y+x*(z*z)*2.0-z*z-(x*x)*y*z*2.0+x*y*z*2.0)*2.0;
}

ProblemData *GetProblemStatic3D()
{
    ProblemData *pd = new ProblemData(e0, rho0, gamma_func, v0, A0, B0);
    pd->SetSource(nullptr, new FunctionCoefficient(e_source));
    pd->SetViscosity(false); 
    pd->SetVorticity(false); 
    pd->SetMu(mu0); // magnetic permeability
    pd->SetDimension(3);
    
    Array<int> ess_bdr_x({0,0,1,0,1,0});
    Array<int> ess_bdr_y({0,1,0,1,0,0});
    Array<int> ess_bdr_z({1,0,0,0,0,1});
    pd->AddEssentialBoundaryCondition(ess_bdr_x, ess_bdr_y, ess_bdr_z);
    
    Array<int> ess_bdr_H({1,1,1,1,1,1});
    pd->SetHBoundaryCondition(ess_bdr_H, H0);
    
    pd->SetExactSolution(rho_exact, v_exact, e_exact, A_exact, B_exact);
        
    return pd;
}

}