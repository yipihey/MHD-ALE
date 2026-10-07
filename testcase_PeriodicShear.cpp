#include "Problemdata.hpp"

namespace mfem
{
// Weak-field shear: v_y = amplitude*sin(pi*x)^2 + boost_y.
// Without backreaction, E_B/E_B(0) = 1 + amplitude^2*pi^2*t^2/2.
static real_t shear_amplitude = 1.0, boost_y = 0.0, boost_x = 0.0;
static bool periodic_x = false;

void SetPeriodicShearParameters(real_t amplitude, real_t uy, real_t ux,
                               bool px)
{
    shear_amplitude = amplitude;
    boost_y = uy;
    boost_x = ux;
    periodic_x = px;
}

static real_t density(const Vector &) { return 1.0; }
static real_t gamma(const Vector &) { return 5.0/3.0; }
static real_t energy(const Vector &) { return 100.0/(5.0/3.0-1.0); }
static void velocity(const Vector &x, Vector &v)
{
    v = 0.0;
    v(0) = boost_x;
    v(1) = shear_amplitude*pow(sin(M_PI*x(0)),2.0) + boost_y;
}
static void potential(const Vector &, Vector &a) { a = 0.0; }
static void magnetic(const Vector &, Vector &b) { b = 0.0; b(0) = 1e-3; }

ProblemData *GetProblemPeriodicShear(int dim)
{
    MFEM_VERIFY(dim == 2 || dim == 3, "Periodic shear requires dimension 2 or 3");
    MFEM_VERIFY(periodic_x || boost_x == 0.0, "An x boost requires -spx");
    auto *pd = new ProblemData(energy,density,gamma,velocity,potential,magnetic);
    pd->SetDimension(dim);
    pd->SetMu(1.0);
    pd->SetViscosity(false);
    pd->SetVorticity(false);
    Array<int> walls(dim == 2 ? 4 : 6);
    walls = 0;
    if (!periodic_x)
    {
        if (dim == 2) { walls[1] = walls[3] = 1; }
        else { walls[2] = walls[4] = 1; }
    }
    Array<real_t> zero(walls.Size()), vy(walls.Size());
    zero = 0.0;
    vy = boost_y;
    if (dim == 2) { pd->AddEssentialBoundaryCondition(walls,walls,zero,vy); }
    else { pd->AddEssentialBoundaryCondition(walls,walls,walls,zero,vy,zero); }
    pd->SetHBoundaryCondition(walls,magnetic);
    pd->SetPeriodic(periodic_x ? 1.0 : -1.0, 1.0, dim == 3 ? 1.0 : -1.0,
                    new VectorFunctionCoefficient(dim,magnetic));
    return pd;
}
}
