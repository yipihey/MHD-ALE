#include "Problemdata.hpp"
#include "box_ic.hpp"
#include <memory>

namespace mfem
{
static std::unique_ptr<PeriodicBoxIC> ic;
void SetPeriodicBoxIC(const char *path) { ic.reset(new PeriodicBoxIC(path)); }
static real_t density(const Vector &) { return 1.0; }
static real_t gamma(const Vector &) { return ic->gamma; }
static real_t energy(const Vector &) { return ic->pressure/(ic->gamma-1.0); }
static void velocity(const Vector &x, Vector &v)
{
    auto value = ic->velocity({{x(0),x(1),x.Size() == 3 ? x(2) : 0.0}});
    for (int d=0; d<v.Size(); ++d) { v(d) = value[d]; }
}
static void potential(const Vector &x, Vector &a)
{
    auto value = ic->potential({{x(0),x(1),x.Size() == 3 ? x(2) : 0.0}});
    if (a.Size() == 1) { a(0) = value[2]; }
    else { for (int d=0; d<a.Size(); ++d) { a(d) = value[d]; } }
}
static void magnetic(const Vector &, Vector &b)
{
    for (int d=0; d<b.Size(); ++d) { b(d) = ic->magnetic[d]; }
}

ProblemData *GetProblemPeriodicBox(int dim)
{
    MFEM_VERIFY(ic && (dim == 2 || dim == 3), "Periodic box IC not initialized");
    if (dim == 2)
    {
        MFEM_VERIFY(ic->boost[2] == 0.0 && ic->magnetic[2] == 0.0,
                    "2D box requires in-plane velocity and magnetic field");
        for (const auto *list : {&ic->modes,&ic->magnetic_modes})
            for (const auto &mode : *list)
                MFEM_VERIFY(mode.n[2] == 0 && mode.a[2] == 0.0 && mode.b[2] == 0.0,
                            "2D box requires in-plane Fourier modes");
    }
    auto *pd = new ProblemData(energy,density,gamma,velocity,potential,magnetic);
    pd->SetDimension(dim);
    pd->SetMu(1.0);
    pd->SetViscosity(false);
    pd->SetVorticity(false);
    Array<int> none(dim == 2 ? 4 : 6);
    none = 0;
    if (dim == 2) { pd->AddEssentialBoundaryCondition(none,none); }
    else { pd->AddEssentialBoundaryCondition(none,none,none); }
    pd->SetHBoundaryCondition(none,magnetic);
    pd->SetPeriodic(1.0,1.0,dim == 3 ? 1.0 : -1.0,
                    new VectorFunctionCoefficient(dim,magnetic));
    return pd;
}
}
