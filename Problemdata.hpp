#ifndef PROBLEM_DATA_HPP
#define PROBLEM_DATA_HPP

#include "mfem.hpp"

namespace mfem
{

typedef std::function <void(const Vector &, Vector &)> Vfunc;
typedef std::function <void(const Vector &, real_t, Vector &)> TDVfunc;
typedef std::function <real_t(const Vector &)> Sfunc;
typedef std::function <real_t(const Vector &, real_t)> TDSfunc;

typedef enum Testcase_ 
{
   TAYLOR_GREEN = 0,
   MHD_ROTOR = 1,
   MHD_BLAST = 3,
   BrioWushocktube = 6,
   STATIC3D = 8,
   SuperFast = 11,
   MHDshocktube3 = 12,
    MHDshocktube4 = 13,
   PERIODIC_SHEAR = 14,
   PERIODIC_BOX = 15
} Testcase;

class ProblemData
{
protected: 
    
public:

    Sfunc e0;
    Sfunc rho0;
    Sfunc gamma_func;
    Vfunc v0;
    Vfunc A0;
    Vfunc B0;
    
    // source
    VectorCoefficient *u_source = nullptr;
    Coefficient *e_source = nullptr;
    
    bool visc;
    bool vorticity;
    real_t mu;
    
    int dim; // dimension of the problem
    
    // Boundary conditions
    /*
    3D: 
    1.z=0  2.y=0  3.x=1  4.y=1  5.x=0  6.z=1
    
    2D:
    1.y=0  2.x=1  3.y=1  4.x=0
    */
   
    // pressure boundary conditions
    Array<int> ess_bdr_p;
    TDSfunc p_bdry_func;
    
    // velocity boundary conditions
    Array<int> *ess_bdrs_v; 
    Array<real_t> *bdr_vals_v;
    
    Array<int> ess_bdr_H; // nxH boundary
    Vfunc H_bdry_func; // nxH boundary coefficient
    
    // Periodic: Background magnetic field
    bool periodic = false;
    real_t px_length = -1.0;
    real_t py_length = -1.0;
    real_t pz_length = -1.0;
    VectorCoefficient *B0_periodic = nullptr;
    
    //exact solution 
    bool has_exact_solution = false;
    TDSfunc exact_rho;
    TDVfunc exact_v;
    TDSfunc exact_e;
    TDVfunc exact_A;
    TDVfunc exact_B;
    
    
    ProblemData(Sfunc e0_, Sfunc rho0_, Sfunc gamma_func_, Vfunc v0_, Vfunc A0_, Vfunc B0_):
    e0(e0_), rho0(rho0_), gamma_func(gamma_func_), v0(v0_), A0(A0_), B0(B0_),visc(true),vorticity(false),mu(1.0),ess_bdrs_v(nullptr){}
    
    void SetViscosity(bool visc_)
    {
        visc = visc_;
    }
    
    void SetVorticity(bool vorticity_)
    {
        vorticity = vorticity_;
    }
    
    void SetMu(real_t mu_)
    {
        mu = mu_;
    }
    
    void SetDimension(int d)
    {
        dim = d;
        ess_bdrs_v = new Array<int>[dim];
        for (int i = 0; i < dim; i++)
        {
            ess_bdrs_v[i].SetSize(0);
        }
        bdr_vals_v = new Array<real_t>[dim];
        for (int i = 0; i < dim; i++)
        {
            bdr_vals_v[i].SetSize(0);
        }
    }
    
    void AddEssentialBoundaryCondition(Array<int> ess_bdr_x,
                                       Array<int> ess_bdr_y)
    {
        if (ess_bdr_x.Size() > 0) { ess_bdrs_v[0] = ess_bdr_x; }
        if (ess_bdr_y.Size() > 0) { ess_bdrs_v[1] = ess_bdr_y; }
        bdr_vals_v[0].SetSize(ess_bdrs_v[0].Size());
        bdr_vals_v[1].SetSize(ess_bdrs_v[1].Size());
        bdr_vals_v[0] = 0.0;
        bdr_vals_v[1] = 0.0;
    }
    
    void AddEssentialBoundaryCondition(Array<int> ess_bdr_x,
                                       Array<int> ess_bdr_y,
                                       Array<real_t> bdr_vals_x,
                                        Array<real_t> bdr_vals_y)
    {
        if (ess_bdr_x.Size() > 0) { ess_bdrs_v[0] = ess_bdr_x; }
        if (ess_bdr_y.Size() > 0) { ess_bdrs_v[1] = ess_bdr_y; }
        bdr_vals_v[0] = bdr_vals_x;
        bdr_vals_v[1] = bdr_vals_y;
    }
    
    void AddEssentialBoundaryCondition(Array<int> ess_bdr_x,
                                       Array<int> ess_bdr_y,
                                       Array<int> ess_bdr_z)
    {
        if (ess_bdr_x.Size() > 0) { ess_bdrs_v[0] = ess_bdr_x; }
        if (ess_bdr_y.Size() > 0) { ess_bdrs_v[1] = ess_bdr_y; }
        if (ess_bdr_z.Size() > 0) { ess_bdrs_v[2] = ess_bdr_z; }
        bdr_vals_v[0].SetSize(ess_bdrs_v[0].Size());
        bdr_vals_v[1].SetSize(ess_bdrs_v[1].Size());
        bdr_vals_v[2].SetSize(ess_bdrs_v[2].Size());
        bdr_vals_v[0] = 0.0;
        bdr_vals_v[1] = 0.0;
        bdr_vals_v[2] = 0.0;
    }
    
    void AddEssentialBoundaryCondition(Array<int> ess_bdr_x,
                                       Array<int> ess_bdr_y,
                                       Array<int> ess_bdr_z,
                                       Array<real_t> bdr_vals_x,
                                        Array<real_t> bdr_vals_y,
                                        Array<real_t> bdr_vals_z)
    {
        if (ess_bdr_x.Size() > 0) { ess_bdrs_v[0] = ess_bdr_x; }
        if (ess_bdr_y.Size() > 0) { ess_bdrs_v[1] = ess_bdr_y; }
        if (ess_bdr_z.Size() > 0) { ess_bdrs_v[2] = ess_bdr_z; }
        bdr_vals_v[0] = bdr_vals_x;
        bdr_vals_v[1] = bdr_vals_y;
        bdr_vals_v[2] = bdr_vals_z;
    }
    
    void SetHBoundaryCondition(Array<int> ess_bdr_H_, Vfunc H_bdry_func_)
    {
        ess_bdr_H = ess_bdr_H_;
        H_bdry_func = H_bdry_func_;
    }
    
    void SetPressureBoundaryCondition(Array<int> ess_bdr_p_, TDSfunc p_bdry_func_)
    {
        ess_bdr_p = ess_bdr_p_;
        p_bdry_func = p_bdry_func_;
    }
    
    void SetPeriodic(real_t px, real_t py, real_t pz, VectorCoefficient *B0_periodic_)
    {
        periodic = true;
        px_length = px;
        py_length = py;
        pz_length = pz;
        B0_periodic = B0_periodic_;
    }
    
    void SetExactSolution(TDSfunc exact_rho_, TDVfunc exact_v_, TDSfunc exact_e_, TDVfunc exact_A_, TDVfunc exact_B_)
    {
        has_exact_solution = true;
        exact_rho = exact_rho_;
        exact_v = exact_v_;
        exact_e = exact_e_;
        exact_A = exact_A_;
        exact_B = exact_B_;
    }
    
    void SetSource(VectorCoefficient *u_source_, Coefficient *e_source_)
    {
        u_source = u_source_;
        e_source = e_source_;
    }
    
    ~ProblemData(){
        if (ess_bdrs_v != nullptr)
        {
            delete[] ess_bdrs_v;
            ess_bdrs_v = nullptr;
        }
        if(u_source) delete u_source;
        if(e_source) delete u_source;
    };
};

ProblemData *GetProblemTaylorGreen(int dim);
ProblemData *GetProblemPeriodicShear(int dim);
void SetPeriodicShearParameters(real_t amplitude, real_t boost_y,
                               real_t boost_x, bool periodic_x);
ProblemData *GetProblemPeriodicBox(int dim);
void SetPeriodicBoxIC(const char *path);
ProblemData *GetProblemMHDrotor();
ProblemData *GetProblemMHDblast(real_t Bmag_);
ProblemData *GetProblemBrioWushocktube(int dim);
ProblemData *GetProblemStatic3D();
ProblemData *GetProblemSuperFast(int dim);
ProblemData *GetProblemMHDshocktube3(int dim);
ProblemData *GetProblemMHDshocktube4(int dim);


ProblemData *GetProblemData(Testcase p, int dim, real_t Bmag);


}


#endif // PROBLEM_DATA_HPP
