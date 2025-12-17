#ifndef MESH_SMOOTHER_HPP
#define MESH_SMOOTHER_HPP

#include "mfem.hpp"

namespace mfem
{
    
class MeshSmoother
{

protected:

    ParMesh *pmesh; // initial mesh, reference mesh
    
public:

    MeshSmoother(ParMesh & pmesh): pmesh(&pmesh) { }
    
    virtual ~MeshSmoother(){}
    
    virtual void Smooth(ParGridFunction &newnodes) = 0;
};

// just do nothing
class IdentitySmoother: public MeshSmoother
{
protected:

public:

    IdentitySmoother(ParMesh & pmesh):MeshSmoother(pmesh){};
    
    virtual ~IdentitySmoother(){}
    
    void Smooth(ParGridFunction &newnodes) override;
};

class InitialSmoother: public MeshSmoother
{

protected:

    ParGridFunction nodes;
    
public:

    InitialSmoother(ParMesh & pmesh_);
    
    virtual ~InitialSmoother() {}
    
    void Smooth(ParGridFunction &newnodes) override;
};

class LimitedHarmonicSmoother: public MeshSmoother
{

protected:

    ParFiniteElementSpace *fes; // finite element space on the reference mesh
    
    ParBilinearForm *Laplace_bf;
    ParBilinearForm *mass_bf;
    Array<int> ess_bdr;
    Array<int> ess_tdofs;
    CGSolver *cg_solver;
    HypreBoomerAMG *amg_prec;
    
    real_t epsilon;
    
public:

    LimitedHarmonicSmoother(ParMesh & pmesh, real_t epsilon_ = 1e-4);
    
    virtual ~LimitedHarmonicSmoother();
    
    void Smooth(ParGridFunction &newnodes) override;
};


// PeriodicSmoother: only perform limited harmonic smoothing on x-direction
// set the other directions as initial positions
class PeriodicSmoother: public MeshSmoother
{

protected:

    ParFiniteElementSpace *fes; // finite element space on the reference mesh
    Vector initial_nodes;
    
    // scalar FE space for x-direction
    ParFiniteElementSpace *L2_fes;
    H1_FECollection *H1_fec;
    ParFiniteElementSpace *H1_fes;
    
    ParBilinearForm *Laplace_bf;
    ParBilinearForm *mass_bf;
    Array<int> ess_bdr;
    Array<int> ess_tdofs;
    CGSolver *cg_solver;
    HypreBoomerAMG *amg_prec;
    
    real_t epsilon;
    
public:

    PeriodicSmoother(ParMesh & pmesh, real_t epsilon_ = 1e-4);
    
    virtual ~PeriodicSmoother();
    
    void Smooth(ParGridFunction &newnodes) override;
    
    // Compute maximum displacement in y,z directions
    real_t MaxDisplacement();
};
    
}


#endif // MESH_SMOOTHER_HPP 