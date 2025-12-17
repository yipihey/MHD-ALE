

#include "mesh_smoother.hpp"
#include "tools.hpp"
#include "Interpolator.hpp"

namespace mfem
{
    
void IdentitySmoother::Smooth(ParGridFunction &newnodes)
{
    ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
    newnodes = *nodes;
}

InitialSmoother::InitialSmoother(ParMesh &pmesh_): MeshSmoother(pmesh_)
{
    ParGridFunction *pmesh_nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
    nodes.SetSpace(pmesh_nodes->ParFESpace());
    nodes = *pmesh_nodes;
}

void InitialSmoother::Smooth(ParGridFunction &newnodes)
{
    newnodes = nodes;
}

LimitedHarmonicSmoother::LimitedHarmonicSmoother(ParMesh &pmesh_, real_t epsilon_)
    : MeshSmoother(pmesh_), 
    fes(nullptr),
    Laplace_bf(nullptr), 
    mass_bf(nullptr),
    cg_solver(nullptr),
    amg_prec(nullptr),
    epsilon(epsilon_)
{
    pmesh->EnsureNodes();
    ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
    fes = dynamic_cast<ParFiniteElementSpace *>(nodes->ParFESpace());
    
    ConstantCoefficient eps_coeff(epsilon);
    
    Laplace_bf = new ParBilinearForm(fes);
    Laplace_bf->AddDomainIntegrator(new VectorMassIntegrator);
    Laplace_bf->AddDomainIntegrator(new VectorDiffusionIntegrator(eps_coeff));
    Laplace_bf->Assemble();
    Laplace_bf->Finalize();
    
    mass_bf = new ParBilinearForm(fes);
    mass_bf->AddDomainIntegrator(new VectorMassIntegrator);
    mass_bf->Assemble();
    mass_bf->Finalize();
    
    ess_bdr.SetSize(pmesh->bdr_attributes.Max());
    ess_bdr = 1;
    fes->GetEssentialTrueDofs(ess_bdr, ess_tdofs);
    
}
    
LimitedHarmonicSmoother::~LimitedHarmonicSmoother() {
    if (Laplace_bf) { delete Laplace_bf; Laplace_bf = nullptr; }
    if (mass_bf) { delete mass_bf; mass_bf = nullptr; }
    if (cg_solver) { delete cg_solver; cg_solver = nullptr; }
    if (amg_prec) { delete amg_prec; amg_prec = nullptr; }
}
    
void LimitedHarmonicSmoother::Smooth(ParGridFunction &newnodes)
{
    ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
    
    newnodes.SetSpace(fes);
        
    OperatorPtr A;
    Vector X, B;
    
    ParLinearForm lf(fes);
    lf.Assemble();
    mass_bf->AddMult(*nodes, lf);
    
    Laplace_bf->FormLinearSystem(ess_tdofs, *nodes, lf, A, X, B);
    
    if (!cg_solver)
    {
        cg_solver = new CGSolver(MPI_COMM_WORLD);
        amg_prec = new HypreBoomerAMG;
        amg_prec->SetPrintLevel(0);
        cg_solver->SetPreconditioner(*amg_prec);
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(100);
        cg_solver->SetAbsTol(1e-15);
        cg_solver->SetPrintLevel(3);
    }
    cg_solver->SetOperator(*A);
    cg_solver->Mult(B, X);
    
    Laplace_bf->RecoverFEMSolution(X, lf, newnodes);
}


PeriodicSmoother::PeriodicSmoother(ParMesh &pmesh_, real_t epsilon_)
    : MeshSmoother(pmesh_), 
    fes(nullptr),
    L2_fes(nullptr),
    H1_fec(nullptr),
    H1_fes(nullptr),
    Laplace_bf(nullptr), 
    mass_bf(nullptr),
    cg_solver(nullptr),
    amg_prec(nullptr),
    epsilon(epsilon_)
{
    pmesh->EnsureNodes();
    ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
    fes = dynamic_cast<ParFiniteElementSpace *>(nodes->ParFESpace());
    
    MFEM_VERIFY(fes->IsDGSpace(),
                "PeriodicSmoother: Finite element space is not discontinuous.");
                
    MFEM_VERIFY(fes->GetOrdering() == Ordering::byNODES,
                "PeriodicSmoother: Finite element space ordering is not byNODES.");
                
    initial_nodes.SetSize(fes->GetVSize());
    initial_nodes = *nodes;
    
    L2_fes = new ParFiniteElementSpace(pmesh, pmesh->GetNodalFESpace()->FEColl());
    H1_fec = new H1_FECollection(fes->GetOrder(0), fes->GetMesh()->Dimension());
    H1_fes = new ParFiniteElementSpace(pmesh, H1_fec);
    
    ConstantCoefficient eps_coeff(epsilon);
    
    Laplace_bf = new ParBilinearForm(H1_fes);
    Laplace_bf->AddDomainIntegrator(new MassIntegrator);
    Laplace_bf->AddDomainIntegrator(new DiffusionIntegrator(eps_coeff));
    Laplace_bf->Assemble();
    Laplace_bf->Finalize();
    
    mass_bf = new ParBilinearForm(H1_fes);
    mass_bf->AddDomainIntegrator(new MassIntegrator);
    mass_bf->Assemble();
    mass_bf->Finalize();
    
    if(pmesh->Dimension() == 3)
    {
        ess_bdr.SetSize(6);
        ess_bdr = 0;
        ess_bdr[2] = 1; 
        ess_bdr[4] = 1; 
    }
    else if(pmesh->Dimension() == 2)
    {
        ess_bdr.SetSize(4);
        ess_bdr = 0;
        ess_bdr[1] = 1; 
        ess_bdr[3] = 1; 
    }
    else
    {
        MFEM_ABORT("PeriodicSmoother: only 2D and 3D meshes are supported.");
    }
    H1_fes->GetEssentialTrueDofs(ess_bdr, ess_tdofs);
    
}
    
PeriodicSmoother::~PeriodicSmoother() {
    if (Laplace_bf) { delete Laplace_bf; Laplace_bf = nullptr; }
    if (mass_bf) { delete mass_bf; mass_bf = nullptr; }
    if (cg_solver) { delete cg_solver; cg_solver = nullptr; }
    if (amg_prec) { delete amg_prec; amg_prec = nullptr; }
    if( H1_fes ) { delete H1_fes; H1_fes = nullptr; }
    if( H1_fec ) { delete H1_fec; H1_fec = nullptr; }
    if( L2_fes ) { delete L2_fes; L2_fes = nullptr; }
}
    
void PeriodicSmoother::Smooth(ParGridFunction &newnodes)
{
    ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
    newnodes.SetSpace(fes);
    newnodes.Update();
    newnodes = *nodes;
    
    ParGridFunction nodes_x;
    nodes_x.MakeRef(L2_fes, *nodes, 0);
    
    ParGridFunction H1_x(H1_fes);
    H1_x.ProjectGridFunction(nodes_x);
        
    OperatorPtr A;
    Vector X, B;
    
    ParLinearForm lf(H1_fes);
    lf.Assemble();
    mass_bf->AddMult(H1_x, lf);
    
    Laplace_bf->FormLinearSystem(ess_tdofs, H1_x, lf, A, X, B);
    
    if (!cg_solver)
    {
        cg_solver = new CGSolver(MPI_COMM_WORLD);
        amg_prec = new HypreBoomerAMG;
        amg_prec->SetPrintLevel(0);
        cg_solver->SetPreconditioner(*amg_prec);
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(100);
        cg_solver->SetAbsTol(1e-15);
        cg_solver->SetPrintLevel(3);
    }
    cg_solver->SetOperator(*A);
    cg_solver->Mult(B, X);
    
    Laplace_bf->RecoverFEMSolution(X, lf, H1_x);
    
    ParGridFunction newnodes_x;
    newnodes_x.MakeRef(L2_fes, newnodes, 0);
    newnodes_x.ProjectGridFunction(H1_x);
    
    ParGridFunction initial_nodes_yz, newnodes_yz;
    initial_nodes_yz.MakeRef(L2_fes, initial_nodes, L2_fes->GetNDofs());
    newnodes_yz.MakeRef(L2_fes, newnodes, L2_fes->GetNDofs());
    newnodes_yz = initial_nodes_yz;
    if( pmesh->Dimension() == 3 )
    {
        initial_nodes_yz.MakeRef(L2_fes, initial_nodes, 2*L2_fes->GetNDofs());
        newnodes_yz.MakeRef(L2_fes, newnodes, 2*L2_fes->GetNDofs());
        newnodes_yz = initial_nodes_yz;
    }
    
}

real_t PeriodicSmoother::MaxDisplacement()
{
    real_t max_disp = 0.0;
    
    ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
    ParGridFunction initial_nodes_yz;
    ParGridFunction current_nodes_yz;
    Vector disp_vec;
    
    for(int d = 1; d < pmesh->Dimension(); d++)
    {
        initial_nodes_yz.MakeRef(L2_fes, initial_nodes, L2_fes->GetNDofs()*d);
        current_nodes_yz.MakeRef(L2_fes, *nodes, L2_fes->GetNDofs()*d);
        
        disp_vec.SetSize(current_nodes_yz.Size());
        disp_vec = 0.0;
        disp_vec = current_nodes_yz;
        disp_vec -= initial_nodes_yz;
        
        for(int i = 0; i < current_nodes_yz.Size(); i++)
        {
            real_t disp = std::abs(disp_vec(i));
            if(disp > max_disp)
            {
                max_disp = disp;
            }
        }
    }
    
    MPI_Allreduce(MPI_IN_PLACE, &max_disp, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
    
    return max_disp;
}



} // namespace mfem