#include "remap.hpp"
#include "tools.hpp"
#include "Integrators.hpp"

namespace mfem
{
    
    void ExactRemap::Remap(ParGridFunction &mesh_velocity, ParGridFunction &gf)
    {
        ParFiniteElementSpace *fes = gf.ParFESpace();
        ParMesh *pmesh = fes->GetParMesh();

        ParMesh *pmesh_new = new ParMesh(*pmesh);
        ParGridFunction *nodes_new = dynamic_cast<ParGridFunction *>(pmesh_new->GetNodes());
        nodes_new->Add(-1.0, mesh_velocity);
        pmesh_new->NodesUpdated();

        ParFiniteElementSpace *fes_new = new ParFiniteElementSpace(pmesh_new, fes->FEColl(), fes->GetVDim());

        ParGridFunction gf_new(fes_new);
        
        if(exact_coeff)
        {
            if(Bernstein && integral)
            {
                const IntegrationRule &ir = IntRules.Get(fes_new->GetFE(0)->GetGeomType(), fes_new->GetOrder(0) * 2 +3);
                ProjectDensity(gf_new, *exact_coeff, ir);
            }
            else if (Bernstein)
            {
                L2_FECollection l2_fec(fes_new->GetOrder(0), fes_new->GetMesh()->Dimension());
                ParFiniteElementSpace l2_fes(pmesh_new, &l2_fec, fes_new->GetVDim());
                ParGridFunction l2_gf(&l2_fes);
                l2_gf.ProjectCoefficient(*exact_coeff);
                gf_new.ProjectGridFunction(l2_gf);
            }
            else
            {
                gf_new.ProjectCoefficient(*exact_coeff);
            }
        }
        else if(exact_vec_coeff)
        {
            gf_new.ProjectCoefficient(*exact_vec_coeff);
        }
        else
        {
            mfem_error("ExactRemap: Exact solution have not been set. ");
        }

        gf = gf_new;
        
        delete fes_new;
        delete pmesh_new;
    }

    void InterpolateRemap::Remap(ParGridFunction &mesh_velocity, ParGridFunction &gf)
    {
        ParFiniteElementSpace *fes = gf.ParFESpace();
        ParMesh *pmesh = fes->GetParMesh();

        ParMesh *pmesh_new = new ParMesh(*pmesh);
        ParGridFunction *nodes_new = dynamic_cast<ParGridFunction *>(pmesh_new->GetNodes());
        nodes_new->Add(-1.0, mesh_velocity);
        pmesh_new->NodesUpdated();

        ParFiniteElementSpace *fes_new = new ParFiniteElementSpace(pmesh_new, fes->FEColl(), fes->GetVDim());

        Interpolator interp(*fes, *fes_new, 1e-4);

        ParGridFunction gf_new(fes_new);
        interp.Interpolate(gf, gf_new);

        gf = gf_new;
        
        delete fes_new;
        delete pmesh_new;
    }
    
    void L2ProjectRemap::Remap(ParGridFunction &mesh_velocity, ParGridFunction &gf)
    {
        
        if(!ir)
        {
            mfem_error("L2ProjectRemap: Integration rule has not been set. ");
        }
        
        ParFiniteElementSpace *fes = gf.ParFESpace();
        ParMesh *pmesh = fes->GetParMesh();

        ParMesh *pmesh_new = new ParMesh(*pmesh);
        ParGridFunction *nodes_new = dynamic_cast<ParGridFunction *>(pmesh_new->GetNodes());
        nodes_new->Add(-1.0, mesh_velocity);
        pmesh_new->NodesUpdated();

        ParFiniteElementSpace *fes_new = new ParFiniteElementSpace(pmesh_new, fes->FEColl(), fes->GetVDim());
        
        L2Projector interp(*fes, *fes_new, *ir, 1e-4, periodic, size_y, size_z);
        interp.SetLimitType(L2Projector::FCT);
        interp.SetBoundPreservingType(bp_type);

        ParGridFunction gf_new(fes_new);
        
        interp.Interpolate(gf, gf_new);

        gf = gf_new;
        
        delete fes_new;
        delete pmesh_new;
    }


    HPRemap::HPRemap(ParMesh &pmesh_,
                     ParFiniteElementSpace *fes_A_,
                     ParFiniteElementSpace *fes_B_,
                     real_t mu_,
                     VectorFunctionCoefficient *H_bdry_coeff_,
                     Array<int> ess_bdr_H_) : pmesh(&pmesh_),
                        mesh_order(pmesh->GetNodes()->FESpace()->GetOrder(0)),
                        fes_A(fes_A_),
                        fes_B(fes_B_),
                        mu(mu_),
                        mu_coeff(mu),
                        H_bdry_coeff(H_bdry_coeff_),
                        ess_bdr_H(ess_bdr_H_),
                        ess_A(true)
    {
    }

    HPRemap::~HPRemap()
    {
    }
    
    void HPRemap::Remap(ParGridFunction &mesh_velocity, ParGridFunction &A_gf)
    {
        
        // save initial mesh position
        ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
        ParGridFunction nodes_init(*nodes);

        int max_iter = 100;
        real_t tol = 1e-12;
        
        ParGridFunction H_gf(fes_A);
        ParGridFunction B_gf(fes_B);
        ParGridFunction A_old_gf(fes_A);
        ParGridFunction A_mid_gf(fes_A);
        ParGridFunction A_prev_gf(fes_A);
        
        {
            ComputeCurl(A_gf, B_gf);
            ParBilinearForm H_mass_bf(fes_A);
            H_mass_bf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            H_mass_bf.AddDomainIntegrator(new VectorFEMassIntegrator(mu_coeff)); 
            H_mass_bf.Assemble();
            ProjectBToH(B_gf, H_gf, H_mass_bf);
        }
        
        int mesh_order = mesh_velocity.ParFESpace()->GetOrder(0);
        int A_order = fes_A->GetOrder(0);
        int quad_order = mesh_order * 2 + A_order * 2;

        int order = fes_A->GetOrder(0);
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i = 0; i < Geometry::NumGeom; ++i)
        {
            irs[i] = &(IntRules.Get(i, quad_order));
        }
        const IntegrationRule *ir = irs[pmesh->GetElementGeometry(0)];
        
        real_t tau = 0.0;
        real_t tau_final = 1.0;
        real_t max_velocity = GFMaxNorm(mesh_velocity);
        real_t dt = 0.1*GetMeshSize(pmesh) / max_velocity / mesh_order;
        int n_steps = ceil(tau_final/dt);
        dt = tau_final/n_steps;
        
        for(int i_tau = 0; i_tau<n_steps; i_tau++)
        {
            // if(Mpi::Root())
                // printf("Remap step %d, tau: %g\n", i_tau, tau);
            
            // update mesh to midpoint value
            nodes->Add(-0.5*dt, mesh_velocity);
            pmesh->NodesUpdated();
            
            A_old_gf = A_gf;
            
            // Bilinear forms
            ConstantCoefficient one_over_dt(1.0/dt);
            auto A_mass_integ = new VectorFEMassIntegrator(one_over_dt);
            A_mass_integ->SetIntRule(ir);
            ParBilinearForm A_mass_bf(fes_A);
            A_mass_bf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            A_mass_bf.AddDomainIntegrator(A_mass_integ);
            A_mass_bf.Assemble();
            
            ParBilinearForm H_mass_bf(fes_A);
            H_mass_bf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
            H_mass_bf.AddDomainIntegrator(new VectorFEMassIntegrator(mu_coeff)); 
            H_mass_bf.Assemble();
            
            ParDiscreteLinearOperator curl(fes_A, fes_B);
            curl.AddDomainInterpolator(new CurlInterpolator);
            curl.Assemble();
            curl.Finalize();
            
            int iter = 0;
            real_t res = 0.0;
            for (iter = 0; iter < max_iter; iter++)
            {
                A_prev_gf = A_gf;
                UpdateA(mesh_velocity, A_old_gf, H_gf, A_gf, A_mass_bf, dt, ir);
                A_mid_gf.Set(0.5, A_old_gf);
                A_mid_gf.Add(0.5, A_gf);
                curl.Mult(A_mid_gf, B_gf);
                ProjectBToH(B_gf,H_gf,H_mass_bf);
                // InterpolatePhi(mesh_velocity, A_mid_gf, phi_gf);

                // real_t test = GFInnerProduct(dA_gf, H_gf);
                // if (Mpi::Root())
                // {
                //     std::cout << "iteration: " << iter
                //             << ", test: " << test << std::endl;
                // }

                A_prev_gf.Add(-1.0, A_gf);
                VectorGridFunctionCoefficient A_prev_gf_coeff(&A_prev_gf);
                res = ComputeGlobalLpNorm(2.0, A_prev_gf_coeff, *pmesh, irs);

                // if (Mpi::Root())
                // {
                //     std::cout << "iteration: " << iter
                //             << ", residual: " << res << std::endl;
                // }

                if (res < tol)
                {
                    if (Mpi::Root())
                        printf("tau = %g, converged after %d iterations, residual: %g\n", tau, iter, res);
                    break;
                }
            }
            
            if (iter == max_iter)
            {
                if (Mpi::Root())
                    printf("tau = %g, did not converge after %d iterations, residual: %g\n", tau, iter, res);
            }
            
            tau += dt;
            
            // update mesh to integer step
            nodes->Add(-0.5*dt, mesh_velocity);
            pmesh->NodesUpdated();
        }
        
        // restore initial mesh position
        *nodes = nodes_init;
        pmesh->NodesUpdated();
    }

    void HPRemap::UpdateA(ParGridFunction &mesh_velocity, ParGridFunction &A_old_gf, ParGridFunction &H_gf, ParGridFunction &A_gf, ParBilinearForm &A_mass_bf, real_t dt, const IntegrationRule *ir)
    {
        VectorGridFunctionCoefficient mesh_velocity_coeff(&mesh_velocity);
        VectorGridFunctionCoefficient H_gf_coeff(&H_gf);
        VectorGridFunctionCoefficient A_old_gf_coeff(&A_old_gf);
        ScalarVectorProductCoefficient A_old_over_dt(1.0/dt, A_old_gf_coeff);
        VectorCrossProductCoefficient wxH_coeff(mesh_velocity_coeff, H_gf_coeff);
        ScalarVectorProductCoefficient mu_wxH_coeff(mu, wxH_coeff);
        ConstantCoefficient one_over_dt(1.0/dt);

        ParLinearForm A_lf(fes_A);
        auto *wxH_integ = new VectorFEDomainLFIntegrator(mu_wxH_coeff);
        wxH_integ->SetIntRule(ir);
        A_lf.AddDomainIntegrator(wxH_integ);
        A_lf.Assemble();
        
        OperatorPtr A;
        Vector X, B;
        Array<int> ess_bdr, ess_tdof_list;
        // ess_bdr.SetSize(pmesh->bdr_attributes.Max());
        // ess_bdr = ess_A ? 1 : 0;
        fes_A->GetEssentialTrueDofs(ess_bdr_H, ess_tdof_list);
        A_gf = 0.0;
        A_mass_bf.FormLinearSystem(ess_tdof_list, A_gf, A_lf, A, X, B);

        OperatorJacobiSmoother A_prec;
        
        CGSolver A_solver(MPI_COMM_WORLD);
        A_solver.SetRelTol(1e-15);
        A_solver.SetAbsTol(1e-15);
        A_solver.SetMaxIter(100);
        A_solver.SetPrintLevel(0);
        A_solver.SetPreconditioner(A_prec);
        
        A_solver.SetOperator(*A);
        A_solver.Mult(B,X);
        A_mass_bf.RecoverFEMSolution(X,A_lf, A_gf);
        
        A_gf.Add(1.0, A_old_gf);
    }

    void HPRemap::ProjectBToH(ParGridFunction &B_gf, ParGridFunction &H_gf, ParBilinearForm &H_mass_bf)
    {
        
        VectorGridFunctionCoefficient B_gf_coeff(&B_gf);
        ParLinearForm H_lf(fes_A);
        H_lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(B_gf_coeff));
        H_lf.Assemble();

        H_gf = 0.0;
        H_gf.ProjectBdrCoefficientTangent(*H_bdry_coeff, ess_bdr_H);

        OperatorPtr A;
        Vector X, B;
        Array<int> ess_tdof_list;
        fes_A->GetEssentialTrueDofs(ess_bdr_H, ess_tdof_list);
        H_mass_bf.FormLinearSystem(ess_tdof_list, H_gf, H_lf, A, X, B);
        
        OperatorJacobiSmoother H_prec;
        
        CGSolver H_solver(MPI_COMM_WORLD);
        H_solver.SetRelTol(1e-15);
        H_solver.SetAbsTol(1e-15);
        H_solver.SetMaxIter(100);
        H_solver.SetPrintLevel(0);
        H_solver.SetPreconditioner(H_prec);

        H_solver.SetOperator(*A);
        H_solver.Mult(B,X);

        H_mass_bf.RecoverFEMSolution(X, H_lf, H_gf);
    }

    DG_Evolution::DG_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParLinearForm &b_, ParGridFunction *nodes_, ParGridFunction *mesh_velocity_, int vdim_)
        : TimeDependentOperator(nodes_->ParFESpace()->TrueVSize() + M_.ParFESpace()->GetTrueVSize()*vdim_), 
          M_bf(&M_),
          K_bf(&K_),
          b_lf(&b_),
          pmesh(M_.ParFESpace()->GetParMesh()),
        nodes(nodes_), 
        mesh_fes(nodes->ParFESpace()),
            mesh_velocity(mesh_velocity_),
            M_prec(new HypreDiagScale),
          M_solver(M_.ParFESpace()->GetComm()),
          vdim(vdim_),
          z(M_.ParFESpace()->GetTrueVSize()*vdim_)
    {
        M_solver.SetPreconditioner(*M_prec);
        M_solver.iterative_mode = false;
        M_solver.SetRelTol(1e-9);
        M_solver.SetAbsTol(0.0);
        M_solver.SetMaxIter(100);
        M_solver.SetPrintLevel(0);
    }

    void DG_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
    {
        MFEM_ABORT("DG_Evolution::ImplicitSolve is not implemented.");
    }

    void DG_Evolution::Mult(const Vector &x, Vector &y) const
    {
        Vector x_nodes(x.GetData(), mesh_fes->GetTrueVSize());
        
        nodes->SetFromTrueDofs(x_nodes);
        pmesh->NodesUpdated();
        
        M_bf->Update();
        M_bf->Assemble(0);
        M_bf->Finalize(0);
        
        K_bf->Update();
        K_bf->Assemble(0);
        K_bf->Finalize(0);
        
        b_lf->Assemble();
        
        Vector B(M_bf->ParFESpace()->GetTrueVSize());
        b_lf->ParallelAssemble(B);
        
        Array<int> ess_tdof_list;
        OperatorPtr M, K;
        M_bf->FormSystemMatrix(ess_tdof_list, M);
        K_bf->FormSystemMatrix(ess_tdof_list, K);
        
        M_solver.SetOperator(*M);
        
        // y = M^{-1} (K x + b)
        for (int d = 0; d < vdim; d++)
        {
            Vector x_u(x.GetData() + mesh_fes->GetTrueVSize() + d*M_bf->ParFESpace()->GetTrueVSize(),
                             M_bf->ParFESpace()->GetTrueVSize());
            Vector y_u(y.GetData() + mesh_fes->GetTrueVSize() + d*M_bf->ParFESpace()->GetTrueVSize(),
                             M_bf->ParFESpace()->GetTrueVSize());
            Vector z_u(z.GetData() + d*M_bf->ParFESpace()->GetTrueVSize(),
                             M_bf->ParFESpace()->GetTrueVSize());
            K->Mult(x_u, z_u);
            z_u += B;
            M_solver.Mult(z_u, y_u); 
        }
        
        Vector y_nodes(y.GetData(), mesh_fes->GetTrueVSize());
        mesh_velocity->GetTrueDofs(y_nodes);
        y_nodes.Neg();
        
    }
    
    void DG_Evolution::SetTime(const real_t t)
    {
    }

    DG_Evolution::~DG_Evolution()
    {
        delete M_prec;
    }

    void DGRemap::Remap(ParGridFunction &mesh_velocity, ParGridFunction &u_gf)
    {
        ParFiniteElementSpace *fes = nullptr;
        int vdim = u_gf.VectorDim();
        if(vdim==1)
        {
            fes = u_gf.ParFESpace();
        }
        else // scalar fes
        {
            const FiniteElementCollection *fec = u_gf.FESpace()->FEColl();
            fes = new ParFiniteElementSpace(u_gf.ParFESpace()->GetParMesh(), fec);
        }
        ParMesh *pmesh = fes->GetParMesh();
        ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
        ParGridFunction nodes_init(*nodes);

        ParBilinearForm mass_bf(fes);
        mass_bf.AddDomainIntegrator(new MassIntegrator);

        VectorGridFunctionCoefficient mesh_velocity_coeff(&mesh_velocity);
        const real_t alpha = -1.0;
        ParBilinearForm convect_bf(fes);
        convect_bf.AddDomainIntegrator(new ConvectionIntegrator(mesh_velocity_coeff, alpha));
        convect_bf.AddInteriorFaceIntegrator(
            new NonconservativeDGTraceIntegrator(mesh_velocity_coeff, alpha));
        convect_bf.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(mesh_velocity_coeff, alpha));

        ParLinearForm b_lf(fes);
        ConstantCoefficient zero(0.0);
        b_lf.AddBdrFaceIntegrator(
            new BoundaryFlowIntegrator(zero, mesh_velocity_coeff, alpha));

        DG_Evolution adv(mass_bf, convect_bf, b_lf, nodes, &mesh_velocity, vdim);

        int ode_solver_type = 3;
        std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

        real_t tau = 0.0;
        real_t final_tau = 1.0;
        real_t h = GetMeshSize(fes->GetParMesh());
        real_t max_velocity = GFMaxNorm(mesh_velocity);
        real_t dt = 0.1*h/ max_velocity / (2*fes->GetOrder(0)+1);
        int n_steps = ceil(final_tau/dt);
        dt = final_tau/n_steps;
        if (Mpi::Root())
        {
            std::cout << "Remap: dt = " << dt << ", n_steps = " << n_steps
                      << ", final_tau = " << final_tau << std::endl;
        }
        adv.SetTime(tau);
        ode_solver->Init(adv);

        Vector U;
        U.SetSize(nodes->ParFESpace()->TrueVSize() + fes->GetTrueVSize()*vdim);
        
        Vector U_nodes(U.GetData(), nodes->ParFESpace()->TrueVSize());
        nodes->GetTrueDofs(U_nodes);
        
        Vector U_u(U.GetData() + nodes->ParFESpace()->TrueVSize(), fes->GetTrueVSize()*vdim);
        u_gf.GetTrueDofs(U_u);

        bool done = false;
        for (int ti = 0; !done;)
        {
            real_t dt_real = fmin(dt, final_tau - tau);
            ode_solver->Step(U, tau, dt_real);
            ti++;
            
            // if (Mpi::Root())
            // {
            //     std::cout << "pseudo time step: " << ti << ", pseudo time: " << tau << std::endl;
            // }

            done = (tau >= final_tau - 1e-8 * dt);
        }
        
        u_gf.SetFromTrueDofs(U_u);
        
        // restore the mesh
        *nodes = nodes_init;
        pmesh->NodesUpdated();
        
        if(vdim>1)
        {
            delete fes;
        }
        
    }
    
    void H1DGRemap::Remap(ParGridFunction &mesh_velocity, ParGridFunction &u_gf)
    {
        int order = u_gf.ParFESpace()->GetOrder(0);
        int dim = u_gf.ParFESpace()->GetMesh()->Dimension();
        ParMesh *pmesh = u_gf.ParFESpace()->GetParMesh();
        int vdim = u_gf.VectorDim();
        
        L2_FECollection l2_fec(order, dim);
        ParFiniteElementSpace l2_fes(pmesh, &l2_fec, vdim);
        
        ParGridFunction u_l2(&l2_fes);
        u_l2.ProjectGridFunction(u_gf);
        
        DGRemap remap;
        remap.Remap(mesh_velocity, u_l2);
        
        ParGridFunction *nodes = dynamic_cast<ParGridFunction *>(pmesh->GetNodes());
        ParGridFunction nodes_init(*nodes);
        
        nodes->Add(-1.0, mesh_velocity);
        pmesh->NodesUpdated();
        
        if(vdim==1)
        {
            GridFunctionCoefficient u_l2_coeff(&u_l2);
            u_gf.ProjectDiscCoefficient(u_l2_coeff, GridFunction::AvgType::ARITHMETIC);
        }
        else
        {
            VectorGridFunctionCoefficient u_l2_coeff(&u_l2);
            u_gf.ProjectDiscCoefficient(u_l2_coeff, GridFunction::AvgType::ARITHMETIC);
        }
        *nodes = nodes_init;
        pmesh->NodesUpdated();
    }
    
}
