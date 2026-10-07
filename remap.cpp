#include "remap.hpp"
#include "tools.hpp"
#include "Integrators.hpp"
#include <algorithm>

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
        
        L2Projector interp(*fes, *fes_new, *ir, 1e-4, periodic, size_y, size_z, size_x);
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
        real_t dt = pseudo_cfl*GetMeshSize(pmesh) / max_velocity / mesh_order;
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

    DG_Evolution::DG_Evolution(ParBilinearForm &K_, ParGridFunction *nodes_,
                               ParGridFunction *mesh_velocity_, int vdim_,
                               const IntegrationRule *mass_ir_)
        : TimeDependentOperator(nodes_->ParFESpace()->TrueVSize() + K_.ParFESpace()->GetTrueVSize()*vdim_),
          K_bf(&K_),
          fes(K_.ParFESpace()),
          pmesh(fes->GetParMesh()),
          nodes(nodes_),
          mesh_fes(nodes->ParFESpace()),
          mesh_velocity(mesh_velocity_),
          mass_ir(mass_ir_),
          vdim(vdim_),
          X(fes),
          z(fes->GetTrueVSize())
    {
        MFEM_VERIFY(fes->GetTrueVSize() == fes->GetVSize(),
                    "DG_Evolution expects an L2 space whose true dofs coincide with its local dofs");
        const int ne = pmesh->GetNE();
        const int ndof = ne > 0 ? fes->GetFE(0)->GetDof() : 0;
        for (Cache &c : cache) { c.Me_inv.SetSize(ndof, ndof, ne); }
    }

    void DG_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
    {
        MFEM_ABORT("DG_Evolution::ImplicitSolve is not implemented.");
    }

    void DG_Evolution::Mult(const Vector &x, Vector &y) const
    {
        const int nmesh = mesh_fes->GetTrueVSize();
        const int n = fes->GetTrueVSize();

        Vector x_nodes(x.GetData(), nmesh);

        // Look for operators built at this mesh position (equal up to
        // rounding: nodes move linearly in pseudo-time, so stage positions
        // either coincide or differ by O(dt |v|)). The decision is collective.
        int hit = -1;
        {
            real_t loc[2] = {std::numeric_limits<real_t>::infinity(), std::numeric_limits<real_t>::infinity()};
            for (int c = 0; c < 2; c++)
            {
                const Cache &C = cache[c];
                if (!C.valid || C.nodes.Size() != nmesh) { continue; }
                real_t diff = 0.0, scale = 0.0;
                for (int i = 0; i < nmesh; i++)
                {
                    diff = fmax(diff, fabs(x_nodes(i) - C.nodes(i)));
                    scale = fmax(scale, fabs(C.nodes(i)));
                }
                loc[c] = diff / fmax(scale, 1e-300);
            }
            real_t glob[2];
            MPI_Allreduce(loc, glob, 2, MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
            for (int c = 0; c < 2; c++) { if (glob[c] <= 1e-12) { hit = c; break; } }
        }

        if (hit < 0)
        {
            nodes->SetFromTrueDofs(x_nodes);
            pmesh->NodesUpdated();
            Cache &C = cache[cache_lru];

            // Convection matrix on the current mesh. After the first assembly
            // the sparsity pattern is fixed, so only the values are rebuilt.
            if (!K_allocated)
            {
                K_bf->Assemble(0);
                K_bf->Finalize(0);
                K_allocated = true;
            }
            else
            {
                // Shared faces are integrated with the neighbors' node
                // positions, which must follow the moving mesh.
                if (fes->GetFaceNbrVSize() > 0) { nodes->ExchangeFaceNbrData(); }
                K_bf->SpMat() = 0.0;
                K_bf->Assemble(0);
            }
            const SparseMatrix &Kasm = K_bf->SpMat();
            // Explicit copy (assigning from an aliasing temporary would move
            // the alias instead of copying the values).
            const int nnz = Kasm.NumNonZeroElems();
            C.K_values.SetSize(nnz);
            std::copy(Kasm.GetData(), Kasm.GetData() + nnz, C.K_values.GetData());

            // Element-local inverses of the block-diagonal DG mass matrix.
            MassIntegrator mi;
            if (mass_ir) { mi.SetIntRule(mass_ir); }
            DenseMatrix Me;
            for (int e = 0; e < pmesh->GetNE(); e++)
            {
                const FiniteElement &fe = *fes->GetFE(e);
                ElementTransformation &Tr = *fes->GetElementTransformation(e);
                mi.AssembleElementMatrix(fe, Tr, Me);
                DenseMatrixInverse inv(Me);
                inv.Factor();
                inv.GetInverseMatrix(C.Me_inv(e));
            }
            C.nodes = x_nodes;
            C.valid = true;
            hit = cache_lru;
        }
        // The other slot becomes the next one to be replaced.
        cache_lru = 1 - hit;
        const Cache &C = cache[hit];
        const SparseMatrix &Kpat = K_bf->SpMat();
        // Non-owning view: cached values over the fixed sparsity pattern.
        SparseMatrix K(const_cast<int*>(Kpat.GetI()), const_cast<int*>(Kpat.GetJ()),
                       const_cast<real_t*>(C.K_values.GetData()),
                       Kpat.Height(), Kpat.Width(), false, false, false);
        const DenseTensor &Me_inv = C.Me_inv;

        // y = M^{-1} K x, component by component.
        const int nfn = fes->GetFaceNbrVSize();
        Array<int> dofs;
        Vector loc_z, loc_y;
        for (int d = 0; d < vdim; d++)
        {
            Vector x_u(x.GetData() + nmesh + d*n, n);
            Vector y_u(y.GetData() + nmesh + d*n, n);
            if (nfn > 0)
            {
                X = x_u;
                X.ExchangeFaceNbrData();
                x_full.SetSize(n + nfn);
                x_full.SetVector(X, 0);
                x_full.SetVector(X.FaceNbrData(), n);
                K.Mult(x_full, z);
            }
            else
            {
                K.Mult(x_u, z);
            }
            for (int e = 0; e < pmesh->GetNE(); e++)
            {
                fes->GetElementDofs(e, dofs);
                z.GetSubVector(dofs, loc_z);
                loc_y.SetSize(loc_z.Size());
                Me_inv(e).Mult(loc_z, loc_y);
                y_u.SetSubVector(dofs, loc_y);
            }
        }

        Vector y_nodes(y.GetData(), nmesh);
        mesh_velocity->GetTrueDofs(y_nodes);
        y_nodes.Neg();
    }

    void DG_Evolution::SetTime(const real_t t)
    {
    }

    DG_Evolution::~DG_Evolution() { }

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

        VectorGridFunctionCoefficient mesh_velocity_coeff(&mesh_velocity);
        const real_t alpha = -1.0;
        ParBilinearForm convect_bf(fes);
        convect_bf.AddDomainIntegrator(new ConvectionIntegrator(mesh_velocity_coeff, alpha));
        convect_bf.AddInteriorFaceIntegrator(
            new NonconservativeDGTraceIntegrator(mesh_velocity_coeff, alpha));
        convect_bf.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(mesh_velocity_coeff, alpha));
        // The inflow boundary value is zero, so the boundary-flow linear form
        // vanishes identically and is not assembled.

        DG_Evolution adv(convect_bf, nodes, &mesh_velocity, vdim);

        int ode_solver_type = 3;
        std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

        real_t tau = 0.0;
        real_t final_tau = 1.0;
        real_t h = GetMeshSize(fes->GetParMesh());
        real_t max_velocity = GFMaxNorm(mesh_velocity);
        real_t dt = pseudo_cfl*h/ max_velocity / (2*fes->GetOrder(0)+1);
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
        remap.SetPseudoCFL(pseudo_cfl);
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
