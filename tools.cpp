#include "tools.hpp"
#include "Integrators.hpp"

namespace mfem
{

    void ComputeCurlOfScalar(const ParGridFunction &E_gf, ParGridFunction &curl_gf)
    {
        StopWatch timer;

        ParFiniteElementSpace *fes_curl = curl_gf.ParFESpace();

        MFEM_VERIFY(E_gf.VectorDim() == 1, "E_gf should be a scalar field");

        curl_gf = 0.0;

        int order = E_gf.ParFESpace()->GetOrder(0);
        int order_quad = order * 3;
        const IntegrationRule *ir = &IntRules.Get(E_gf.ParFESpace()->GetFE(0)->GetGeomType(), order_quad);

        // left hand side: (B,C)
        ParBilinearForm Mass_B_bf(fes_curl);
        auto mass_integ = new VectorFEMassIntegrator;
        mass_integ->SetIntRule(ir);
        Mass_B_bf.AddDomainIntegrator(mass_integ);
        Mass_B_bf.Assemble();
        Mass_B_bf.Finalize();

        // right hand side: (curl E, C)
        // curl E = (d/dy, -d/dx) E = [0,1;-1,0] grad E
        DenseMatrix rot_op(2);
        rot_op = 0.0;
        rot_op(0, 1) = 1.0;
        rot_op(1, 0) = -1.0;
        MatrixConstantCoefficient rot_coeff(rot_op);

        GridFunctionCoefficient E_coeff(&E_gf);
        GradientGridFunctionCoefficient grad_E_coeff(&E_gf);
        VectorCoefficient *curl_E_coeff = new MatrixVectorProductCoefficient(rot_coeff, grad_E_coeff);

        ParLinearForm rhs_lf(fes_curl);
        auto rhs_integ = new VectorFEDomainLFIntegrator(*curl_E_coeff);
        rhs_integ->SetIntRule(ir);
        rhs_lf.AddDomainIntegrator(rhs_integ);
        rhs_lf.Assemble();

        OperatorPtr A;
        Vector X, B;
        Array<int> ess_tdof_list; // empty list
        Mass_B_bf.FormLinearSystem(ess_tdof_list, curl_gf, rhs_lf, A, X, B);

        timer.Restart();
        OperatorJacobiSmoother prec;
        CGSolver solver(MPI_COMM_WORLD);
        solver.SetPreconditioner(prec);
        solver.SetOperator(*A);
        solver.SetRelTol(1e-15);
        solver.SetMaxIter(100);
        solver.SetAbsTol(1e-15);
        solver.Mult(B, X);
        if (Mpi::Root())
            printf("solver_curl: %d its, residual: %g, time: %g\n", solver.GetNumIterations(), solver.GetFinalNorm(), timer.RealTime());

        Mass_B_bf.RecoverFEMSolution(X, rhs_lf, curl_gf);

        /* free memory */
        delete curl_E_coeff;
    }

    void ComputeCurl3D(const ParGridFunction &E_gf, ParGridFunction &curl_gf)
    {
        StopWatch timer;

        ParFiniteElementSpace *fes_E = E_gf.ParFESpace();
        ParFiniteElementSpace *fes_curl = curl_gf.ParFESpace();

        int order = fes_E->GetOrder(0);
        int order_quad = order * 3;
        const IntegrationRule *ir = &IntRules.Get(E_gf.ParFESpace()->GetFE(0)->GetGeomType(), order_quad);

        MFEM_VERIFY(E_gf.VectorDim() == 3, "E_gf should be a 3D vector field");

        curl_gf = 0.0;

        // left hand side: (B,C)
        ParBilinearForm Mass_B_bf(fes_curl);
        auto mass_integ = new VectorFEMassIntegrator;
        mass_integ->SetIntRule(ir);
        Mass_B_bf.AddDomainIntegrator(mass_integ);
        Mass_B_bf.Assemble();
        Mass_B_bf.Finalize();

        // right hand side: (curl E, C)
        ParMixedBilinearForm curlE_bf(fes_E, fes_curl);
        auto curl_integ = new MixedVectorCurlIntegrator;
        curl_integ->SetIntRule(ir);
        curlE_bf.AddDomainIntegrator(curl_integ);
        curlE_bf.Assemble();
        curlE_bf.Finalize();

        ParLinearForm rhs_lf(fes_curl);
        rhs_lf.Assemble();
        curlE_bf.AddMult(E_gf, rhs_lf);

        OperatorPtr A;
        Vector X, B;
        Array<int> ess_tdof_list; // empty list
        Mass_B_bf.FormLinearSystem(ess_tdof_list, curl_gf, rhs_lf, A, X, B);

        timer.Restart();
        OperatorJacobiSmoother prec;
        CGSolver solver(MPI_COMM_WORLD);
        solver.SetPreconditioner(prec);
        solver.SetOperator(*A);
        solver.SetRelTol(1e-15);
        solver.SetMaxIter(100);
        solver.SetAbsTol(1e-15);
        solver.Mult(B, X);
        if (Mpi::Root())
            printf("solver_curl: %d its, residual: %g, time: %g\n", solver.GetNumIterations(), solver.GetFinalNorm(), timer.RealTime());

        Mass_B_bf.RecoverFEMSolution(X, rhs_lf, curl_gf);
    }

    void ComputeCurl3D_interp(const ParGridFunction &E_gf, ParGridFunction &curl_gf)
    {

        ParFiniteElementSpace *fes_E = E_gf.ParFESpace();
        ParFiniteElementSpace *fes_curl = curl_gf.ParFESpace();

        ParDiscreteLinearOperator curl(fes_E, fes_curl);
        curl.AddDomainInterpolator(new CurlInterpolator);
        curl.Assemble();
        curl.Finalize();

        curl.Mult(E_gf, curl_gf);
    }

    void ComputeCurl(ParGridFunction &A_gf, ParGridFunction &B_gf)
    {
        int A_dim = A_gf.VectorDim();
        if (A_dim == 1)
        {
            ComputeCurlOfScalar(A_gf, B_gf);
        }
        else if (A_dim == 3)
        {
            ComputeCurl3D_interp(A_gf, B_gf);
        }
        else
        {
            MFEM_ABORT("A_gf should be a 1D or 3D vector field");
        }
    }

    real_t GFInnerProduct(const ParGridFunction &gf_1, const ParGridFunction &gf_2)
    {
        VectorGridFunctionCoefficient gf_2_coeff(&gf_2);
        ParLinearForm inner_form(gf_1.ParFESpace());
        inner_form.AddDomainIntegrator(new VectorFEDomainLFIntegrator(gf_2_coeff));
        inner_form.Assemble();

        real_t inner_loc = InnerProduct(inner_form, gf_1);
        real_t inner_glob;
        MPI_Allreduce(&inner_loc, &inner_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return inner_glob;
    }

    real_t GFMaxNorm(const ParGridFunction &gf)
    {
        ParFiniteElementSpace *fes = gf.ParFESpace();
        ParMesh *pmesh = fes->GetParMesh();
        int vdim = gf.VectorDim();

        real_t max_norm = 0.0;
        for (int i_elem = 0; i_elem < pmesh->GetNE(); i_elem++)
        {
            ElementTransformation *T = fes->GetElementTransformation(i_elem);
            const FiniteElement *fe = fes->GetFE(i_elem);
            const IntegrationRule &ir = fe->GetNodes();
            for (int q = 0; q < ir.GetNPoints(); q++)
            {
                const IntegrationPoint &ip = ir.IntPoint(q);
                T->SetIntPoint(&ip);
                Vector val(vdim);
                gf.GetVectorValue(i_elem, ip, val);
                max_norm = fmax(max_norm, val.Norml2());
            }
        }

        real_t max_norm_glob;
        MPI_Allreduce(&max_norm, &max_norm_glob, 1, MPITypeMap<real_t>::mpi_type,
                      MPI_MAX, pmesh->GetComm());

        return max_norm_glob;
    }

    void GFMinMax(const ParGridFunction &gf, real_t &min_val, real_t &max_val, const IntegrationRule *ir_)
    {
        ParFiniteElementSpace *fes = gf.ParFESpace();
        ParMesh *pmesh = fes->GetParMesh();
        int vdim = gf.VectorDim();

        const IntegrationRule *ir = ir_ ? ir_ : &fes->GetFE(0)->GetNodes();

        real_t max_val_loc = -1e10;
        real_t min_val_loc = 1e10;
        for (int i_elem = 0; i_elem < pmesh->GetNE(); i_elem++)
        {
            ElementTransformation *T = fes->GetElementTransformation(i_elem);
            const FiniteElement *fe = fes->GetFE(i_elem);
            for (int q = 0; q < ir->GetNPoints(); q++)
            {
                const IntegrationPoint &ip = ir->IntPoint(q);
                T->SetIntPoint(&ip);
                Vector val(vdim);
                gf.GetVectorValue(i_elem, ip, val);
                for (int d = 0; d < vdim; d++)
                {
                    min_val_loc = std::min(min_val_loc, val[d]);
                    max_val_loc = std::max(max_val_loc, val[d]);
                }
            }
        }

        real_t min_val_glob, max_val_glob;
        MPI_Allreduce(&min_val_loc, &min_val_glob, 1, MPITypeMap<real_t>::mpi_type,
                      MPI_MIN, pmesh->GetComm());
        MPI_Allreduce(&max_val_loc, &max_val_glob, 1, MPITypeMap<real_t>::mpi_type,
                      MPI_MAX, pmesh->GetComm());

        min_val = min_val_glob;
        max_val = max_val_glob;
    }

    real_t GFDivError(ParGridFunction &gf)
    {
        ConstantCoefficient zero(0.0);
        real_t div_err = gf.ComputeDivError(&zero);
        return div_err;
    }

    real_t GetMeshSize(ParMesh *pmesh)
    {
        pmesh->EnsureNodes();
        int dim = pmesh->Dimension();
        DenseMatrix Jacobian(dim, dim);

        int NE = pmesh->GetNE();
        real_t h_min = std::numeric_limits<real_t>::max();
        for (int i_elem = 0; i_elem < NE; i_elem++)
        {
            pmesh->GetElementJacobian(i_elem, Jacobian);
            h_min = fmin(h_min, Jacobian.CalcSingularvalue(dim - 1));
        }

        real_t h_min_glob;
        MPI_Allreduce(&h_min, &h_min_glob, 1, MPITypeMap<real_t>::mpi_type,
                      MPI_MIN, pmesh->GetComm());
        return h_min_glob;
    }

    void ProjectDensity(ParGridFunction &rho_gf, Coefficient &rho_coeff, const IntegrationRule &ir, bool bound_preserving, BoundPreservingType bp_type)
    {
        rho_gf = 0.0;
        ParFiniteElementSpace *fes = rho_gf.ParFESpace();
        ParMesh *pmesh = fes->GetParMesh();
        ReferenceMassIntegrator mass_integ;
        ReferenceLFIntegrator rho_integ(rho_coeff);
        DenseMatrix elmat;
        Vector elvect, rhovec;
        {
            const FiniteElement *fe = fes->GetFE(0);
            ElementTransformation *Tr = pmesh->GetElementTransformation(0);
            int Ndof = fe->GetDof();
            elmat.SetSize(Ndof);
            mass_integ.AssembleElementMatrix(*fe, *Tr, elmat);
        }
        MatrixInverse *inv = elmat.Inverse();

        int npt = ir.GetNPoints();
        int NE = pmesh->GetNE();
        Vector loc_min_vec(NE), loc_max_vec(NE);
        real_t glb_min, glb_max;
        loc_min_vec = 1e20;
        loc_max_vec = -1e20;
        if (bound_preserving)
        {
            for (int i = 0; i < pmesh->GetNE(); i++)
            {
                ElementTransformation *Tr = pmesh->GetElementTransformation(i);
                for (int j = 0; j < npt; j++)
                {
                    const IntegrationPoint &ip = ir.IntPoint(j);
                    Tr->SetIntPoint(&ip);
                    real_t val = rho_coeff.Eval(*Tr, ip);
                    loc_min_vec(i) = std::min(loc_min_vec(i), val);
                    loc_max_vec(i) = std::max(loc_max_vec(i), val);
                }
            }
            real_t loc_proc_min = loc_min_vec.Min();
            real_t loc_proc_max = loc_max_vec.Max();
            MPI_Allreduce(&loc_proc_min, &glb_min, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
            MPI_Allreduce(&loc_proc_max, &glb_max, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());

            if (Mpi::Root())
                printf("ProjectDensity: coeff global min = %g, global max = %g \n", glb_min, glb_max);
        }

        for (int i = 0; i < pmesh->GetNE(); i++)
        {
            const FiniteElement *fe = fes->GetFE(i);
            ElementTransformation *Tr = pmesh->GetElementTransformation(i);
            int Ndof = fe->GetDof();

            elvect.SetSize(Ndof);
            rhovec.SetSize(Ndof);

            rho_integ.AssembleRHSElementVect(*fe, *Tr, elvect);

            inv->Mult(elvect, rhovec);
            Array<int> vdofs;
            fes->GetElementVDofs(i, vdofs);
            rho_gf.SetSubVector(vdofs, rhovec);

            if (bound_preserving)
            {
                real_t min_bound, max_bound;
                switch (bp_type)
                {
                case BoundPreservingType::NONE:
                {
                    min_bound = -1e20;
                    max_bound = 1e20;
                    break;
                }
                case BoundPreservingType::POSITIVE:
                {
                    min_bound = 0.0;
                    max_bound = 1e20;
                    break;
                }
                case BoundPreservingType::LOCAL:
                {
                    min_bound = loc_min_vec(i);
                    max_bound = loc_max_vec(i);
                    break;
                }
                case BoundPreservingType::GLOBAL:
                {
                    min_bound = glb_min;
                    max_bound = glb_max;
                    break;
                }
                default:
                {
                    mfem_error("Unknown bound preserving type");
                    break;
                }
                }

                int ndofs = Ndof;

                // compute low-order coeffs
                real_t c_bar = elvect.Sum();

                // lumped mass
                Vector m_lump(ndofs);
                m_lump = 0.0;
                for (int d = 0; d < ndofs; d++)
                {
                    for (int e = 0; e < ndofs; e++)
                    {
                        m_lump(d) += elmat(d, e);
                    }
                }

                Vector z(rhovec);
                for (int d = 0; d < ndofs; d++)
                {
                    z(d) = elvect(d) - c_bar * m_lump(d);
                }

                DenseMatrix fij(ndofs, ndofs);
                for (int a = 0; a < ndofs; a++)
                {
                    for (int b = 0; b < ndofs; b++)
                    {
                        fij(a, b) = elmat(a, b) * (rhovec(a) - rhovec(b)) + (z(a) - z(b)) / real_t(ndofs);
                    }
                }

                // FCT coeffs
                DenseMatrix beta(ndofs, ndofs);
                Vector deltac_p(ndofs), deltac_m(ndofs);
                Vector detJ(npt);
                for (int j = 0; j < npt; j++)
                {
                    const IntegrationPoint &ip = ir.IntPoint(j);
                    Tr->SetIntPoint(&ip);
                    detJ(j) = Tr->Weight();
                }
                real_t detJ_max = detJ.Max();
                real_t detJ_min = detJ.Min();
                real_t deltac_max = max_bound * detJ_max - c_bar;
                real_t deltac_min = min_bound * detJ_min - c_bar;
                Vector beta_p(ndofs), beta_m(ndofs);

                deltac_p = 0.0;
                deltac_m = 0.0;
                beta_p = 0.0;
                beta_m = 0.0;
                for (int a = 0; a < ndofs; a++)
                {
                    for (int b = 0; b < ndofs; b++)
                    {
                        deltac_p(a) += 1.0 / m_lump(a) * std::max(0.0, fij(a, b));
                        deltac_m(a) += 1.0 / m_lump(a) * std::min(0.0, fij(a, b));
                    }
                    beta_p(a) = std::min(1.0, deltac_max / (deltac_p(a) + 1e-16));
                    beta_m(a) = std::min(1.0, deltac_min / (deltac_m(a) - 1e-16));
                }

                for (int a = 0; a < ndofs; a++)
                {
                    for (int b = 0; b < ndofs; b++)
                    {
                        if (fij(a, b) > 0)
                        {
                            beta(a, b) = std::min(beta_p(a), beta_m(b));
                        }
                        else
                        {
                            beta(a, b) = std::min(beta_m(a), beta_p(b));
                        }
                    }
                }

                for (int a = 0; a < ndofs; a++)
                {
                    rhovec(a) = c_bar;
                    for (int b = 0; b < ndofs; b++)
                    {
                        rhovec(a) += 1.0 / m_lump(a) * beta(a, b) * fij(a, b);
                    }
                }

                rho_gf.SetSubVector(vdofs, rhovec);
            }
        }

        if (bound_preserving)
        {
            GFMinMax(rho_gf, glb_min, glb_max);
            if (Mpi::Root())
                printf("ProjectDensity: global min = %g, global max = %g \n", glb_min, glb_max);
        }

        delete inv;
    }

    void DivFreeProject(ParGridFunction &gf, ParFiniteElementSpace *W_space, VectorCoefficient &fcoeff)
    {
        // A fully periodic divergence constraint has a constant null mode.
        // Uniform harmonic fields already have an exact RT representation;
        // retain it rather than invoking a singular mixed-system solve.
        gf.ProjectCoefficient(fcoeff);
        const real_t norm = sqrt(GFInnerProduct(gf,gf));
        if(GFDivError(gf) <= 1e-11*std::max(norm,real_t(1e-30))) return;

        ParFiniteElementSpace *R_space = gf.ParFESpace();
        StopWatch chrono;

        if (Mpi::Root())
        {
            std::cout << "Solving div-free projection problem..." << std::endl;
        }

        chrono.Start();

        Array<int> block_offsets(3); // number of variables + 1
        block_offsets[0] = 0;
        block_offsets[1] = R_space->GetVSize();
        block_offsets[2] = W_space->GetVSize();
        block_offsets.PartialSum();

        Array<int> block_trueOffsets(3); // number of variables + 1
        block_trueOffsets[0] = 0;
        block_trueOffsets[1] = R_space->TrueVSize();
        block_trueOffsets[2] = W_space->TrueVSize();
        block_trueOffsets.PartialSum();

        BlockVector x(block_offsets), rhs(block_offsets);
        BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

        ParLinearForm fform;
        fform.Update(R_space, rhs.GetBlock(0), 0);
        fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
        fform.Assemble();
        fform.SyncAliasMemory(rhs);
        fform.ParallelAssemble(trueRhs.GetBlock(0));
        trueRhs.GetBlock(0).SyncAliasMemory(trueRhs);

        ParLinearForm gform;
        gform.Update(W_space, rhs.GetBlock(1), 0);
        gform.Assemble();
        gform.SyncAliasMemory(rhs);
        gform.ParallelAssemble(trueRhs.GetBlock(1));
        trueRhs.GetBlock(1).SyncAliasMemory(trueRhs);

        ConstantCoefficient one(1.0);
        ConstantCoefficient m_one(-1.0);

        ParBilinearForm mVarf(R_space);
        ParMixedBilinearForm bVarf(R_space, W_space);

        HypreParMatrix *M = nullptr;
        HypreParMatrix *B = nullptr;

        mVarf.AddDomainIntegrator(new VectorFEMassIntegrator(one));
        mVarf.Assemble();
        mVarf.Finalize();

        bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator(m_one));
        bVarf.Assemble();
        bVarf.Finalize();

        BlockOperator darcyOp(block_trueOffsets);
        Array<int> empty_tdof_list; // empty

        M = mVarf.ParallelAssemble();
        B = bVarf.ParallelAssemble();
        TransposeOperator Bt(B);
        darcyOp.SetBlock(0, 0, M);
        darcyOp.SetBlock(0, 1, &Bt);
        darcyOp.SetBlock(1, 0, B);

        chrono.Stop();
        if (Mpi::Root())
        {
            std::cout << "  Assembly time: " << chrono.RealTime() << " s" << std::endl;
        }

        chrono.Restart();

        // preconditioner
        HypreParMatrix *MinvBt = NULL;
        HypreParMatrix *S = NULL;

        HypreParVector Md(MPI_COMM_WORLD,
                          M->GetGlobalNumRows(),
                          M->GetRowStarts());
        M->GetDiag(Md);

        MinvBt = B->Transpose();
        MinvBt->InvScaleRows(Md);
        S = ParMult(B, MinvBt);

        HypreDiagScale invM;
        HypreBoomerAMG invS;
        invM.SetOperator(*M);
        invS.SetPrintLevel(0);
        invS.SetOperator(*S);

        invM.iterative_mode = false;
        invS.iterative_mode = false;

        BlockDiagonalPreconditioner darcyPr(block_trueOffsets);
        darcyPr.SetDiagonalBlock(0, &invM);
        darcyPr.SetDiagonalBlock(1, &invS);

        chrono.Stop();
        if (Mpi::Root())
        {
            std::cout << "  Preconditioner setup time: " << chrono.RealTime() << "s" << std::endl;
        }

        int maxIter(500);
        real_t rtol(1.e-15);
        real_t atol(1.e-15);

        MINRESSolver solver(MPI_COMM_WORLD);
        solver.SetAbsTol(atol);
        solver.SetRelTol(rtol);
        solver.SetMaxIter(maxIter);
        solver.SetOperator(darcyOp);
        solver.SetPreconditioner(darcyPr);
        solver.SetPrintLevel(3);
        trueX = 0.0;
        chrono.Restart();
        solver.Mult(trueRhs, trueX);
        chrono.Stop();
        if (Mpi::Root())
        {
            std::cout << "  Solver time: " << chrono.RealTime() << " s" << std::endl;
        }

        gf.Distribute(trueX.GetBlock(0));

        delete MinvBt;
        delete S;
        delete M;
        delete B;

        if (Mpi::Root())
        {
            std::cout << "Div-free projection problem solved." << std::endl;
        }

        return;
    }

}
