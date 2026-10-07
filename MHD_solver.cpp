// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.


#include "general/forall.hpp"
#include "MHD_solver.hpp"
#include "linalg/kernels.hpp"
#include <unordered_map>
#include <fstream>
#include "mesh_smoother.hpp"
#include "Interpolator.hpp"
#include "remap.hpp"
#include "mean_field.hpp"
#include "tools.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace hydrodynamics
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   gf.HostRead();
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
                            ? "mAcRjl" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys " << keys;
         if ( vec ) { sock << "vvv"; }
         sock << std::endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

LagrangianHydroOperator::LagrangianHydroOperator(const int size,
                                                 ParMesh *pmesh_,
                                                 ParFiniteElementSpace &x_fes_,
                                                 ParFiniteElementSpace &v_fes_,
                                                 ParFiniteElementSpace &l2_e,
                                                 ParFiniteElementSpace &l2_rho,
                                                 const Array<int> &ess_tdofs,
                                                 ParGridFunction &rho0_gf,
                                                 ParGridFunction &gamma_gf,
                                                 const double cfl,
                                                 const bool visc,
                                                 const bool vort,
                                                 const double cgt,
                                                 const int cgiter,
                                                 const int oq,
                                                 ParGridFunction &A_gf,
                                                 ParGridFunction &B_gf,
                                                 ParFiniteElementSpace &divB_fes_,
                                                 real_t mu_,
                                                 VectorFunctionCoefficient &H_bdry_coeff_,
                                                 Array<int> ess_bdr_H_,
                                                 MeshSmoothType mesh_smooth_type_,
                                                 real_t mesh_smooth_eps_,
                                                 ProblemData *pd_,
                                                bool fix_step_remesh_,
                                                int fix_step_remesh_interval_) :
   TimeDependentOperator(size),
   pmesh(pmesh_),
   fes_x(x_fes_), fes_v(v_fes_), fes_e(l2_e), fes_rho(l2_rho),
   xVsize(fes_x.GetVSize()), 
   vVsize(fes_v.GetVSize()), vTVSize(fes_v.TrueVSize()), vGTVSize(fes_v.GlobalTrueVSize()),
   eVsize(fes_e.GetVSize()), eTVSize(fes_e.TrueVSize()), eGTVSize(fes_e.GlobalTrueVSize()),
   rhoVsize(fes_rho.GetVSize()), rhoTVSize(fes_rho.TrueVSize()), rhoGTVSize(fes_rho.GlobalTrueVSize()),
   order_v(fes_v.GetOrder(0)), order_e(fes_e.GetOrder(0)), order_rho(fes_rho.GetOrder(0)),
   block_offsets(4),
   x_gf(),
   ess_tdofs(ess_tdofs),
   dim(pmesh->Dimension()),
   NE(pmesh->GetNE()),
   edofs_cnt(fes_e.GetFE(0)->GetDof()),
   cfl(cfl),
   use_viscosity(visc), use_vorticity(vort),
   cg_rel_tol(cgt), cg_max_iter(cgiter),
   gamma_gf(gamma_gf),
   Mv(nullptr), Mv_spmat_copy(),
   Me(edofs_cnt, edofs_cnt, NE), Me_inv(edofs_cnt, edofs_cnt, NE),
   rho_gf(rho0_gf), A_gf(A_gf), B_gf(B_gf),
   fes_A(*A_gf.ParFESpace()),
   fes_B(*B_gf.ParFESpace()),
   fes_divB(divB_fes_),
   order_A(fes_A.GetOrder(0)),
   order_B(fes_B.GetOrder(0)),
   mu(mu_),
   H_bdry_coeff(H_bdry_coeff_),
   ess_bdr_H(ess_bdr_H_),
   order_q((oq > 0) ? oq : 3 * order_v + order_e - 1 + order_A + order_rho),
   ir(IntRules.Get(pmesh->GetElementBaseGeometry(0),order_q)),
   qdata(dim, NE, ir.GetNPoints()),
   Force(nullptr),
   one(eVsize),
   rhs(vVsize),
   e_rhs(eVsize),
   mesh_smoother(nullptr),
   nodes(nullptr),
   mesh_smooth_type(mesh_smooth_type_),
   mesh_smooth_eps(mesh_smooth_eps_),
   pd(pd_),
   fix_step_remesh(fix_step_remesh_),
   fix_step_remesh_interval(fix_step_remesh_interval_),
   min_detJ(0.0),
   max_detJ(1e20),
   max_ratio(1e20),
   max_disp(1e20),
   remap_type_v(RemapType::DGconvection),
   remap_type_e(RemapType::DGconvection),
   remap_type_A(dim==3?RemapType::HelicityPreserving:RemapType::DGconvection),
   remap_type_rho(RemapType::INTERPOLATE)
{
   
   if(Mpi::Root()) printf("Order of quadrature: %d \n", order_q);
   
   FunctionCoefficient rho_coeff(pd->rho0);
   ProjectDensity(rho_gf, rho_coeff, ir, true, BoundPreservingType::POSITIVE);
      
   block_offsets[0] = 0;
   block_offsets[1] = block_offsets[0] + xVsize;
   block_offsets[2] = block_offsets[1] + vVsize;
   block_offsets[3] = block_offsets[2] + eVsize;
   one.UseDevice(true);
   one = 1.0;
   
   AssembleMvMe();

   // Values of Jac0inv at all quadrature points.
   // Initial local mesh size (assumes all mesh elements are the same).
   int Ne, ne = NE;
   double Volume, vol = 0.0;
   {
      const int NQ = ir.GetNPoints();
      for (int e = 0; e < NE; e++)
      {
         ElementTransformation &Tr = *pmesh->GetElementTransformation(e);
         for (int q = 0; q < NQ; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            DenseMatrixInverse Jinv(Tr.Jacobian());
            Jinv.GetInverseMatrix(qdata.Jac0inv(e*NQ + q));
         }
      }
      for (int e = 0; e < NE; e++) { vol += pmesh->GetElementVolume(e); }
   }
   MPI_Allreduce(&vol, &Volume, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&ne, &Ne, 1, MPI_INT, MPI_SUM, pmesh->GetComm());
   switch (pmesh->GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT: qdata.h0 = Volume / Ne; break;
      case Geometry::SQUARE: qdata.h0 = sqrt(Volume / Ne); break;
      case Geometry::TRIANGLE: qdata.h0 = sqrt(2.0 * Volume / Ne); break;
      case Geometry::CUBE: qdata.h0 = pow(Volume / Ne, 1./3.); break;
      case Geometry::TETRAHEDRON: qdata.h0 = pow(6.0 * Volume / Ne, 1./3.); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   qdata.h0 /= (double) order_v;
   
   switch (mesh_smooth_type)
   {
   case MeshSmoothType::NONE:
      mesh_smoother = new IdentitySmoother(*pmesh);
      break;
   case MeshSmoothType::INITIAL:
      mesh_smoother = new InitialSmoother(*pmesh);
      break;
   case MeshSmoothType::LIMITED_HARMONIC:
      mesh_smoother = new LimitedHarmonicSmoother(*pmesh, mesh_smooth_eps);
      break;
   case MeshSmoothType::PERIODIC:
      mesh_smoother = new PeriodicSmoother(*pmesh, mesh_smooth_eps);
      break;
   default:
      mfem_error("Unknown mesh smoothing type! ");
      break;
   }
   nodes = dynamic_cast<ParGridFunction *> (pmesh->GetNodes());
   
}

LagrangianHydroOperator::~LagrangianHydroOperator()
{
   if(Mv) delete Mv;
   if(Force) delete Force;
   delete mesh_smoother;
}

void LagrangianHydroOperator::AssembleMvMe()
{
   GridFunctionCoefficient rho_coeff(&rho_gf);
   VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &ir);
   
   if(Mv) delete Mv;
   Mv = new ParBilinearForm(&fes_v);
   Mv->AddDomainIntegrator(vmi);

   // Standard local assembly and inversion for energy mass matrices.
   // 'Me' is used in the computation of the internal energy
   // which is used twice: once at the start and once at the end of the run.
   MassIntegrator mi(rho_coeff, &ir);
   for (int e = 0; e < NE; e++)
   {
      Me(e) = 0.0;
      Me_inv(e) = 0.0;
      DenseMatrixInverse inv(&Me(e));
      const FiniteElement &fe = *fes_e.GetFE(e);
      ElementTransformation &Tr = *fes_e.GetElementTransformation(e);
      mi.AssembleElementMatrix(fe, Tr, Me(e));
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(e));
   }
   // Standard assembly for the velocity mass matrix.
   Mv->Assemble();
   Mv->Finalize();
   Mv_spmat_copy = Mv->SpMat();
   
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt) const
{
   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   UpdateMesh(S);
   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   Vector* sptr = const_cast<Vector*>(&S);
   ParGridFunction v;
   v.MakeRef(&fes_v, *sptr, xVsize);
   // Set dx_dt = v (explicit).
   ParGridFunction dx;
   dx.MakeRef(&fes_x, dS_dt, 0);
   dx.ProjectGridFunction(v);
   SolveVelocity(S, dS_dt);
   SolveEnergy(S, v, dS_dt);
}

void LagrangianHydroOperator::SolveVelocity(const Vector &S,
                                            Vector &dS_dt) const
{
   UpdateQuadratureData(S);
   AssembleForceMatrix();
   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   ParGridFunction dv;
   dv.MakeRef(&fes_v, dS_dt, xVsize);
   dv = 0.0;

   Force->Mult(one, rhs);
   rhs.Neg();

   if (pd->u_source)
   {
      ParLinearForm rhs_accel_lf(&fes_v);
      pd->u_source->SetTime(t);
      auto *integ = new VectorDomainLFIntegrator(*pd->u_source);
      integ->SetIntRule(&ir);
      rhs_accel_lf.AddDomainIntegrator(integ);
      rhs_accel_lf.Assemble();
      rhs += rhs_accel_lf;
   }
   
   ParLinearForm Lorentz_lf(&fes_v);
   LorentzForceIntegrator *lorentz_integ = new LorentzForceIntegrator(B_gf, mu);
   lorentz_integ->SetIntRule(&ir);
   Lorentz_lf.AddDomainIntegrator(lorentz_integ);
   Lorentz_lf.Assemble();
   rhs -= Lorentz_lf;
   
   ParLinearForm pressure_lf(&fes_v);
   FunctionCoefficient pressure_bdry_coeff(pd->p_bdry_func);
   pressure_bdry_coeff.SetTime(t);
   auto pressure_integ = new VectorBoundaryFluxLFIntegrator(pressure_bdry_coeff);
   pressure_integ->SetIntRule(&ir);
   if(pd->ess_bdr_p.Size() > 0)
      pressure_lf.AddBoundaryIntegrator(pressure_integ, pd->ess_bdr_p);
   pressure_lf.Assemble();
   rhs += pressure_lf;
   
   ParLinearForm magnetic_bdry_lf(&fes_v);
   VectorGridFunctionCoefficient B_coeff(&B_gf);
   auto magnetic_bdry_integ = new MagneticBoundaryIntegrator(B_gf, mu);
   magnetic_bdry_integ->SetIntRule(&ir);
   magnetic_bdry_lf.AddBdrFaceIntegrator(magnetic_bdry_integ);
   magnetic_bdry_lf.Assemble();
   // rhs += magnetic_bdry_lf;

   HypreParMatrix A;
   Vector X(vTVSize);
   Vector B(vTVSize);
   Mv->FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);

   CGSolver cg(pmesh->GetComm());
   HypreSmoother prec;
   prec.SetType(HypreSmoother::Jacobi, 1);
   cg.SetPreconditioner(prec);
   cg.SetOperator(A);
   cg.SetRelTol(cg_rel_tol);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(cg_max_iter);
   cg.SetPrintLevel(-1);
   ;
   cg.Mult(B, X);
   Mv->RecoverFEMSolution(X, rhs, dv);
}

void LagrangianHydroOperator::SolveEnergy(const Vector &S, const Vector &v,
                                          Vector &dS_dt) const
{
   UpdateQuadratureData(S);
   AssembleForceMatrix();

   // The monolithic BlockVector stores the unknown fields as follows:
   // (Position, Velocity, Specific Internal Energy).
   ParGridFunction de;
   de.MakeRef(&fes_e, dS_dt, xVsize + vVsize);
   de = 0.0;

   // Solve for energy, assemble the energy source if such exists.
   LinearForm *e_source = nullptr;
   if (pd->e_source) 
   {
      // Needed since the Assemble() defaults to PA.
      fes_e.GetMesh()->DeleteGeometricFactors();
      e_source = new LinearForm(&fes_e);
      pd->e_source->SetTime(t);
      DomainLFIntegrator *d = new DomainLFIntegrator(*pd->e_source, &ir);
      e_source->AddDomainIntegrator(d);
      e_source->Assemble();
   }

   Array<int> l2dofs;
   {
      Force->MultTranspose(v, e_rhs);
      if (e_source) { e_rhs += *e_source; }
      Vector loc_rhs(edofs_cnt), loc_de(edofs_cnt);
      for (int e = 0; e < NE; e++)
      {
         fes_e.GetElementDofs(e, l2dofs);
         e_rhs.GetSubVector(l2dofs, loc_rhs);
         Me_inv(e).Mult(loc_rhs, loc_de);
         de.SetSubVector(l2dofs, loc_de);
      }
   }
   delete e_source;
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
   Vector* sptr = const_cast<Vector*>(&S);
   x_gf.MakeRef(&fes_x, *sptr, 0);
   pmesh->NewNodes(x_gf, false);
}

void LagrangianHydroOperator::SetVelocityBoundaryValue(ParGridFunction &v_gf)
{
   Array<int> dofs_marker, dofs_list, ess_vdofs;
   Array<real_t> ess_vdofs_val;
   for (int d = 0; d < v_gf.VectorDim(); d++)
   {
      Array<int> ess_bdr = pd->ess_bdrs_v[d];
      Array<int> ess_bdr_copy(ess_bdr);
      for (int i = 0; i < ess_bdr.Size(); i++)
      {
         if (ess_bdr[i] == 0)
            continue;
         ess_bdr_copy = 0;
         ess_bdr_copy[i] = 1;
         fes_v.GetEssentialVDofs(ess_bdr_copy, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
         Array<real_t> vals(dofs_list.Size());
         vals = pd->bdr_vals_v[d][i];
         ess_vdofs_val.Append(vals);
      }
   }
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      v_gf(ess_vdofs[i]) = ess_vdofs_val[i];
   }
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
   UpdateMesh(S);
   UpdateQuadratureData(S);
   double glob_dt_est;
   const MPI_Comm comm = pmesh->GetComm();
   MPI_Allreduce(&qdata.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN, comm);
   return glob_dt_est;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   qdata.dt_est = std::numeric_limits<double>::infinity();
}

double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &gf) const
{
   GridFunctionCoefficient rho_gf_coeff(&rho_gf);
   ParLinearForm inner_form(gf.ParFESpace());
   auto integ = new DomainLFIntegrator(rho_gf_coeff);
   integ->SetIntRule(&ir);
   inner_form.AddDomainIntegrator(integ);
   inner_form.Assemble();

   real_t inner_loc = InnerProduct(inner_form, gf);
   real_t inner_glob;
   MPI_Allreduce(&inner_loc, &inner_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   return inner_glob;
}

double LagrangianHydroOperator::KineticEnergy(const ParGridFunction &v) const
{
   GridFunctionCoefficient rho_gf_coeff(&rho_gf);
   VectorGridFunctionCoefficient v_gf_coeff(&v);
   ScalarVectorProductCoefficient rho_v_coeff(rho_gf_coeff, v_gf_coeff);
   ParLinearForm inner_form(v.ParFESpace());
   auto integ = new VectorDomainLFIntegrator(rho_v_coeff);
   integ->SetIntRule(&ir);
   inner_form.AddDomainIntegrator(integ);
   inner_form.Assemble();

   real_t inner_loc = InnerProduct(inner_form, v);
   real_t inner_glob;
   MPI_Allreduce(&inner_loc, &inner_glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   return 0.5*inner_glob;
}

// Smooth transition between 0 and 1 for x in [-eps, eps].
MFEM_HOST_DEVICE inline double smooth_step_01(double x, double eps)
{
   const double y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   // This code is only for the 1D/FA mode
   const int nqp = ir.GetNPoints();
   ParGridFunction x, v, e;
   Vector* sptr = const_cast<Vector*>(&S);
   x.MakeRef(&fes_x, *sptr, 0);
   v.MakeRef(&fes_v, *sptr, xVsize);
   e.MakeRef(&fes_e, *sptr, xVsize + vVsize);
   Vector e_vals;
   DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim);

   // Batched computations are needed, because hydrodynamic codes usually
   // involve expensive computations of material properties. Although this
   // miniapp uses simple EOS equations, we still want to represent the batched
   // cycle structure.
   int nzones_batch = 3;
   const int nbatches =  NE / nzones_batch + 1; // +1 for the remainder.
   int nqp_batch = nqp * nzones_batch;
   double *gamma_b = new double[nqp_batch],
   *rho_b = new double[nqp_batch],
   *e_b   = new double[nqp_batch],
   *p_b   = new double[nqp_batch],
   *cs_b  = new double[nqp_batch];
   Vector *B_b = new Vector[nqp_batch];
   for(int i = 0; i < nqp_batch; i++)
   {
      B_b[i].SetSize(dim);
   }
   // Jacobians of reference->physical transformations for all quadrature points
   // in the batch.
   DenseTensor *Jpr_b = new DenseTensor[nzones_batch];
   for (int b = 0; b < nbatches; b++)
   {
      int z_id = b * nzones_batch; // Global index over zones.
      // The last batch might not be full.
      if (z_id == NE) { break; }
      else if (z_id + nzones_batch > NE)
      {
         nzones_batch = NE - z_id;
         nqp_batch    = nqp * nzones_batch;
      }

      double min_detJ = std::numeric_limits<double>::infinity();
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = pmesh->GetElementTransformation(z_id);
         Jpr_b[z].SetSize(dim, dim, nqp);
         e.GetValues(z_id, ir, e_vals);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            Jpr_b[z](q) = T->Jacobian();
            const double detJ = Jpr_b[z](q).Det();
            min_detJ = fmin(min_detJ, detJ);
            const int idx = z * nqp + q;
            // Assuming piecewise constant gamma that moves with the mesh.
            gamma_b[idx] = gamma_gf(z_id);
            rho_b[idx] = rho_gf.GetValue(*T, ip);
            // if (rho_b[idx] < 0.0)
            // {
               // real_t min_detJ, max_detJ;
               // ComputeMinMaxDetJ(min_detJ, max_detJ);
               // const FiniteElement *fe = fes_rho.GetFE(z_id);
               // Vector shape(fe->GetDof());
               // real_t weight = T->Weight();
               // real_t detJ = T->Jacobian().Det();
               // fe->CalcShape(ip, shape);
               // fe->CalcPhysShape(*Tr_rho, shape);
               // printf("Weight: %g \n", weight);
               // printf("DetJ: %g \n", detJ);
               // for(int d = 0; d < fe->GetDof(); d++)
               // {
               //    printf("shape function at zone %d, "
               //           "quadrature point %d, dof %d: %g\n",
               //           z_id, q, d, shape(d));
               // }
               // MFEM_ABORT("Negative density at zone " << z_id
               //            << ", quadrature point " << q);
            // }
            e_b[idx] = fmax(0.0, e_vals(q));
            B_gf.GetVectorValue(z_id, ip, B_b[idx]);
         }
         ++z_id;
      }

      // Batched computation of material properties.
      ComputeMaterialProperties(nqp_batch, gamma_b, rho_b, e_b, B_b, p_b, cs_b);

      z_id -= nzones_batch;
      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = pmesh->GetElementTransformation(z_id);
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            // Note that the Jacobian was already computed above. We've chosen
            // not to store the Jacobians for all batched quadrature points.
            const DenseMatrix &Jpr = Jpr_b[z](q);
            CalcInverse(Jpr, Jinv);
            const double detJ = Jpr.Det(), rho = rho_b[z*nqp + q],
                         p = p_b[z*nqp + q], sound_speed = cs_b[z*nqp + q];
            stress = 0.0;
            for (int d = 0; d < dim; d++) { stress(d, d) = -p; }
            double visc_coeff = 0.0;
            if (use_viscosity)
            {
               // Compression-based length scale at the point. The first
               // eigenvector of the symmetric velocity gradient gives the
               // direction of maximal compression. This is used to define the
               // relative change of the initial length scale.
               v.GetVectorGradient(*T, sgrad_v);

               double vorticity_coeff = 1.0;
               if (use_vorticity)
               {
                  const double grad_norm = sgrad_v.FNorm();
                  const double div_v = fabs(sgrad_v.Trace());
                  vorticity_coeff = (grad_norm > 0.0) ? div_v / grad_norm : 1.0;
               }

               sgrad_v.Symmetrize();
               double eig_val_data[3], eig_vec_data[9];
               if (dim==1)
               {
                  eig_val_data[0] = sgrad_v(0, 0);
                  eig_vec_data[0] = 1.;
               }
               else { sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data); }
               Vector compr_dir(eig_vec_data, dim);
               // Computes the initial->physical transformation Jacobian.
               mfem::Mult(Jpr, qdata.Jac0inv(z_id*nqp + q), Jpi);
               Vector ph_dir(dim); Jpi.Mult(compr_dir, ph_dir);
               // Change of the initial mesh size in the compression direction.
               const double h = qdata.h0 * ph_dir.Norml2() /
                                compr_dir.Norml2();
               // Measure of maximal compression.
               const double mu = eig_val_data[0];
               visc_coeff = 2.0 * rho * h * h * fabs(mu);
               // The following represents a "smooth" version of the statement
               // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
               // eps must be scaled appropriately if a different unit system is
               // being used.
               const double eps = 1e-12;
               visc_coeff += 0.5 * rho * h * sound_speed * vorticity_coeff *
                             (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
               stress.Add(visc_coeff, sgrad_v);
            }
            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min =
               Jpr.CalcSingularvalue(dim-1) / order_v;
            const double inv_dt = sound_speed / h_min +
                                  2.5 * visc_coeff / rho / h_min / h_min;
            if (min_detJ < 0.0)
            {
               // This will force repetition of the step with smaller dt.
               qdata.dt_est = 0.0;
            }
            else
            {
               if (inv_dt>0.0)
               {
                  qdata.dt_est = fmin(qdata.dt_est, cfl*(1.0/inv_dt));
               }
            }
            // Quadrature data for partial assembly of the force operator.
            MultABt(stress, Jinv, stressJiT);
            stressJiT *= ir.IntPoint(q).weight * detJ;
            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  qdata.stressJinvT(vd)(z_id*nqp + q, gd) =
                     stressJiT(vd, gd);
               }
            }
         }
         ++z_id;
      }
   }
   delete [] gamma_b;
   delete [] rho_b;
   delete [] e_b;
   delete [] p_b;
   delete [] cs_b;
   delete [] Jpr_b;
   delete [] B_b;
}

void LagrangianHydroOperator::AssembleForceMatrix() const
{
   if(Force) delete Force;
   Force = new MixedBilinearForm(&fes_e, &fes_v);
   auto fi = new ForceIntegrator(qdata);
   fi->SetIntRule(&ir);
   Force->AddDomainIntegrator(fi);
   Force->Assemble();
}

void LagrangianHydroOperator::ComputeMeshQuality(real_t &min_detJ, real_t &max_detJ, real_t &singular_ratio) const
{
   real_t min_rel_detJ = std::numeric_limits<real_t>::infinity();
   real_t max_rel_detJ = -std::numeric_limits<real_t>::infinity();
   real_t max_ratio = -std::numeric_limits<real_t>::infinity();
   DenseMatrix Jac(dim, dim), Jac0inv(dim, dim), Jac0invJac(dim, dim);
   const int NQ = ir.GetNPoints();
   for(int i_elem = 0; i_elem < NE; i_elem++)
   {
      ElementTransformation *T = pmesh->GetElementTransformation(i_elem);
      for(int q = 0; q<NQ; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         Jac = T->Jacobian();
         Jac0inv = qdata.Jac0inv(i_elem*NQ+q);
         mfem::Mult(Jac0inv, Jac, Jac0invJac);
         real_t rel_detJ = Jac0invJac.Det();
         min_rel_detJ = fmin(min_rel_detJ, rel_detJ);
         max_rel_detJ = fmax(max_rel_detJ, rel_detJ);
         real_t sing_max = Jac0invJac.CalcSingularvalue(0);
         real_t sing_min = Jac0invJac.CalcSingularvalue(dim-1);
         max_ratio = fmax(max_ratio, sing_max / sing_min);
      }  
   }
   
   MPI_Allreduce(&min_rel_detJ, &min_detJ, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh->GetComm());
   MPI_Allreduce(&max_rel_detJ, &max_detJ, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
   MPI_Allreduce(&max_ratio, &singular_ratio, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
   
   // if(Mpi::Root())
   // {
   //    printf("Minimum relative detJ: %g, maximum relative detJ: %g, singular value ratio: %g \n",
   //             min_detJ, max_detJ, singular_ratio);
   // }
}

bool LagrangianHydroOperator::NeedRemesh()
{
   static int false_time = 0;
   
   real_t disp = mesh_smoother->MaxDisplacement();
   if(pd->periodic)
   {
      if(Mpi::Root())
      {
         printf("periodic displacement: %g \n", disp);
      }
   }
   
   real_t min, max, ratio;
   ComputeMeshQuality(min, max, ratio);
   bool need_remesh = false;
   need_remesh = (ratio > max_ratio) || (min<min_detJ) || (max>max_detJ) || (fix_step_remesh && false_time>=fix_step_remesh_interval-1) 
   || (pd->periodic && (disp > max_disp))
   ;
   if(!need_remesh)
   {
      false_time++;
   }
   else
   {
      false_time = 0;
   }
   // if(need_remesh)
   {
      if(Mpi::Root())
      {
         printf("Need remesh: ratio: %g, min detJ: %g, max detJ: %g disp: %g \n",
                ratio, min, max, disp);
      }
   }
   
   return need_remesh;
   
}

void LagrangianHydroOperator::SetComovingRezone(bool enabled)
{
   auto *initial = dynamic_cast<InitialSmoother *>(mesh_smoother);
   MFEM_VERIFY(!enabled || (initial && pd->periodic),
               "Comoving rezoning requires a periodic problem and initial-mesh smoother (-mst 1)");
   if (initial && pd->periodic)
   {
      Vector lengths(dim);
      lengths(0) = pd->px_length;
      lengths(1) = pd->py_length;
      if (dim == 3) { lengths(2) = pd->pz_length; }
      initial->ConfigurePeriodic(lengths, enabled);
   }
}

void LagrangianHydroOperator::SetPreserveMeanField(bool enabled)
{
   preserve_mean_field = enabled && pd->periodic;
   if (!preserve_mean_field) { return; }
   MFEM_VERIFY(pd->B0_periodic, "Periodic background coefficient is required");
   int required_order = dim == 2 ? order_v : 2*order_v;
   MFEM_VERIFY(order_A >= required_order,
               "Mean-field correction requires A order >= mesh order in 2D, "
               "or >= twice mesh order in 3D");
   mean_field_reference_nodes.SetSpace(&fes_x);
   mean_field_reference_nodes = *nodes;
   uniform_background.SetSize(dim);
   ElementTransformation *first = pmesh->GetElementTransformation(0);
   pd->B0_periodic->Eval(uniform_background, *first, ir.IntPoint(0));
   // Every rank must check against the same value, including a background that
   // happens to be constant within each partition but differs across partitions.
   MPI_Bcast(uniform_background.GetData(), dim, MPI_DOUBLE, 0, pmesh->GetComm());
   real_t local_error = 0.0;
   Vector sample(dim);
   for (int e = 0; e < NE; ++e)
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         auto *T = pmesh->GetElementTransformation(e);
         pd->B0_periodic->Eval(sample, *T, ir.IntPoint(q));
         sample -= uniform_background;
         local_error = std::max(local_error, sample.Norml2());
      }
   real_t global_error;
   MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
   MFEM_VERIFY(global_error <= 1e-12*std::max(uniform_background.Norml2(),1e-30),
               "Mean-field correction supports spatially uniform backgrounds only");
}

void LagrangianHydroOperator::RemeshAndRemap(Vector &S)
{   
   if(Mpi::Root())
   {
      printf("-----REMAP-----\n");
   }
      
   ParGridFunction newnodes(nodes->ParFESpace());
   mesh_smoother->Smooth(newnodes);
   
   x_gf.MakeRef(&fes_x, S, 0);
   ParGridFunction mesh_velocity(&fes_x);
   mesh_velocity.Set(1.0, x_gf);
   mesh_velocity.Add(-1.0, newnodes);
   
   ParGridFunction v_gf;
   v_gf.MakeRef(&fes_v, S, xVsize);
   
   ParGridFunction e_gf;
   e_gf.MakeRef(&fes_e, S, xVsize + vVsize);
   
   real_t rho_min, rho_max;
   real_t e_min, e_max;
   GFMinMax(e_gf, e_min, e_max, &ir);
   GFMinMax(rho_gf, rho_min, rho_max, &ir);

   if (preserve_mean_field)
   {
      AddUniformMeanFieldPotential(A_gf, mean_field_reference_nodes,
                                   *nodes, uniform_background);
      // Verify the decomposition BEFORE applying any remap or moving the mesh.
      ParGridFunction reconstructed(&fes_B), background(&fes_B);
      ComputeCurl(A_gf, reconstructed);
      background = 0.0;
      DivFreeProject(background, &fes_divB, *pd->B0_periodic);
      reconstructed += background;
      reconstructed -= B_gf;
      real_t relative_error = sqrt(GFInnerProduct(reconstructed,reconstructed) /
                                    std::max(GFInnerProduct(B_gf,B_gf),1e-300));
      if (Mpi::Root())
         printf("Mean-field decomposition relative L2 error: %.16e\n", relative_error);
      MFEM_VERIFY(relative_error < 1e-8,
                  "Mean-field potential decomposition failed its native flux check");
   }

   // Remap rho
   if(Mpi::Root())
   {
      printf("Remapping rho ... \n");
      printf("Before remap: rho min: %g, max: %g \n", rho_min, rho_max);
   }
   switch (remap_type_rho) 
   {
   case RemapType::L2Projection:
   {
      L2ProjectRemap l2_remap_rho;
      l2_remap_rho.SetIntegrationRule(ir);
      l2_remap_rho.SetBoundPreservingType(bp_type);
      l2_remap_rho.SetPeriodic(pd->periodic, pd->py_length, pd->pz_length, pd->px_length);
      l2_remap_rho.Remap(mesh_velocity, rho_gf);
      
      break;
   }
   case RemapType::Exact:
   {
      ExactRemap rho_remap;
      FunctionCoefficient exact_rho_coeff(pd->exact_rho);
      rho_remap.SetExactCoefficient(&exact_rho_coeff, t);
      rho_remap.SetPositive();
      rho_remap.SetIntegralType();
      rho_remap.Remap(mesh_velocity, rho_gf); 
      break;
   }
   default:
   {
      mfem_error("Unknown remap type for density.");
      break;
   }
   }
   
   GFMinMax(rho_gf, rho_min, rho_max, &ir);
   if(Mpi::Root())
   {
      printf("After remap: rho min: %g, max: %g \n", rho_min, rho_max);
   }

   // Remap velocity
   if(Mpi::Root())
   {
      printf("Remapping velocity ... \n");
   }
   switch (remap_type_v)
   {
   case RemapType::INTERPOLATE:
   {
      InterpolateRemap interp_remap;
      interp_remap.Remap(mesh_velocity, v_gf);
      break;
   }
   case RemapType::DGconvection:
   {
      H1DGRemap v_remap;
      v_remap.Remap(mesh_velocity, v_gf);
      
      // set boundary data 
      SetVelocityBoundaryValue(v_gf);
      break;
   }
   case RemapType::Exact:
   {
      ExactRemap v_remap;
      VectorFunctionCoefficient exact_v_coeff(dim, pd->exact_v);
      v_remap.SetExactVectorCoefficient(&exact_v_coeff, t);
      v_remap.Remap(mesh_velocity, v_gf);
      break;
   }
   default:
      mfem_error("Unknown remap type for velocity.");
      break;
   }
   
   
   if(Mpi::Root())
   {
      printf("Remapping internal energy ... \n");
      printf("Before remap: e min: %g, max: %g \n", e_min, e_max);
   }
   switch (remap_type_e)
   {
   case RemapType::INTERPOLATE:
   {
      InterpolateRemap interp_remap;
      interp_remap.Remap(mesh_velocity, e_gf);
      break;
   }
   case RemapType::DGconvection:
   {
      DGRemap dg_remap;
      dg_remap.Remap(mesh_velocity, e_gf);
      break;
   }
   case RemapType::Exact:
   {
      ExactRemap e_remap;
      FunctionCoefficient exact_e_coeff(pd->exact_e);
      e_remap.SetExactCoefficient(&exact_e_coeff, t);
      e_remap.SetPositive();
      e_remap.Remap(mesh_velocity, e_gf);
      break;
   }
   default:
      mfem_error("Unknown remap type for internal energy.");
      break;
   }
   
   GFMinMax(e_gf, e_min, e_max, &ir);
   if(Mpi::Root())
   {
      printf("After remap: e min: %g, max: %g \n", e_min, e_max);
   }
   
   if(Mpi::Root())
   {
      printf("Remapping magnetic vector potential ... \n");
   }
   
   switch (remap_type_A)
   {
   case RemapType::Exact:
   {
      ExactRemap A_remap;
      if(dim==3)
      {
         VectorFunctionCoefficient exact_A_coeff(dim, pd->exact_A);
         A_remap.SetExactVectorCoefficient(&exact_A_coeff, t);
      }
      else if(dim==2)
      {
         VectorFunctionCoefficient exact_A_coeff(1, pd->exact_A);
         A_remap.SetExactVectorCoefficient(&exact_A_coeff, t);
      }
      
      A_remap.Remap(mesh_velocity, A_gf);
      break;
   }
   case RemapType::INTERPOLATE:
   {
      InterpolateRemap interp_remap;
      interp_remap.Remap(mesh_velocity, A_gf);
      break;
   }
   case RemapType::HelicityPreserving:
   {
      MFEM_VERIFY(dim==3, "Helicity preserving remap is only implemented for 3D.");
      HPRemap remap_A(*pmesh, &fes_A, &fes_B, mu, &H_bdry_coeff, ess_bdr_H);
      remap_A.SetEssentialA(mesh_smooth_type!=MeshSmoothType::INITIAL);
      remap_A.Remap(mesh_velocity, A_gf);
      break;
   }
   case RemapType::DGconvection:
   {
      MFEM_VERIFY(dim==2, "DG convection remap is only implemented for 2D.");
      H1DGRemap remap_A;
      remap_A.Remap(mesh_velocity, A_gf);
      break;
   }
   default:
      mfem_error("Unknown remap type for magnetic vector potential.");
      break;
   }

   // todo: gamma remap? check the use of gamma in the whole code.
   
   // Update the mesh nodes
   x_gf = newnodes;
   UpdateMesh(S);
   
   // update B
   ComputeCurl(A_gf, B_gf);
   if(pd->periodic)
   {
      ParGridFunction B0_gf(&fes_B);
      B0_gf = 0.0;
      DivFreeProject(B0_gf, &fes_divB, *pd->B0_periodic);
      B_gf += B0_gf;
   }
   
   if (preserve_mean_field) { mean_field_reference_nodes = *nodes; }

   // restart Lagrangian 
   AssembleMvMe();
         
   if(Mpi::Root())
   {
      printf("-----REMAP DONE-----\n");
   }
   
}

} // namespace hydrodynamics


} // namespace mfem

#endif // MFEM_USE_MPI
