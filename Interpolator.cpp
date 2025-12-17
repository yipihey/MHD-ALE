#include "mfem.hpp"
#include "Interpolator.hpp"
#include "Integrators.hpp"
#include "tools.hpp"

using namespace mfem;

Interpolator::Interpolator(const ParFiniteElementSpace &src_,
                           const ParFiniteElementSpace &tar_,
                           real_t bdr_tol_)
    : src_fes(&src_),
      tar_fes(&tar_),
      pmesh_src(*src_fes->GetParMesh()),
      pmesh_tar(*tar_fes->GetParMesh())
{

   // check dimension
   mesh_dim = pmesh_tar.Dimension();
   MFEM_VERIFY(mesh_dim == pmesh_src.Dimension(), "Dimension mismatch between source and target meshes.");
   MFEM_VERIFY(mesh_dim > 1, "GSLIB requires a 2D or a 3D mesh");

   target_NE = pmesh_tar.GetNE();
   target_mesh_order = pmesh_tar.GetNodes()->FESpace()->GetElementOrder(0);
   target_fes_order = tar_fes->GetElementOrder(0);

   {
      const FiniteElementCollection *tar_fec = tar_fes->FEColl();
      const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(tar_fec);
      const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *>(tar_fec);
      const RT_FECollection *fec_rt = dynamic_cast<const RT_FECollection *>(tar_fec);
      const ND_FECollection *fec_nd = dynamic_cast<const ND_FECollection *>(tar_fec);
      if (fec_h1)
      {
         target_fieldtype = 0;
      }
      else if (fec_l2)
      {
         target_fieldtype = 1;
      }
      else if (fec_rt)
      {
         target_fieldtype = 2;
      }
      else if (fec_nd)
      {
         target_fieldtype = 3;
      }
      else
      {
         MFEM_ABORT("GridFunction type not supported yet.");
      }
   }

   target_nsp = tar_fes->GetFE(0)->GetNodes().GetNPoints();

   // ensure pmesh has nodes
   pmesh_src.EnsureNodes();
   pmesh_tar.EnsureNodes();

   // Generate list of points where the grid function will be evaluated.
   Vector vxyz;
   int point_ordering;
   if (target_fieldtype == 0 && target_fes_order == target_mesh_order)
   {
      vxyz = *pmesh_tar.GetNodes();
      point_ordering = pmesh_tar.GetNodes()->FESpace()->GetOrdering();
   }
   else
   {
      vxyz.SetSize(target_nsp * target_NE * mesh_dim);
      for (int i = 0; i < target_NE; i++)
      {
         const FiniteElement *fe = tar_fes->GetFE(i);
         const IntegrationRule ir = fe->GetNodes();
         ElementTransformation *et = tar_fes->GetElementTransformation(i);

         DenseMatrix pos;
         et->Transform(ir, pos);
         Vector rowx(vxyz.GetData() + i * target_nsp, target_nsp),
             rowy(vxyz.GetData() + i * target_nsp + target_NE * target_nsp, target_nsp),
             rowz;
         if (mesh_dim == 3)
         {
            rowz.SetDataAndSize(vxyz.GetData() + i * target_nsp + 2 * target_NE * target_nsp, target_nsp);
         }
         pos.GetRow(0, rowx);
         pos.GetRow(1, rowy);
         if (mesh_dim == 3)
         {
            pos.GetRow(2, rowz);
         }
      }
      point_ordering = Ordering::byNODES;
   }

   nodes_cnt = vxyz.Size() / mesh_dim;

   // set up finder
   finder = new FindPointsGSLIB();
   finder->SetDistanceToleranceForPointsFoundOnBoundary(bdr_tol_);
   finder->Setup(pmesh_src);
   finder->FindPoints(vxyz, point_ordering);
}

Interpolator::~Interpolator()
{
   finder->FreeData();
   delete finder;
}

void Interpolator::Interpolate(const ParGridFunction &func_source, ParGridFunction &func_target)
{

   // Verify that the source/target grid functions are in the fespaces that the interpolator was created with
   MFEM_VERIFY(func_source.ParFESpace() == src_fes, "Source grid function is not in the source finite element space.");
   MFEM_VERIFY(func_target.ParFESpace() == tar_fes, "Target grid function is not in the target finite element space.");

   int target_vectordim = func_target.VectorDim();
   Vector interp_vals(nodes_cnt * func_target.VectorDim());
   finder->Interpolate(func_source, interp_vals);

   // project the interpolated values to the target grid function
   if (target_fieldtype <= 1) // H1 or L2
   {
      if ((target_fieldtype == 0 && target_fes_order == target_mesh_order))
      {
         func_target = interp_vals;
      }
      else if (target_fieldtype == 1)
      {
         Array<int> vdofs;
         Vector vals;
         Vector elem_dof_vals(target_nsp * target_vectordim);

         for (int i = 0; i < pmesh_tar.GetNE(); i++)
         {
            const FiniteElement *fe = tar_fes->GetFE(i);
            const IntegrationRule ir = fe->GetNodes();
            ElementTransformation *Tr = tar_fes->GetElementTransformation(i);
            tar_fes->GetElementVDofs(i, vdofs);
            vals.SetSize(vdofs.Size());
            for (int j = 0; j < target_nsp; j++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               Tr->SetIntPoint(&ip);
               for (int d = 0; d < target_vectordim; d++)
               {
                  // Arrange values byNodes
                  int idx = src_fes->GetOrdering() == Ordering::byNODES ? d * target_nsp * target_NE + i * target_nsp + j : i * target_nsp * mesh_dim + d + j * mesh_dim;
                  elem_dof_vals(j + d * target_nsp) = interp_vals(idx);
                  if (fe->GetMapType() == FiniteElement::MapType::INTEGRAL)
                  {
                     elem_dof_vals(j + d * target_nsp) *= Tr->Weight();
                  }
               }
            }
            func_target.SetSubVector(vdofs, elem_dof_vals);
         }
      }
      else // H1 - but mesh order != GridFunction order
      {
         Array<int> vdofs;
         Vector vals;
         Vector elem_dof_vals(target_nsp * target_vectordim);

         for (int i = 0; i < pmesh_tar.GetNE(); i++)
         {
            tar_fes->GetElementVDofs(i, vdofs);
            vals.SetSize(vdofs.Size());
            for (int j = 0; j < target_nsp; j++)
            {
               for (int d = 0; d < target_vectordim; d++)
               {
                  // Arrange values byNodes
                  int idx = src_fes->GetOrdering() == Ordering::byNODES ? d * target_nsp * target_NE + i * target_nsp + j : i * target_nsp * mesh_dim + d + j * mesh_dim;
                  elem_dof_vals(j + d * target_nsp) = interp_vals(idx);
               }
            }
            func_target.SetSubVector(vdofs, elem_dof_vals);
         }
      }
   }
   else // H(div) or H(curl)
   {
      Array<int> vdofs;
      Vector vals;
      Vector elem_dof_vals(target_nsp * target_vectordim);

      for (int i = 0; i < pmesh_tar.GetNE(); i++)
      {
         tar_fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         for (int j = 0; j < target_nsp; j++)
         {
            for (int d = 0; d < target_vectordim; d++)
            {
               // Arrange values byVDim
               int idx = src_fes->GetOrdering() == Ordering::byNODES ? d * target_nsp * target_NE + i * target_nsp + j : i * target_nsp * mesh_dim + d + j * mesh_dim;
               elem_dof_vals(j * target_vectordim + d) = interp_vals(idx);
            }
         }
         tar_fes->GetFE(i)->ProjectFromNodes(elem_dof_vals, *tar_fes->GetElementTransformation(i), vals);
         func_target.SetSubVector(vdofs, vals);
      }
   }
}

L2Projector::L2Projector(const ParFiniteElementSpace &src_,
                         const ParFiniteElementSpace &tar_,
                         const IntegrationRule &ir_,
                         real_t bdr_tol_, 
                         bool periodic_, 
                         real_t size_y_,
                           real_t size_z_)
    : src_fes(&src_),
      tar_fes(&tar_),
      pmesh_src(*src_fes->GetParMesh()),
      pmesh_tar(*tar_fes->GetParMesh()),
      target_ir(ir_),
      target_qs(pmesh_tar, target_ir),
      periodic(periodic_),
      size_y(size_y_),
      size_z(size_z_)
{

   // check dimension
   mesh_dim = pmesh_tar.Dimension();
   MFEM_VERIFY(mesh_dim == pmesh_src.Dimension(), "Dimension mismatch between source and target meshes.");
   MFEM_VERIFY(mesh_dim > 1, "GSLIB requires a 2D or a 3D mesh");

   target_NE = pmesh_tar.GetNE();
   target_mesh_order = pmesh_tar.GetNodes()->FESpace()->GetElementOrder(0);
   target_fes_order = tar_fes->GetElementOrder(0);

   target_nsp = target_ir.GetNPoints();

   // ensure pmesh has nodes
   pmesh_src.EnsureNodes();
   pmesh_tar.EnsureNodes();

   // Generate list of points where the grid function will be evaluated.
   Vector vxyz;
   int point_ordering;
   {
      vxyz.SetSize(target_nsp * target_NE * mesh_dim);
      for (int i = 0; i < target_NE; i++)
      {
         const FiniteElement *fe = tar_fes->GetFE(i);
         ElementTransformation *et = tar_fes->GetElementTransformation(i);

         DenseMatrix pos;
         et->Transform(target_ir, pos);
         Vector rowx(vxyz.GetData() + i * target_nsp, target_nsp),
             rowy(vxyz.GetData() + i * target_nsp + target_NE * target_nsp, target_nsp),
             rowz;
         if (mesh_dim == 3)
         {
            rowz.SetDataAndSize(vxyz.GetData() + i * target_nsp + 2 * target_NE * target_nsp, target_nsp);
         }
         pos.GetRow(0, rowx);
         pos.GetRow(1, rowy);
         if (mesh_dim == 3)
         {
            pos.GetRow(2, rowz);
         }
      }
      point_ordering = Ordering::byNODES;
   }

   nodes_cnt = vxyz.Size() / mesh_dim;

   // set up finder
   int finder_size = 1;
   std::vector<Vector> vxyz_shifts;
   if(periodic)
   {
      if(mesh_dim == 2)
      {
         finder_size = 3;
         vxyz_shifts.resize(3);
         vxyz_shifts[0] = Vector({0.0, 0.0});
         vxyz_shifts[1] = Vector({0.0, size_y});
         vxyz_shifts[2] = Vector({0.0, -size_y});
      }
      else if(mesh_dim == 3)
      {
         finder_size = 9;
         vxyz_shifts.resize(9);
         vxyz_shifts[0] = Vector({0.0, 0.0, 0.0});
         vxyz_shifts[1] = Vector({0.0, 0.0, size_z});
         vxyz_shifts[2] = Vector({0.0, 0.0, -size_z});
         vxyz_shifts[3] = Vector({0.0, size_y, 0.0});
         vxyz_shifts[4] = Vector({0.0, size_y, size_z});
         vxyz_shifts[5] = Vector({0.0, size_y, -size_z});
         vxyz_shifts[6] = Vector({0.0, -size_y, 0.0});
         vxyz_shifts[7] = Vector({0.0, -size_y, size_z});
         vxyz_shifts[8] = Vector({0.0, -size_y, -size_z});
      }
   }
   finders.SetSize(finder_size);
   vxyz_shifted.resize(finder_size);
   for(int f = 0; f < finder_size; f++)
   {
      finders[f] = new FindPointsGSLIB();
      finders[f]->SetDistanceToleranceForPointsFoundOnBoundary(bdr_tol_);
      finders[f]->Setup(pmesh_src);
      vxyz_shifted[f] = vxyz;
      if(periodic && f > 0)
      {
         Vector rowx(vxyz_shifted[f].GetData(), nodes_cnt);
         Vector rowy(vxyz_shifted[f].GetData() + nodes_cnt, nodes_cnt);
         Vector rowz;
         if(mesh_dim == 3)
         {
            rowz.SetDataAndSize(vxyz_shifted[f].GetData() + 2 * nodes_cnt, nodes_cnt);
         }
         rowx += vxyz_shifts[f](0);
         rowy += vxyz_shifts[f](1);
         if(mesh_dim == 3)
         {
            rowz += vxyz_shifts[f](2);
         }
      }
      finders[f]->SetDefaultInterpolationValue(-1e20);
      finders[f]->FindPoints(vxyz_shifted[f], point_ordering);
   }
}

L2Projector::~L2Projector()
{
   for(int f = 0; f < finders.Size(); f++)
   {
      finders[f]->FreeData();
      delete finders[f];
   }
   vxyz_shifted.clear();
}

void L2Projector::Interpolate(const ParGridFunction &func_source, ParGridFunction &func_target)
{

   // Verify that the source/target grid functions are in the fespaces that the interpolator was created with
   MFEM_VERIFY(func_source.ParFESpace() == src_fes, "Source grid function is not in the source finite element space.");
   MFEM_VERIFY(func_target.ParFESpace() == tar_fes, "Target grid function is not in the target finite element space.");

   int target_vectordim = func_target.VectorDim();
   int finder_size = finders.Size();
   std::vector<Vector> interp_vals_vec(finder_size);
   for(int f = 0; f < finder_size; f++)
   {
      interp_vals_vec[f].SetSize(nodes_cnt * func_target.VectorDim());
      finders[f]->Interpolate(func_source, interp_vals_vec[f]);
   }
   
   Vector interp_vals_final(nodes_cnt * func_target.VectorDim());
   for(int i = 0; i < nodes_cnt * func_target.VectorDim(); i++)
   {
      real_t val = -1e20;
      for(int f = 0; f < finder_size; f++)
      {
         real_t val_f = interp_vals_vec[f](i);
         if(val_f > -1e15)
         {
            val = val_f;
            break;
         }
      }
      if(val < -1e15)
      {
         printf("[Rank %d] L2Projector warning: point %d could not be mapped to the source mesh!\n", Mpi::WorldRank(), i);
         printf(" val[0] = %g\n", interp_vals_vec[0](i));
         printf(" val[1] = %g\n", interp_vals_vec[1](i));
         printf(" val[2] = %g\n", interp_vals_vec[2](i));
         int i_node = i % nodes_cnt;
         printf(" f = 0: x[0] = %g, y[0] = %g\n", vxyz_shifted[0](i_node), vxyz_shifted[0](i_node + nodes_cnt));
         printf(" f = 1: x[1] = %g, y[1] = %g\n", vxyz_shifted[1](i_node), vxyz_shifted[1](i_node + nodes_cnt));
         printf(" f = 2: x[2] = %g, y[2] = %g\n", vxyz_shifted[2](i_node), vxyz_shifted[2](i_node + nodes_cnt));
      }
      interp_vals_final(i) = val;
   }

   QuadratureFunction target_qf(&target_qs, target_vectordim);

   // set the values to the quadrature function
   Vector tmp(target_vectordim);
   for (int i = 0; i < pmesh_tar.GetNE(); i++)
   {
      const FiniteElement *fe = tar_fes->GetFE(i);
      ElementTransformation *Tr = tar_fes->GetElementTransformation(i);
      for (int j = 0; j < target_nsp; j++)
      {
         const IntegrationPoint &ip = target_ir.IntPoint(j);
         Tr->SetIntPoint(&ip);

         target_qf.GetValues(i, j, tmp);
         for (int d = 0; d < target_vectordim; d++)
         {
            // Arrange values byNodes
            int idx = src_fes->GetOrdering() == Ordering::byNODES ? d * target_nsp * target_NE + i * target_nsp + j : i * target_nsp * mesh_dim + d + j * mesh_dim;
            tmp(d) = interp_vals_final(idx);
         }
      }
   }

   real_t glb_min_loc = interp_vals_final.Min();
   real_t glb_max_loc = interp_vals_final.Max();
   real_t glb_min, glb_max;
   MPI_Allreduce(&glb_min_loc, &glb_min, 1, MPI_DOUBLE, MPI_MIN, pmesh_tar.GetComm());
   MPI_Allreduce(&glb_max_loc, &glb_max, 1, MPI_DOUBLE, MPI_MAX, pmesh_tar.GetComm());
   if (Mpi::Root())
   {
      printf("L2Projector: global min = %g, global max = %g\n", glb_min, glb_max);
   }

   QuadratureFunctionCoefficient target_qf_coeff(target_qf);
   ReferenceMassIntegrator mass_integ;
   QuadratureLFIntegrator target_qf_lf_integ(target_qf_coeff);
   Array<int> vdofs;
   DenseMatrix pos_mass;
   Vector rhs_vec, vec;

   {
      const FiniteElement *fe = tar_fes->GetFE(0);
      ElementTransformation *Tr = tar_fes->GetElementTransformation(0);
      int Ndof = fe->GetDof();
      pos_mass.SetSize(Ndof, Ndof);
      mass_integ.AssembleElementMatrix(*fe, *Tr, pos_mass);
   }

   auto *inv = pos_mass.Inverse();

   int qp_count = 0;
   int qp_success = 0;

   for (int i = 0; i < pmesh_tar.GetNE(); i++)
   {
      const FiniteElement *fe = tar_fes->GetFE(i);
      int ndofs = fe->GetDof();
      ElementTransformation *Tr = tar_fes->GetElementTransformation(i);

      rhs_vec.SetSize(ndofs);
      vec.SetSize(ndofs);
      tar_fes->GetElementVDofs(i, vdofs);
      target_qf_lf_integ.AssembleRHSElementVect(*fe, *Tr, rhs_vec);
      inv->Mult(rhs_vec, vec);
      func_target.SetSubVector(vdofs, vec);

      // bound preserving
      real_t min_bound, max_bound;
      real_t loc_min, loc_max;
      DenseMatrix local_vals_mat;
      Vector local_vals_vec;
      target_qf.GetValues(i, local_vals_mat);
      local_vals_mat.GetRow(0, local_vals_vec);
      loc_min = local_vals_vec.Min();
      loc_max = local_vals_vec.Max();

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
         min_bound = loc_min;
         max_bound = loc_max;
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

      switch (limit_type)
      {
      case FCT:
      {
         // compute low-order coeffs
         real_t c_bar = rhs_vec.Sum();

         // lumped mass
         Vector m_lump(ndofs);
         m_lump = 0.0;
         for (int d = 0; d < ndofs; d++)
         {
            for (int e = 0; e < ndofs; e++)
            {
               m_lump(d) += pos_mass(d, e);
            }
         }

         Vector z(rhs_vec);
         for (int d = 0; d < ndofs; d++)
         {
            z(d) = rhs_vec(d) - c_bar * m_lump(d);
         }

         DenseMatrix fij(ndofs, ndofs);
         for (int a = 0; a < ndofs; a++)
         {
            for (int b = 0; b < ndofs; b++)
            {
               fij(a, b) = pos_mass(a, b) * (vec(a) - vec(b)) + (z(a) - z(b)) / real_t(ndofs);
            }
         }

         // FCT coeffs
         DenseMatrix beta(ndofs, ndofs);
         Vector deltac_p(ndofs), deltac_m(ndofs);
         Vector detJ(target_nsp);
         for (int j = 0; j < target_nsp; j++)
         {
            const IntegrationPoint &ip = target_ir.IntPoint(j);
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

         // if(Mpi::Root())
         // {
         //    printf("Element %d, min bound = %g, max bound = %g \n", i, min_bound*detJ_min, max_bound*detJ_max);
         //    printf("Element %d, c_bar = %g \n", i, c_bar);
         //    printf("vec min = %g, vec max = %g \n", vec.Min(), vec.Max());
         //    if(vec.Min()<min_bound*detJ_min || vec.Max()>max_bound*detJ_max)
         //    {
         //       printf("Element %d, Violation of bounds \n", i);
         //    }
         // }

         for (int a = 0; a < ndofs; a++)
         {
            vec(a) = c_bar;
            for (int b = 0; b < ndofs; b++)
            {
               vec(a) += 1.0 / m_lump(a) * beta(a, b) * fij(a, b);
            }
         }

         // if(Mpi::Root())
         // {
         //    printf("Element %d, min bound = %g, max bound = %g , vec min = %g, vec max = %g \n", i, min_bound*detJ_min, max_bound*detJ_max, vec.Min(), vec.Max());
         // }

         func_target.SetSubVector(vdofs, vec);

         break;
      }

      default:
         mfem_error("Unknown limiter type");
         break;
      }
   }

   // if(bound_preserving)
   // {
   //    int qp_count_glb, qp_success_glb;
   //    MPI_Reduce(&qp_count, &qp_count_glb, 1, MPI_INT, MPI_SUM, 0, pmesh_tar.GetComm());
   //    MPI_Reduce(&qp_success, &qp_success_glb, 1, MPI_INT, MPI_SUM, 0, pmesh_tar.GetComm());
   //    if(Mpi::Root())
   //    {
   //       printf("L2Projector: Number of QP needed: %d, Number of QP succeeded: %d \n", qp_count_glb, qp_success_glb);
   //    }
   // }

   GFMinMax(func_target, glb_min, glb_max, &target_qs.GetIntRule(0));
   if (Mpi::Root())
   {
      printf("L2Projector: global min = %g, global max = %g\n", glb_min, glb_max);
   }

   delete inv;
}