#include "MHD_assembly.hpp"
#include <unordered_map>

namespace mfem
{

namespace hydrodynamics
{
   
void Bvec2stress(const Vector &Bvec, DenseMatrix &Stressmat)
{
   const int dim = Stressmat.Height();
   real_t B2 = 0.0;
   for (int i = 0; i < dim; i++)
   {
      B2 += Bvec(i)*Bvec(i);
   }
   B2 *= 0.5;
   Stressmat = 0.0;
   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         Stressmat(i,j) = Bvec(i)*Bvec(j);
      }
      Stressmat(i,i) -= B2;
   }
   
}
      
void LorentzForceIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                           ElementTransformation &Tr,
                           Vector &elvect)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();
   const int vdim = B_gf.VectorDim();
   const int sdim = Tr.GetSpaceDim();
   
   MFEM_VERIFY(dim == sdim, "Dimension mismatch");
   MFEM_VERIFY(vdim == dim, "Vector dimension mismatch");

   dshape.SetSize(dof,dim);
   elvect.SetSize(dof*dim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      mfem_error("Integration rule not set in LorentzForceIntegrator");
   }
   
   Vector Bvec(dim);
   DenseMatrix Stressmat(dim,dim);
   DenseMatrix StressDotDshape(dim,dof);

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);

      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);
      
      B_gf.GetVectorValue(Tr, ip, Bvec);
      Bvec2stress(Bvec, Stressmat);
      Stressmat *= 1.0/mu*ip.weight * Tr.Weight();
      
      MultABt(Stressmat, dshape, StressDotDshape);
      
      for (int i = 0; i < dof; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            elvect(i+j*dof) += StressDotDshape(j,i);
         }
      }      
   }
}

void ForceIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                             const FiniteElement &test_fe,
                                             ElementTransformation &Tr,
                                             DenseMatrix &elmat)
{
   const int e = Tr.ElementNo;
   const int nqp = IntRule->GetNPoints();
   const int dim = trial_fe.GetDim();
   const int h1dofs_cnt = test_fe.GetDof();
   const int l2dofs_cnt = trial_fe.GetDof();
   elmat.SetSize(h1dofs_cnt*dim, l2dofs_cnt);
   elmat = 0.0;
   DenseMatrix vshape(h1dofs_cnt, dim), loc_force(h1dofs_cnt, dim);
   Vector shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt*dim);
   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = IntRule->IntPoint(q);
      // Form stress:grad_shape at the current point.
      test_fe.CalcDShape(ip, vshape);
      for (int i = 0; i < h1dofs_cnt; i++)
      {
         for (int vd = 0; vd < dim; vd++) // Velocity components.
         {
            loc_force(i, vd) = 0.0;
            for (int gd = 0; gd < dim; gd++) // Gradient components.
            {
               const int eq = e*nqp + q;
               const double stressJinvT = qdata.stressJinvT(vd)(eq, gd);
               loc_force(i, vd) +=  stressJinvT * vshape(i,gd);
            }
         }
      }
      trial_fe.CalcShape(ip, shape);
      AddMultVWt(Vloc_force, shape, elmat);
   }
}

void MagneticBoundaryIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                           FaceElementTransformations &Tr,
                           Vector &elvect)
{
   const int dof = el.GetDof();
   const int vdim = B_gf.VectorDim();
   const int sdim = Tr.GetSpaceDim();
   
   shape.SetSize(dof);
   elvect.SetSize(dof*sdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      mfem_error("Integration rule not set in MagneticBoundaryIntegrator");
   }
   
   Vector Bvec(sdim);
   Vector normal(sdim);
   real_t Bdotn;

   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetAllIntPoints(&ip);
      
      const real_t w = ip.weight / mu ;
      
      el.CalcPhysShape(Tr, shape);
      B_gf.GetVectorValue(*(Tr.Elem1), Tr.Elem1->GetIntPoint(), Bvec);
      CalcOrtho(Tr.Jacobian(), normal);
      Bdotn = Bvec * normal;
      
      for (int i = 0; i < dof; i++)
      {
         for (int j = 0; j < sdim; j++)
         {
            elvect(i+j*dof) += w * Bdotn * Bvec(j) * shape(i);
         }
      }      
   }
}

} // namespace hydrodynamics

} // namespace mfem
