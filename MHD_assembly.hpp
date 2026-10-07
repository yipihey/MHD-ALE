#ifndef MFEM_MHD_ASSEMBLY
#define MFEM_MHD_ASSEMBLY

#include "mfem.hpp"
#include "general/forall.hpp"
#include "linalg/dtensor.hpp"

namespace mfem
{

namespace hydrodynamics
{

// Container for all data needed at quadrature points.
struct QuadratureData
{
   // Reference to physical Jacobian for the initial mesh.
   // These are computed only at time zero and stored here.
   DenseTensor Jac0inv;

   // Quadrature data used for full/partial assembly of the force operator.
   // At each quadrature point, it combines the stress, inverse Jacobian,
   // determinant of the Jacobian and the integration weight.
   // It must be recomputed in every time step.
   DenseTensor stressJinvT;

   // Same combination for the Maxwell stress (B_i B_j - B^2/2 delta_ij)/mu.
   // It enters only the momentum equation, so it is kept apart from the
   // hydrodynamic stress that also drives the internal energy equation.
   DenseTensor maxwellJinvT;

   // Initial length scale. This represents a notion of local mesh size.
   // We assume that all initial zones have similar size.
   double h0;

   // Estimate of the minimum time step over all quadrature points. This is
   // recomputed at every time step to achieve adaptive time stepping.
   double dt_est;

   QuadratureData(int dim, int NE, int quads_per_el)
      : Jac0inv(dim, dim, NE * quads_per_el),
        stressJinvT(NE * quads_per_el, dim, dim),
        maxwellJinvT(NE * quads_per_el, dim, dim) { }
};


// Class for the linear integrator \int (sigma, grad v) dx
// where sigma is the Maxwell stress tensor
// sigma = 1/mu*(B_i B_j - 1/2 B^2 I) 
// and v is a vector valued test function.
class LorentzForceIntegrator : public LinearFormIntegrator
{
private:
   real_t mu;
   ParGridFunction &B_gf;
   DenseMatrix dshape;

public:
   LorentzForceIntegrator(ParGridFunction &B_gf_, real_t mu_)
      : LinearFormIntegrator(), mu(mu_), B_gf(B_gf_) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override;

   using LinearFormIntegrator::AssembleRHSElementVect;
};

// Linear form \int (sigma : grad w) dx for a vector test function w, where
// sigma*Jinv^T*weight*detJ is taken from a precomputed quadrature tensor
// (e.g. QuadratureData::maxwellJinvT). Only reference gradients are needed.
class QuadratureForceLFIntegrator : public LinearFormIntegrator
{
private:
   const DenseTensor &JinvT;
   DenseMatrix dshape;

public:
   QuadratureForceLFIntegrator(const DenseTensor &JinvT_)
      : LinearFormIntegrator(), JinvT(JinvT_) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override;

   using LinearFormIntegrator::AssembleRHSElementVect;
};

// Performs full assembly for the force operator.
class ForceIntegrator : public BilinearFormIntegrator
{
private:
   const QuadratureData &qdata;
public:
   ForceIntegrator(QuadratureData &qdata) : qdata(qdata) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};


// Class for the linear integrator \int n cdot sigma cdot v ds
// on the boundary, where sigma = 1/mu*(B_i B_j)
class MagneticBoundaryIntegrator : public LinearFormIntegrator
{
private:
   real_t mu;
   ParGridFunction &B_gf;
   Vector shape;

public:
   MagneticBoundaryIntegrator(ParGridFunction &B_gf_, real_t mu_)
      : LinearFormIntegrator(), mu(mu_), B_gf(B_gf_) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect) override;

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override
   {
      MFEM_ABORT("MagneticBoundaryIntegrator::AssembleRHSElementVect: "
                 "not implemented for volume elements.");
   }
   using LinearFormIntegrator::AssembleRHSElementVect;
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_MHD_ASSEMBLY
