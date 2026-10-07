#ifndef REMAP_HPP
#define REMAP_HPP

#include "mfem.hpp"
#include "Interpolator.hpp"

namespace mfem
{

   // Base class for remapping schemes
   class Remapping
   {
   protected:
   public:
      Remapping() {}
      virtual ~Remapping() {}

      virtual void Remap(ParGridFunction &mesh_velocity, ParGridFunction &gf) = 0;
   };

   class ExactRemap : public Remapping
   {
   protected:
      Coefficient *exact_coeff;
      VectorCoefficient *exact_vec_coeff;
      bool Bernstein = false;
      bool integral = false;

   public:
      ExactRemap() : exact_coeff(nullptr), exact_vec_coeff(nullptr) {}

      void SetExactCoefficient(Coefficient *exact_coeff_, real_t t)
      {
         exact_coeff = exact_coeff_;
         if (exact_coeff)
         {
            exact_coeff->SetTime(t);
         }
      }

      void SetExactVectorCoefficient(VectorCoefficient *exact_vec_coeff_, real_t t)
      {
         exact_vec_coeff = exact_vec_coeff_;
         if (exact_vec_coeff)
         {
            exact_vec_coeff->SetTime(t);
         }
      }

      void SetPositive() { Bernstein = true; }

      void SetIntegralType() { integral = true; }

      virtual void Remap(ParGridFunction &mesh_velocity, ParGridFunction &gf);
   };

   class InterpolateRemap : public Remapping
   {
   protected:
   public:
      InterpolateRemap() {}

      virtual void Remap(ParGridFunction &mesh_velocity, ParGridFunction &gf);
   };

   class L2ProjectRemap : public Remapping
   {
   protected:
      BoundPreservingType bp_type = BoundPreservingType::NONE;
      const IntegrationRule *ir = nullptr;
      
      bool periodic = false;
      real_t size_x;
      real_t size_y;
      real_t size_z;
   public:
      L2ProjectRemap() : size_x(-1.0), size_y(1.0), size_z(1.0) {}
      
      void SetBoundPreservingType(BoundPreservingType bp){bp_type = bp;};
      
      void SetPeriodic(bool periodic_, real_t size_y_, real_t size_z_, real_t size_x_ = -1.0)
      {
         periodic = periodic_;
         size_x = size_x_;
         size_y = size_y_;
         size_z = size_z_;
      }
      
      void SetIntegrationRule(const IntegrationRule &ir_) {ir = &ir_;}
      
      virtual void Remap(ParGridFunction &mesh_velocity, ParGridFunction &gf);
   };
   
   /* Helicity preserving remap for magnetic vector potential */
   /* For H(curl) only */
   class HPRemap : public Remapping
   {
   protected:
      ParMesh *pmesh;
      int mesh_order;
      ParFiniteElementSpace *fes_A;
      ParFiniteElementSpace *fes_B;

      real_t mu;
      ConstantCoefficient mu_coeff;
      VectorFunctionCoefficient *H_bdry_coeff;
      Array<int> ess_bdr_H;
      bool ess_A;

   public:
      HPRemap(ParMesh &pmesh_,
              ParFiniteElementSpace *fes_A_,
              ParFiniteElementSpace *fes_B_,
              real_t mu_,
              VectorFunctionCoefficient *H_bdry_coeff_,
              Array<int> ess_bdr_H_);

      virtual ~HPRemap();

      virtual void Remap(ParGridFunction &mesh_velocity, ParGridFunction &A_gf) override;

      void UpdateA(ParGridFunction &mesh_velocity,
                   ParGridFunction &A_old_gf,
                   ParGridFunction &H_gf,
                   ParGridFunction &A_gf,
                   ParBilinearForm &A_mass_bf,
                   real_t dt,
                   const IntegrationRule *ir);

      void ProjectBToH(ParGridFunction &B_gf, ParGridFunction &H_gf, ParBilinearForm &H_mass_bf);

      void SetEssentialA(bool ess_A_) { ess_A = ess_A_; };
   };

   class DG_Evolution : public TimeDependentOperator
   {
   private:
      ParBilinearForm *M_bf, *K_bf;
      ParLinearForm *b_lf;
      ParMesh *pmesh;
      ParGridFunction *nodes;
      ParFiniteElementSpace *mesh_fes;
      ParGridFunction *mesh_velocity;
      mutable Solver *M_prec;
      mutable CGSolver M_solver;

      int vdim;
      mutable Vector z;

   public:
      DG_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, ParLinearForm &b_, ParGridFunction *nodes_, ParGridFunction *mesh_velocity_, int vdim_);

      void Mult(const Vector &x, Vector &y) const override;
      void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

      void SetTime(const real_t t) override;

      ~DG_Evolution() override;
   };

   /* DG remap for L2/H1 spaces */
   class DGRemap : public Remapping
   {
   protected:
   public:
      DGRemap() {}

      virtual ~DGRemap() {};

      virtual void Remap(ParGridFunction &mesh_velocity, ParGridFunction &u_gf) override;
   };

   /* H1DG remap for H1 spaces */
   class H1DGRemap : public Remapping
   {
   protected:
   public:
      H1DGRemap() {};

      virtual ~H1DGRemap() {};

      virtual void Remap(ParGridFunction &mesh_velocity, ParGridFunction &u_gf) override;
   };

}

#endif