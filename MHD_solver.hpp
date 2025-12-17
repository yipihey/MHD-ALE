#ifndef MFEM_MHD_SOLVER
#define MFEM_MHD_SOLVER

#include "mfem.hpp"
#include "MHD_assembly.hpp"
#include "mesh_smoother.hpp"
#include "Problemdata.hpp"
#include "tools.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{
   
enum MeshSmoothType
{
   NONE = 0, // No smoothing
   INITIAL = 1, // Initial smoothing
   LIMITED_HARMONIC = 2, // Limited harmonic smoothing
   PERIODIC = 3, // Periodic smoothing
};

enum RemapType
{
   INTERPOLATE = 0, // Interpolation
   DGconvection = 1, // discontinuous Galerkin method
   HelicityPreserving = 2, 
   L2Projection = 3,
   Exact=99, // only for testing!
};

namespace hydrodynamics
{

   
/// Visualize the given parallel grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

// Given a solutions state (x, v, e), this class performs all necessary
// computations to evaluate the new slopes (dx_dt, dv_dt, de_dt).
class LagrangianHydroOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fes_x, &fes_v, &fes_e, &fes_rho;
   ParMesh *pmesh;
   // FE spaces local and global sizes
   const int xVsize;
   const int vVsize;
   const int vTVSize;
   const HYPRE_BigInt vGTVSize;
   const int eVsize;
   const int eTVSize;
   const HYPRE_BigInt eGTVSize;
   const int rhoVsize;
   const int rhoTVSize;
   const HYPRE_BigInt rhoGTVSize;
   const int order_v;
   const int order_e;
   const int order_rho;
   Array<int> block_offsets;
   // Reference to the current mesh configuration.
   mutable ParGridFunction x_gf;
   const Array<int> &ess_tdofs;
   const int dim, NE, edofs_cnt;
   const double cfl;
   const bool use_viscosity, use_vorticity;
   const double cg_rel_tol;
   const int cg_max_iter;
   const ParGridFunction &gamma_gf;
   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm *Mv;
   SparseMatrix Mv_spmat_copy;
   DenseTensor Me, Me_inv;
   
   ParGridFunction &rho_gf;
   ParGridFunction &A_gf;
   ParGridFunction &B_gf;
   ParFiniteElementSpace &fes_A;
   ParFiniteElementSpace &fes_B;
   ParFiniteElementSpace &fes_divB;
   const int order_A;
   const int order_B;
   real_t mu;
   VectorFunctionCoefficient &H_bdry_coeff;
   Array<int> ess_bdr_H;
   ProblemData *pd;
   
   // Integration rule for all assemblies.
   int order_q;
   const IntegrationRule &ir;
   // Data associated with each quadrature point in the mesh.
   // These values are recomputed at each time step.
   mutable QuadratureData qdata;
   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it is used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm *Force;
   // Same as above, but done through partial assembly.
   // Linear solver for energy.
   mutable Vector one, rhs, e_rhs;
   
   // ALE
   MeshSmoother *mesh_smoother;
   ParGridFunction *nodes; // pmesh->nodes
   MeshSmoothType mesh_smooth_type;
   real_t mesh_smooth_eps;
   bool fix_step_remesh;
   int fix_step_remesh_interval;
   real_t min_detJ;
   real_t max_detJ;
   real_t max_ratio;
   real_t max_disp;
   
   RemapType remap_type_v;
   RemapType remap_type_e;
   RemapType remap_type_A;
   RemapType remap_type_rho;
   BoundPreservingType bp_type;

   virtual void ComputeMaterialProperties(int nvalues, const double gamma[],
                                          const double rho[], const double e[], const Vector B[],
                                          double p[], double cs[]) const
   {
      for (int v = 0; v < nvalues; v++)
      {
         p[v]  = (gamma[v] - 1.0) * rho[v] * e[v];
         cs[v] = sqrt(gamma[v] * (gamma[v]-1.0) * e[v] + B[v].Norml2()/sqrt(mu*rho[v]));
         ;
      }
   }

   void UpdateQuadratureData(const Vector &S) const;
   void AssembleForceMatrix() const;

public:
   LagrangianHydroOperator(const int size,
                           ParMesh *pmesh_,
                           ParFiniteElementSpace &x_fes,
                           ParFiniteElementSpace &v_fes,
                           ParFiniteElementSpace &e_fes,
                           ParFiniteElementSpace &rho_fes,
                           const Array<int> &ess_tdofs,
                           ParGridFunction &rho0_gf,
                           ParGridFunction &gamma_gf,
                           const double cfl,
                           const bool visc, const bool vort, 
                           const double cgt, const int cgiter,
                           const int order_q,
                           ParGridFunction &A_gf,
                           ParGridFunction &B_gf,
                           ParFiniteElementSpace &divB_fes_,
                           real_t mu_,
                           VectorFunctionCoefficient &H_bdry_coeff_, 
                           Array<int> ess_bdr_H_,
                           MeshSmoothType mesh_smooth_type_,
                           real_t mesh_smooth_eps_,
                           ProblemData *pd_,
                           bool fix_step_remesh,
                           int fix_step_remesh_interval);
   ~LagrangianHydroOperator();

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   virtual MemoryClass GetMemoryClass() const
   { return Device::GetMemoryClass(); }

   void SolveVelocity(const Vector &S, Vector &dS_dt) const;
   void SolveEnergy(const Vector &S, const Vector &v, Vector &dS_dt) const;
   void UpdateMesh(const Vector &S) const;
   
   void SetVelocityBoundaryValue(ParGridFunction &v_gf);

   // Calls UpdateQuadratureData to compute the new qdata.dt_estimate.
   double GetTimeStepEstimate(const Vector &S) const;
   void ResetTimeStepEstimate() const;

   double InternalEnergy(const ParGridFunction &e) const;
   double KineticEnergy(const ParGridFunction &v) const;

   const Array<int> &GetBlockOffsets() const { return block_offsets; }
   
   void AssembleMvMe();
   void ComputeMeshQuality(real_t &min_detJ, real_t &max_detJ, real_t &singular_ratio) const;
   bool NeedRemesh();
   void RemeshAndRemap(Vector &S);
   void SetRemeshParameters(real_t min_detJ_, real_t max_detJ_, real_t max_ratio_, real_t max_disp_)
   { min_detJ = min_detJ_; max_detJ = max_detJ_; max_ratio = max_ratio_; max_disp = max_disp_; }
   void SetRemapType_v(RemapType rt){ remap_type_v = rt; }
   void SetRemapType_e(RemapType rt){ remap_type_e = rt; }
   void SetRemapType_A(RemapType rt){ remap_type_A = rt; }
   void SetRemapType_rho(RemapType rt, BoundPreservingType bp){ remap_type_rho = rt; bp_type = bp; }
   
   const IntegrationRule & GetIntegrationRule() {return ir;}
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_MHD
