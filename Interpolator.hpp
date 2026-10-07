#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include "mfem.hpp"
#include "tools.hpp"

using namespace mfem;

namespace mfem
{

    /*
    Class for interpolation between different meshes
    based on field-interp miniapp
    */
    class Interpolator
    {
    protected:
        const ParFiniteElementSpace *src_fes;
        const ParFiniteElementSpace *tar_fes;

        ParMesh &pmesh_src;
        ParMesh &pmesh_tar;

        int mesh_dim;

        int target_NE;
        int target_mesh_order;
        int target_fes_order;
        int target_fieldtype;
        int target_nsp;

        int nodes_cnt;
        FindPointsGSLIB *finder;

    public:
        Interpolator(const ParFiniteElementSpace &src_fes, const ParFiniteElementSpace &tar_fes, real_t bdr_tol_ = 1e-8);

        ~Interpolator();

        void Interpolate(const ParGridFunction &func_source, ParGridFunction &func_target);
    };

    /*
    Class for interpolation between different meshes
    based on field-interp miniapp
    Using Local L^2 Projection
    Only for density! Bernstein basis + integral map type
    */
    class L2Projector
    {
    protected:
        const ParFiniteElementSpace *src_fes;
        const ParFiniteElementSpace *tar_fes;

        ParMesh &pmesh_src;
        ParMesh &pmesh_tar;

        int mesh_dim;

        int target_NE;
        int target_mesh_order;
        int target_fes_order;
        int target_fieldtype;
        int target_nsp;

        int nodes_cnt;
        Array<FindPointsGSLIB*> finders;
        
        std::vector<Vector> vxyz_shifted;
        
        const IntegrationRule &target_ir;
        QuadratureSpace target_qs;
        
        // bool bound_preserving = false;
        BoundPreservingType bp_type = BoundPreservingType::NONE;
        
        // Positive lengths enable periodic images independently in each axis.
        // The trailing x length preserves the original y/z calling convention.
        bool periodic = false;
        real_t size_x;
        real_t size_y; 
        real_t size_z;

    public:
    
        enum LimitType{FCT};
        LimitType limit_type = FCT;
        
        L2Projector(const ParFiniteElementSpace &src_fes, const ParFiniteElementSpace &tar_fes, const IntegrationRule &ir_,  real_t bdr_tol_ = 1e-8, bool periodic_ = false, real_t size_y_ = 1.0, real_t size_z_ = 1.0, real_t size_x_ = -1.0);

        ~L2Projector();
        
        // void EnableBoundPreserving() { bound_preserving = true;};
        
        void SetBoundPreservingType(BoundPreservingType bt) { bp_type = bt; }

        void Interpolate(const ParGridFunction &func_source, ParGridFunction &func_target);
        
        void SetLimitType(LimitType lt) { limit_type = lt; }
        
    };

}

#endif
