#include "Interpolator.hpp"
#include "tools.hpp"
#include "mesh_smoother.hpp"
#include <iostream>

using namespace mfem;

int main(int argc, char **argv)
{
    Mpi::Init(argc,argv);
    Hypre::Init();
    bool passed = true;
    for (int dim : {2,3})
    for (int axis : {0,1,2}) // x only, y only (legacy API), all axes
    for (auto ordering : {Ordering::byNODES,Ordering::byVDIM})
    {
        Mesh mesh = dim == 2
            ? Mesh::MakeCartesian2D(4,4,Element::QUADRILATERAL,true)
            : Mesh::MakeCartesian3D(4,4,4,Element::HEXAHEDRON);
        Vector lengths(dim), shift(dim);
        lengths = -1.0;
        shift = 0.0;
        std::vector<Vector> translations;
        for (int d=0; d<dim; ++d)
            if (axis == 2 || axis == d)
            {
                lengths(d) = 1.0;
                shift(d) = d == 0 ? 4.37 : (d == 1 ? -3.19 : 2.41);
                Vector translation(dim);
                translation = 0.0;
                translation(d) = 1.0;
                translations.push_back(translation);
            }
        Mesh periodic = Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping(translations));
        periodic.SetCurvature(2,true,-1,ordering);
        ParMesh source(MPI_COMM_WORLD,periodic), target(MPI_COMM_WORLD,periodic);
        InitialSmoother smoother(source);
        smoother.ConfigurePeriodic(lengths,true);
        auto *nodes = dynamic_cast<ParGridFunction *>(source.GetNodes());
        auto *space = nodes->ParFESpace();
        for (int i=0; i<space->GetNDofs(); ++i)
            for (int d=0; d<dim; ++d)
                (*nodes)(space->DofToVDof(i,d)) += shift(d);
        source.NodesUpdated();
        ParGridFunction rezoned(space), expected(space);
        expected = *nodes;
        smoother.Smooth(rezoned);
        rezoned -= expected;
        real_t local_error = rezoned.Normlinf(), shift_error;
        MPI_Allreduce(&local_error,&shift_error,1,MPITypeMap<real_t>::mpi_type,
                      MPI_MAX,source.GetComm());
        real_t displacement = smoother.MaxDisplacement();
        passed = passed && shift_error < 1e-12 && displacement < 1e-12;
        // The density remapper uses the integral-preserving reference map.
        L2_FECollection fec(1,dim,BasisType::Positive,FiniteElement::INTEGRAL);
        ParFiniteElementSpace src_fes(&source,&fec), tar_fes(&target,&fec);
        ParGridFunction src(&src_fes), result(&tar_fes);
        ConstantCoefficient constant(1.25);
        src.ProjectCoefficient(constant);
        const auto &ir = IntRules.Get(target.GetElementBaseGeometry(0),6);
        L2Projector projector(src_fes,tar_fes,ir,1e-8,true,
                              lengths(1),dim == 3 ? lengths(2) : -1.0,lengths(0));
        projector.Interpolate(src,result);
        real_t error = result.ComputeL2Error(constant);
        if (Mpi::Root())
            std::cout << "dim=" << dim << " axis=" << axis << " ordering=" << ordering
                      << " projection_L2=" << error << " comoving_error=" << shift_error
                      << " displacement=" << displacement << std::endl;
        passed = passed && error < 1e-10;
    }
    return passed ? 0 : 1;
}
