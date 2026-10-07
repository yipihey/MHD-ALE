#include "mean_field.hpp"
#include "tools.hpp"
#include <iostream>
#include <iomanip>

using namespace mfem;

int main(int argc, char **argv)
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    bool passed = true;
    for (int dim : {2,3})
    for (real_t amplitude : {0.0,0.05,0.2})
    {
        Mesh mesh = dim == 2
            ? Mesh::MakeCartesian2D(2,2,Element::QUADRILATERAL,true)
            : Mesh::MakeCartesian3D(2,2,2,Element::HEXAHEDRON);
        mesh.SetCurvature(2, false, -1, Ordering::byNODES);
        ParMesh pmesh(MPI_COMM_WORLD, mesh);
        auto *nodes = dynamic_cast<ParGridFunction *>(pmesh.GetNodes());
        ParGridFunction reference(*nodes), distorted(nodes->ParFESpace());
        int order = dim == 2 ? 2 : 4;
        std::unique_ptr<FiniteElementCollection> potential_fec;
        if (dim == 2) { potential_fec.reset(new H1_FECollection(order,dim)); }
        else { potential_fec.reset(new ND_FECollection(order,dim)); }
        RT_FECollection flux_fec(order-1,dim);
        ParFiniteElementSpace potential_fes(&pmesh,potential_fec.get());
        ParFiniteElementSpace flux_fes(&pmesh,&flux_fec);
        ParGridFunction potential(&potential_fes), material(&flux_fes);
        ParGridFunction reconstructed(&flux_fes), physical_background(&flux_fes);
        Vector background(dim);
        background(0)=0.3; background(1)=-0.2;
        if (dim == 3) { background(2)=0.4; }
        VectorFunctionCoefficient constant_field(dim,[&](const Vector &, Vector &b) { b=background; });
        material.ProjectCoefficient(constant_field);
        VectorFunctionCoefficient map(dim,[&](const Vector &x, Vector &y)
        {
            y=x;
            y(0) += amplitude*x(0)*(1-x(0))*x(1)*(1-x(1));
            if (dim == 2)
                y(1) -= 0.7*amplitude*x(0)*(1-x(0))*x(1)*(1-x(1));
            else
            {
                y(1) -= 0.7*amplitude*x(1)*(1-x(1))*x(2)*(1-x(2));
                y(2) += 0.6*amplitude*x(2)*(1-x(2))*x(0)*(1-x(0));
            }
        });
        distorted.ProjectCoefficient(map);
        *nodes = distorted;
        pmesh.NodesUpdated();
        potential=0.0;
        AddUniformMeanFieldPotential(potential,reference,*nodes,background);
        ComputeCurl(potential,reconstructed);
        physical_background.ProjectCoefficient(constant_field);
        reconstructed += physical_background;
        real_t divergence = GFDivError(reconstructed);
        reconstructed -= material;
        real_t relative_error=sqrt(GFInnerProduct(reconstructed,reconstructed) /
                                   GFInnerProduct(material,material));
        if (Mpi::Root())
            std::cout << std::scientific << std::setprecision(16)
                      << "dim=" << dim << " amplitude=" << amplitude
                      << " relative_L2=" << relative_error
                      << " divergence_L2=" << divergence << std::endl;
        passed = passed && relative_error < 1e-10 && divergence < 1e-11;
    }
    return passed ? 0 : 1;
}
