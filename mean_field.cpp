#include "mean_field.hpp"

namespace mfem
{
namespace
{
class MeanFieldPotential : public VectorCoefficient
{
    const ParGridFunction &displacement;
    Vector background;
    int dimension;
public:
    MeanFieldPotential(const ParGridFunction &u, const Vector &b)
        : VectorCoefficient(b.Size() == 2 ? 1 : 3),
          displacement(u), background(b), dimension(b.Size()) {}

    void Eval(Vector &value, ElementTransformation &T,
              const IntegrationPoint &ip) override
    {
        T.SetIntPoint(&ip);
        Vector u(dimension);
        displacement.GetVectorValue(T, ip, u);
        value.SetSize(GetVDim());
        if (dimension == 2)
        {
            // curl(Bx*(Y-y)-By*(X-x)) = transported B0 - physical B0.
            value(0) = background(0)*u(1)-background(1)*u(0);
        }
        else
        {
            // Homotopy of the magnetic 2-form from x to X=x+u:
            // a = (I + grad(u)^T/2) (B0 cross u).
            // Its curl is the pushed-forward reference flux minus B0.
            Vector cross(3);
            cross(0) = background(1)*u(2)-background(2)*u(1);
            cross(1) = background(2)*u(0)-background(0)*u(2);
            cross(2) = background(0)*u(1)-background(1)*u(0);
            DenseMatrix gradient(3);
            displacement.GetVectorGradient(T, gradient);
            value = cross;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    value(i) += 0.5*gradient(j,i)*cross(j);
        }
    }
};
}

void AddUniformMeanFieldPotential(ParGridFunction &potential,
                                 const ParGridFunction &reference_nodes,
                                 const ParGridFunction &current_nodes,
                                 const Vector &uniform_field)
{
    MFEM_VERIFY(reference_nodes.ParFESpace() == current_nodes.ParFESpace(),
                "Mean-field geometry spaces must match");
    ParGridFunction displacement(reference_nodes.ParFESpace());
    displacement = reference_nodes;
    displacement.Add(-1.0, current_nodes);
    MeanFieldPotential coefficient(displacement, uniform_field);
    ParGridFunction correction(potential.ParFESpace());
    correction.ProjectCoefficient(coefficient);
    potential.Add(1.0, correction);
}
}
