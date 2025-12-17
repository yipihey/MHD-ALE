#include "vecgf_error.hpp"

namespace mfem
{

    real_t ComputeVectorGradError(ParGridFunction &sol,
                                  VectorCoefficient &exgrad,
                                  const IntegrationRule *irs[])
    {
        ParFiniteElementSpace *fes = sol.ParFESpace();
        real_t error = 0.0;
        const FiniteElement *fe;
        ElementTransformation *Tr;
        DenseMatrix grad;
        int intorder;
        int sdim = fes->GetMesh()->SpaceDimension();
        int vdim = fes->GetVDim();
        Vector vec(vdim * sdim);

        for (int i = 0; i < fes->GetNE(); i++)
        {
            fe = fes->GetFE(i);
            Tr = fes->GetElementTransformation(i);
            intorder = 2 * fe->GetOrder() + 3; // <--------
            const IntegrationRule *ir;
            if (irs)
            {
                ir = irs[fe->GetGeomType()];
            }
            else
            {
                ir = &(IntRules.Get(fe->GetGeomType(), intorder));
            }
            for (int j = 0; j < ir->GetNPoints(); j++)
            {
                const IntegrationPoint &ip = ir->IntPoint(j);
                Tr->SetIntPoint(&ip);
                sol.GetVectorGradient(*Tr, grad);
                exgrad.Eval(vec, *Tr, ip);
                for (int d = 0; d < vdim; d++)
                {
                    for (int k = 0; k < sdim; k++)
                    {
                        vec(d * sdim + k) -= grad(d, k);
                    }
                }
                error += ip.weight * Tr->Weight() * (vec * vec);
            }
        }
        
        real_t glb_error = 0.0;
        MPI_Allreduce(&error, &glb_error, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, 
                      fes->GetComm());
                      
        return sqrt(glb_error);
    }
    
    
    real_t ComputeVectorH1Error(ParGridFunction &sol,
                                VectorCoefficient &exsol, 
                                VectorCoefficient &exgrad,
                                const IntegrationRule *irs[])
    {
        real_t L2error = sol.ComputeL2Error(exsol, irs);
        real_t GradError = ComputeVectorGradError(sol, exgrad, irs);
        
        return sqrt(L2error * L2error + GradError * GradError);
    }

}
