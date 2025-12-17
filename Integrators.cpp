#include "Integrators.hpp"
#include "mfem.hpp"

namespace mfem
{
    ReferenceMassIntegrator::ReferenceMassIntegrator(const IntegrationRule *ir)
        : BilinearFormIntegrator(ir), Q(nullptr), maps(nullptr), geom(nullptr)
    {
    }

    ReferenceMassIntegrator::ReferenceMassIntegrator(Coefficient &q, const IntegrationRule *ir)
        : ReferenceMassIntegrator(ir)
    {
        Q = &q;
    }

    void ReferenceMassIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans,
                                                        DenseMatrix &elmat)
    {
        int nd = el.GetDof();
        // int dim = el.GetDim();
        real_t w;

#ifdef MFEM_THREAD_SAFE
        Vector shape;
#endif
        elmat.SetSize(nd);
        shape.SetSize(nd);

        const IntegrationRule *ir = GetIntegrationRule(el, Trans);
        elmat = 0.0;
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);

            el.CalcShape(ip, shape);

            w = ip.weight;

            AddMult_a_VVt(w, shape, elmat);
        }
    }

    void ReferenceMassIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
    {
        int tr_nd = trial_fe.GetDof();
        int te_nd = test_fe.GetDof();
        real_t w;

#ifdef MFEM_THREAD_SAFE
        Vector shape, te_shape;
#endif
        elmat.SetSize(te_nd, tr_nd);
        shape.SetSize(tr_nd);
        te_shape.SetSize(te_nd);

        const IntegrationRule *ir = GetIntegrationRule(trial_fe, test_fe, Trans);
        elmat = 0.0;
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Trans.SetIntPoint(&ip);

            trial_fe.CalcShape(ip, shape);
            test_fe.CalcShape(ip, te_shape);

            w = ip.weight;

            te_shape *= w;
            AddMultVWt(te_shape, shape, elmat);
        }
    }

    const IntegrationRule &ReferenceMassIntegrator::GetRule(const FiniteElement &trial_fe,
                                                            const FiniteElement &test_fe,
                                                            const ElementTransformation &Trans)
    {
        // int order = trial_fe.GetOrder() + test_fe.GetOrder();
        const int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() + 2;

        if (trial_fe.Space() == FunctionSpace::rQk)
        {
            return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
        }
        return IntRules.Get(trial_fe.GetGeomType(), order);
    }

    void ReferenceLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                       ElementTransformation &Tr,
                                                       Vector &elvect)
    {
        int dof = el.GetDof();

        shape.SetSize(dof); // vector of size dof
        elvect.SetSize(dof);
        elvect = 0.0;

        const IntegrationRule *ir = GetIntegrationRule(el, Tr);

        if (ir == NULL)
        {
            ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        }

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);

            Tr.SetIntPoint(&ip);
            real_t val = Tr.Weight() * Q.Eval(Tr, ip);

            el.CalcShape(ip, shape);

            add(elvect, ip.weight * val, shape, elvect);
        }
    }

}