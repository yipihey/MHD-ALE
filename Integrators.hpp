#ifndef _INTEGRATORS_HPP
#define _INTEGRATORS_HPP

#include "mfem.hpp"

namespace mfem
{

    class ReferenceMassIntegrator : public BilinearFormIntegrator
    {
    protected:
#ifndef MFEM_THREAD_SAFE
        Vector shape, te_shape;
#endif
        Coefficient *Q;
        // PA extension
        const FiniteElementSpace *fespace;
        Vector pa_data;
        const DofToQuad *maps;                 ///< Not owned
        const GeometricFactors *geom;          ///< Not owned
        const FaceGeometricFactors *face_geom; ///< Not owned
        int dim, ne, nq, dofs1D, quad1D;

    public:
    public:
        ReferenceMassIntegrator(const IntegrationRule *ir = nullptr);

        /// Construct a mass integrator with coefficient q
        ReferenceMassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL);

        /** Given a particular Finite Element computes the element mass matrix
            elmat. */
        void AssembleElementMatrix(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   DenseMatrix &elmat) override;
        void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                    const FiniteElement &test_fe,
                                    ElementTransformation &Trans,
                                    DenseMatrix &elmat) override;

        static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              const ElementTransformation &Trans);

        const Coefficient *GetCoefficient() const { return Q; }

    protected:
        const IntegrationRule *GetDefaultIntegrationRule(
            const FiniteElement &trial_fe,
            const FiniteElement &test_fe,
            const ElementTransformation &trans) const override
        {
            return &GetRule(trial_fe, test_fe, trans);
        }
    };

    /// Class for domain integration $ L(v) := (f, v) $
    class ReferenceLFIntegrator : public LinearFormIntegrator
    {
        Vector shape;
        Coefficient &Q;
        int oa, ob;

    public:
        /// Constructs a domain integrator with a given Coefficient
        ReferenceLFIntegrator(Coefficient &QF, int a = 2, int b = 0)
            // the old default was a = 1, b = 1
            // for simple elliptic problems a = 2, b = -2 is OK
            : Q(QF), oa(a), ob(b)
        {
        }

        /// Constructs a domain integrator with a given Coefficient
        ReferenceLFIntegrator(Coefficient &QF, const IntegrationRule *ir)
            : Q(QF), oa(1), ob(1) {}

        /** Given a particular Finite Element and a transformation (Tr)
            computes the element right hand side element vector, elvect. */
        void AssembleRHSElementVect(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    Vector &elvect) override;

        using LinearFormIntegrator::AssembleRHSElementVect;
    };

}

#endif