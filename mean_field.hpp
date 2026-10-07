#ifndef MFEM_MEAN_FIELD_REMAP_HPP
#define MFEM_MEAN_FIELD_REMAP_HPP
#include "mfem.hpp"

namespace mfem
{
// Convert the transported uniform-field flux to a periodic potential correction
// on the current geometry. reference_nodes is the geometry at the last rezone.
// No target field or analytic time-dependent magnetic solution is used.
void AddUniformMeanFieldPotential(ParGridFunction &potential,
                                 const ParGridFunction &reference_nodes,
                                 const ParGridFunction &current_nodes,
                                 const Vector &uniform_field);
}
#endif
