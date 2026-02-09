/*---------------------------------------------------------------------------*\
|                                                                             |
| MULTIC-TS-LBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/MULTIC-TS-LBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

Description
    Derived fields registry

Namespace
    Derived

SourceFiles
    registry.cuh

\*---------------------------------------------------------------------------*/

#ifndef REGISTRY_CUH
#define REGISTRY_CUH

#include "selection.cuh"

#include "operators/timeAverage.cuh"
#include "operators/reynoldsMoments.cuh"
#include "operators/vorticityFields.cuh"
#include "operators/passiveScalar.cuh"

namespace Derived
{
    __host__ [[nodiscard]] static inline std::vector<host::FieldConfig> makeOutputFields()
    {
        std::vector<host::FieldConfig> out;

        size_t cap = 0;

#if TIME_AVERAGE
        cap += TimeAverage::fields.size();
#endif
#if REYNOLDS_MOMENTS
        cap += ReynoldsMoments::fields.size();
#endif
#if VORTICITY_FIELDS
        cap += VorticityFields::fields.size();
#endif
#if PASSIVE_SCALAR
        cap += PassiveScalar::fields.size();
#endif

        out.reserve(cap);

        auto append_selected = [&](const auto &arr)
        {
            for (const auto &cfg : arr)
            {
                if (Selection::enabledName(cfg.name))
                {
                    out.push_back(cfg);
                }
            }
        };

#if TIME_AVERAGE
        append_selected(TimeAverage::fields);
#endif
#if REYNOLDS_MOMENTS
        append_selected(ReynoldsMoments::fields);
#endif
#if VORTICITY_FIELDS
        append_selected(VorticityFields::fields);
#endif
#if PASSIVE_SCALAR
        append_selected(PassiveScalar::fields);
#endif

        return out;
    }

    template <dim3 grid, dim3 block, size_t dynamic>
    __host__ static inline void launchAllDerived(
        cudaStream_t queue,
        LBMFields d,
        const label_t step) noexcept
    {
#if TIME_AVERAGE
        if (Selection::anyEnabled(TimeAverage::fields))
        {
            TimeAverage::launch<grid, block, dynamic>(queue, d, step);
        }
#endif

#if REYNOLDS_MOMENTS
        if (Selection::anyEnabled(ReynoldsMoments::fields))
        {
            ReynoldsMoments::launch<grid, block, dynamic>(queue, d, step);
        }
#endif

#if VORTICITY_FIELDS
        if (Selection::anyEnabled(VorticityFields::fields))
        {
            VorticityFields::launch<grid, block, dynamic>(queue, d);
        }
#endif

#if PASSIVE_SCALAR
        if (Selection::anyEnabled(PassiveScalar::fields))
        {
            PassiveScalar::launch<grid, block, dynamic>(queue, d);
        }
#endif
    }

    __host__ static inline void freeAll(LBMFields &d) noexcept
    {
#if TIME_AVERAGE
        TimeAverage::free(d);
#endif
#if REYNOLDS_MOMENTS
        ReynoldsMoments::free(d);
#endif
#if VORTICITY_FIELDS
        VorticityFields::free(d);
#endif
#if PASSIVE_SCALAR
        PassiveScalar::free(d);
#endif
    }
}

#endif
