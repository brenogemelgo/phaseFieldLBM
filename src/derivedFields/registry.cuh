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

License
    This file is part of MULTIC-TS-LBM.

    MULTIC-TS-LBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Derived fields registry

Namespace
    Derived

SourceFiles
    registry.cuh

\*---------------------------------------------------------------------------*/

#ifndef REGISTRY_CUH
#define REGISTRY_CUH

#include "timeAverage.cuh"
#include "reynoldsMoments.cuh"
#include "vorticityFields.cuh"
#include "passiveScalar.cuh"

namespace Derived
{

    __host__ [[nodiscard]] static inline std::vector<host::FieldConfig> makeOutputFields()
    {
        std::vector<host::FieldConfig> fields;
        fields.reserve(4 + 6 + 4); // 4 time averages + 6 reynolds moments + 4 vorticity fields + 1 concentration field
#if TIME_AVERAGE
        fields.insert(fields.end(), TimeAvg::fields.begin(), TimeAvg::fields.end());
#endif
#if REYNOLDS_MOMENTS
        fields.insert(fields.end(), Reynolds::fields.begin(), Reynolds::fields.end());
#endif
#if VORTICITY_FIELDS
        fields.insert(fields.end(), Vorticity::fields.begin(), Vorticity::fields.end());
#endif
#if PASSIVE_SCALAR
        fields.insert(fields.end(), PassiveScalar::fields.begin(), PassiveScalar::fields.end());
#endif
        return fields;
    }

    template <dim3 grid, dim3 block, size_t dynamic>
    __host__ static inline void launchAllDerived(
        cudaStream_t queue,
        LBMFields d,
        const label_t step) noexcept
    {
#if TIME_AVERAGE
        TimeAvg::launch<grid, block, dynamic>(queue, d, step);
#endif
#if REYNOLDS_MOMENTS
        Reynolds::launch<grid, block, dynamic>(queue, d, step);
#endif
#if VORTICITY_FIELDS
        Vorticity::launch<grid, block, dynamic>(queue, d);
#endif
#if PASSIVE_SCALAR
        PassiveScalar::launch<grid, block, dynamic>(queue, d);
#endif
    }

    __host__ static inline void freeAll(LBMFields &d) noexcept
    {
#if TIME_AVERAGE
        TimeAvg::free(d);
#endif
#if REYNOLDS_MOMENTS
        Reynolds::free(d);
#endif
#if VORTICITY_FIELDS
        Vorticity::free(d);
#endif
#if PASSIVE_SCALAR
        PassiveScalar::free(d);
#endif
    }
}

#endif
