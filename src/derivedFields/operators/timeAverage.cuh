/*---------------------------------------------------------------------------*\
|                                                                             |
| phaseFieldLBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/phaseFieldLBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

Description
   Time averaging of primary flow and phase fields

Namespace
    lbm
    derived
    average

SourceFiles
    timeAverage.cuh

\*---------------------------------------------------------------------------*/

#ifndef TIMEAVERAGE_CUH
#define TIMEAVERAGE_CUH

#include "fileIO/fields.cuh"

#if TIME_AVERAGE

namespace lbm
{
    __global__ void timeAverage(
        LBMFields d,
        const label_t t)
    {
        const label_t x = threadIdx.x + block::nx() * blockIdx.x;
        const label_t y = threadIdx.y + block::ny() * blockIdx.y;
        const label_t z = threadIdx.z + block::nz() * blockIdx.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        if (d.avg_phi == nullptr && d.avg_ux == nullptr && d.avg_uy == nullptr && d.avg_uz == nullptr)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        auto update = [t] __device__(scalar_t old_val, scalar_t new_val)
        {
            return old_val + (new_val - old_val) / static_cast<scalar_t>(t);
        };

        if (d.avg_phi)
        {
            const scalar_t phi = d.phi[idx3];
            d.avg_phi[idx3] = update(d.avg_phi[idx3], phi);
        }
        if (d.avg_ux)
        {
            const scalar_t ux = d.ux[idx3];
            d.avg_ux[idx3] = update(d.avg_ux[idx3], ux);
        }
        if (d.avg_uy)
        {
            const scalar_t uy = d.uy[idx3];
            d.avg_uy[idx3] = update(d.avg_uy[idx3], uy);
        }
        if (d.avg_uz)
        {
            const scalar_t uz = d.uz[idx3];
            d.avg_uz[idx3] = update(d.avg_uz[idx3], uz);
        }
    }
}

namespace derived
{
    namespace average
    {
        static constexpr auto fields = std::to_array<host::FieldConfig>({
            {host::FieldID::Avg_phi, "avg_phi", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_ux, "avg_ux", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uy, "avg_uy", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uz, "avg_uz", host::FieldDumpShape::Grid3D, true},
        });

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d,
            const label_t t) noexcept
        {
#if TIME_AVERAGE
            lbm::timeAverage<<<grid, block, dynamic, queue>>>(d, t + 1);
#endif
        }

        __host__ static inline void free(LBMFields &d) noexcept
        {
#if TIME_AVERAGE
            if (d.avg_phi)
            {
                cudaFree(d.avg_phi);
                d.avg_phi = nullptr;
            }
            if (d.avg_ux)
            {
                cudaFree(d.avg_ux);
                d.avg_ux = nullptr;
            }
            if (d.avg_uy)
            {
                cudaFree(d.avg_uy);
                d.avg_uy = nullptr;
            }
            if (d.avg_uz)
            {
                cudaFree(d.avg_uz);
                d.avg_uz = nullptr;
            }
#endif
        }
    }
}

#endif

#endif
