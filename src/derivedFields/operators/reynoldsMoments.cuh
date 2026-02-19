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
    Computation of Reynolds stress moments via incremental time averaging

Namespace
    lbm
    derived
    reynolds

SourceFiles
    reynoldsMoments.cuh

\*---------------------------------------------------------------------------*/

#ifndef REYNOLDSMOMENTS_CUH
#define REYNOLDSMOMENTS_CUH

#include "fileIO/fields.cuh"

#if REYNOLDS_MOMENTS

namespace lbm
{
    __global__ void reynoldsMomentsAverage(
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

        if (d.avg_uxux == nullptr && d.avg_uyuy == nullptr && d.avg_uzuz == nullptr && d.avg_uxuy == nullptr && d.avg_uxuz == nullptr && d.avg_uyuz == nullptr)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t ux = d.ux[idx3];
        const scalar_t uy = d.uy[idx3];
        const scalar_t uz = d.uz[idx3];

        auto update = [t] __device__(scalar_t oldv, scalar_t instv)
        {
            return oldv + (instv - oldv) / static_cast<scalar_t>(t);
        };

        if (d.avg_uxux)
        {
            const scalar_t uxux = ux * ux;
            d.avg_uxux[idx3] = update(d.avg_uxux[idx3], uxux);
        }
        if (d.avg_uyuy)
        {
            const scalar_t uyuy = uy * uy;
            d.avg_uyuy[idx3] = update(d.avg_uyuy[idx3], uyuy);
        }
        if (d.avg_uzuz)
        {
            const scalar_t uzuz = uz * uz;
            d.avg_uzuz[idx3] = update(d.avg_uzuz[idx3], uzuz);
        }
        if (d.avg_uxuy)
        {
            const scalar_t uxuy = ux * uy;
            d.avg_uxuy[idx3] = update(d.avg_uxuy[idx3], uxuy);
        }
        if (d.avg_uxuz)
        {
            const scalar_t uxuz = ux * uz;
            d.avg_uxuz[idx3] = update(d.avg_uxuz[idx3], uxuz);
        }
        if (d.avg_uyuz)
        {
            const scalar_t uyuz = uy * uz;
            d.avg_uyuz[idx3] = update(d.avg_uyuz[idx3], uyuz);
        }
    }
}

namespace derived
{
    namespace reynolds
    {
        static constexpr auto fields = std::to_array<host::FieldConfig>({
            {host::FieldID::Avg_uxux, "avg_uxux", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uyuy, "avg_uyuy", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uzuz, "avg_uzuz", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uxuy, "avg_uxuy", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uxuz, "avg_uxuz", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Avg_uyuz, "avg_uyuz", host::FieldDumpShape::Grid3D, true},
        });

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d,
            const label_t t) noexcept
        {
            lbm::reynoldsMomentsAverage<<<grid, block, dynamic, queue>>>(d, t + 1);
        }

        __host__ static inline void free(LBMFields &d) noexcept
        {
            if (d.avg_uxux)
            {
                cudaFree(d.avg_uxux);
                d.avg_uxux = nullptr;
            }
            if (d.avg_uyuy)
            {
                cudaFree(d.avg_uyuy);
                d.avg_uyuy = nullptr;
            }
            if (d.avg_uzuz)
            {
                cudaFree(d.avg_uzuz);
                d.avg_uzuz = nullptr;
            }
            if (d.avg_uxuy)
            {
                cudaFree(d.avg_uxuy);
                d.avg_uxuy = nullptr;
            }
            if (d.avg_uxuz)
            {
                cudaFree(d.avg_uxuz);
                d.avg_uxuz = nullptr;
            }
            if (d.avg_uyuz)
            {
                cudaFree(d.avg_uyuz);
                d.avg_uyuz = nullptr;
            }
        }
    }
}

#endif

#endif
