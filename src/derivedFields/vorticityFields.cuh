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
    Vorticity derived-field kernels (components and magnitude)

Namespace
    LBM

SourceFiles
    vorticityFields.cuh

Notes
    - Assumes lattice spacing = 1 (LB units) for finite differences.
    - Boundary stencils are one-sided to avoid out-of-bounds accesses.

\*---------------------------------------------------------------------------*/

#ifndef VORTICITYFIELDS_CUH
#define VORTICITYFIELDS_CUH

#include "functions/ioFields.cuh"

#if VORTICITY_FIELDS

namespace LBM
{
    __global__ void vorticityCompute(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        // Default: zero vorticity (boundaries)
        scalar_t wx = static_cast<scalar_t>(0);
        scalar_t wy = static_cast<scalar_t>(0);
        scalar_t wz = static_cast<scalar_t>(0);

        // Interior only
        if (x > 0 && x < mesh::nx - 1 &&
            y > 0 && y < mesh::ny - 1 &&
            z > 0 && z < mesh::nz - 1)
        {
            const label_t idx_xp = device::global3(x + 1, y, z);
            const label_t idx_xm = device::global3(x - 1, y, z);
            const label_t idx_yp = device::global3(x, y + 1, z);
            const label_t idx_ym = device::global3(x, y - 1, z);
            const label_t idx_zp = device::global3(x, y, z + 1);
            const label_t idx_zm = device::global3(x, y, z - 1);

            const scalar_t half = static_cast<scalar_t>(0.5);

            const scalar_t duz_dy = half * (d.uz[idx_yp] - d.uz[idx_ym]);
            const scalar_t duy_dz = half * (d.uy[idx_zp] - d.uy[idx_zm]);

            const scalar_t dux_dz = half * (d.ux[idx_zp] - d.ux[idx_zm]);
            const scalar_t duz_dx = half * (d.uz[idx_xp] - d.uz[idx_xm]);

            const scalar_t duy_dx = half * (d.uy[idx_xp] - d.uy[idx_xm]);
            const scalar_t dux_dy = half * (d.ux[idx_yp] - d.ux[idx_ym]);

            wx = duz_dy - duy_dz;
            wy = dux_dz - duz_dx;
            wz = duy_dx - dux_dy;
        }

        const scalar_t wmag =
            math::sqrt(wx * wx + wy * wy + wz * wz);

        d.vort_x[idx3] = wx;
        d.vort_y[idx3] = wy;
        d.vort_z[idx3] = wz;
        d.vort_mag[idx3] = wmag;
    }
}

namespace Derived
{
    namespace Vorticity
    {
        constexpr std::array<host::FieldConfig, 4> fields{{
            {host::FieldID::Vort_x, "vort_x", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Vort_y, "vort_y", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Vort_z, "vort_z", host::FieldDumpShape::Grid3D, true},
            {host::FieldID::Vort_mag, "vort_mag", host::FieldDumpShape::Grid3D, true},
        }};

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d) noexcept
        {
#if VORTICITY_FIELDS
            LBM::vorticityCompute<<<grid, block, dynamic, queue>>>(d);
#endif
        }

        __host__ static inline void free(LBMFields &d)
        {
#if VORTICITY_FIELDS
            cudaFree(d.vort_x);
            cudaFree(d.vort_y);
            cudaFree(d.vort_z);
            cudaFree(d.vort_mag);
#endif
        }
    }
}

#endif

#endif
