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
    Device-side indexing, wrapping, and utility functions for kernel execution

Namespace
    device

SourceFiles
    deviceFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef DEVICEFUNCTIONS_CUH
#define DEVICEFUNCTIONS_CUH

#include "globalFunctions.cuh"

namespace device
{
    __device__ [[nodiscard]] static inline label_t global3(
        const label_t x,
        const label_t y,
        const label_t z) noexcept
    {
        return x + y * mesh::nx + z * size::stride();
    }

    __device__ [[nodiscard]] static inline label_t global4(
        const label_t x,
        const label_t y,
        const label_t z,
        const label_t Q) noexcept
    {
        return Q * size::cells() + global3(x, y, z);
    }

    __device__ [[nodiscard]] static inline bool guard(
        const label_t x,
        const label_t y,
        const label_t z) noexcept
    {
        return (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
                x == 0 || x == mesh::nx - 1 ||
                y == 0 || y == mesh::ny - 1 ||
                z == 0 || z == mesh::nz - 1);
    }

    __device__ [[nodiscard]] static inline label_t wrapX(const label_t xx) noexcept
    {
        if (xx == 0)
        {
            return mesh::nx - 2;
        }
        if (xx == mesh::nx - 1)
        {
            return 1;
        }

        return xx;
    }

    __device__ [[nodiscard]] static inline label_t wrapY(const label_t yy) noexcept
    {
        if (yy == 0)
        {
            return mesh::ny - 2;
        }
        if (yy == mesh::ny - 1)
        {
            return 1;
        }

        return yy;
    }

    __device__ [[nodiscard]] static inline label_t wrapZ(const label_t zz) noexcept
    {
        if (zz == 0)
        {
            return mesh::nz - 2;
        }
        if (zz == mesh::nz - 1)
        {
            return 1;
        }

        return zz;
    }

    __device__ [[nodiscard]] static inline label_t globalThreadIdx(
        const label_t tx,
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz) noexcept
    {
        return tx + block::nx * (ty + block::ny * (tz + block::nz * (bx + block::num_block_x() * (by + block::num_block_y() * bz))));
    }

    __device__ static inline void prefetch_L2(const void *ptr) noexcept
    {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
    }

    // __device__ [[nodiscard]] static inline scalar_t visc_sponge(const label_t z) noexcept
    // {
    //     if constexpr (LBM::FlowCase::jet_case())
    //     {
    //         const scalar_t zn = static_cast<scalar_t>(z) * sponge::inv_nz_m1();
    //         const scalar_t s = math::min(math::max((zn - sponge::z_start()) * sponge::inv_sponge(), static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
    //         const scalar_t ramp = s * s * s;

    //         const scalar_t nu_s = relaxation::visc_ref() * math::fma(sponge::K_gain(), ramp, static_cast<scalar_t>(1));

    //         return relaxation::omega_from_nu(nu_s);
    //     }
    //     else if constexpr (LBM::FlowCase::droplet_case())
    //     {
    //         return 0;
    //     }
    // }

    __device__ [[nodiscard]] static inline scalar_t sponge_ramp(const label_t z) noexcept
    {
        const scalar_t zn = static_cast<scalar_t>(z) * sponge::inv_nz_m1();
        const scalar_t s = math::min(math::max((zn - sponge::z_start()) * sponge::inv_sponge(), static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
        return s * s * s;
    }
}

#endif