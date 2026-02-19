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
        constexpr label_t r = lbm::velocitySet::max_abs_c();

        return (x < r || y < r || z < r ||
                x >= mesh::nx - r ||
                y >= mesh::ny - r ||
                z >= mesh::nz - r);
    }

    template <label_t N>
    __device__ [[nodiscard]] static inline label_t periodic_wrap(label_t x) noexcept
    {
        if (x < 1)
        {
            return N - 2;
        }
        if (x > N - 2)
        {
            return 1;
        }

        return x;
    }

    __device__ [[nodiscard]] static inline label_t idx(
        const label_t tx,
        const label_t ty,
        const label_t tz,
        const label_t bx,
        const label_t by,
        const label_t bz) noexcept
    {
        return tx + block::nx() * (ty + block::ny() * (tz + block::nz() * (bx + block::num_x() * (by + block::num_y() * bz))));
    }

    __device__ static inline void prefetch_L2(const void *ptr) noexcept
    {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
    }

    __device__ [[nodiscard]] static inline scalar_t sponge_ramp(const label_t z) noexcept
    {
        const scalar_t zn = static_cast<scalar_t>(z) * sponge::inv_nz_m1();
        scalar_t s = (zn - sponge::z_start()) * sponge::inv_sponge();
        s = math::min(math::max(s, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
        return s * s * (static_cast<scalar_t>(3) - static_cast<scalar_t>(2) * s); // cubic smoothstep
        // return s * s * s * (static_cast<scalar_t>(10) + s * (-static_cast<scalar_t>(15) + 6 * s)); // quintic smoothstep
    }
}

#endif