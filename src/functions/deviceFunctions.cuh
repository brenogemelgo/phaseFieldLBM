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
        const scalar_t s = math::min(math::max((zn - sponge::z_start()) * sponge::inv_sponge(), static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
        return s * s * s;
    }
}

namespace lbm
{
    namespace esopull
    {
        // Load physical populations into pop[] for node (x,y,z) at time t
        template <class VS>
        __device__ inline void load_f(
            const LBMFields &d,
            const label_t x, const label_t y, const label_t z,
            scalar_t *__restrict__ pop,
            const label_t t) noexcept
        {
            const label_t n = device::global3(x, y, z);

            pop[0] = from_pop(d.f[0 * size::cells() + n]); // Listing A5 line 2

            const bool odd = (t & 1);

            // i = 1,3,5,... are the "base" indices in consecutive opposite pairs (i,i+1)
            device::constexpr_for<0, (VS::Q() - 1) / 2>(
                [&](const auto K)
                {
                    constexpr label_t k = K.value;
                    constexpr label_t i = static_cast<label_t>(2 * k + 1);

                    const label_t xx = x + static_cast<label_t>(VS::template cx<i>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<i>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<i>());

                    const label_t j = device::global3(xx, yy, zz); // j[i] in Listing A5

                    // Listing A5:
                    // fhn[i]   = load(n, odd ? i : i+1)
                    // fhn[i+1] = load(j[i], odd ? i+1 : i)
                    pop[i] = from_pop(d.f[(odd ? i : (i + 1)) * size::cells() + n]);
                    pop[i + 1] = from_pop(d.f[(odd ? (i + 1) : i) * size::cells() + j]);
                });
        }

        // Store physical post-collision populations post[] for node (x,y,z) for step t
        // This writes exactly the addresses that load_f() reads at time t (race-free single-buffer).
        template <class VS>
        __device__ inline void store_f(
            LBMFields d,
            const label_t x, const label_t y, const label_t z,
            const scalar_t *__restrict__ post,
            const label_t t) noexcept
        {
            const label_t n = device::global3(x, y, z);

            d.f[0 * size::cells() + n] = to_pop(post[0]); // Listing A5 line 9

            const bool odd = (t & 1);

            device::constexpr_for<0, (VS::Q() - 1) / 2>(
                [&](const auto K)
                {
                    constexpr label_t k = K.value;
                    constexpr label_t i = static_cast<label_t>(2 * k + 1);

                    const label_t xx = x + static_cast<label_t>(VS::template cx<i>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<i>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<i>());

                    const label_t j = device::global3(xx, yy, zz);

                    // Listing A5:
                    // store(j[i], odd ? i+1 : i, fhn[i])
                    // store(n,    odd ? i   : i+1, fhn[i+1])
                    d.f[(odd ? (i + 1) : i) * size::cells() + j] = to_pop(post[i]);
                    d.f[(odd ? i : (i + 1)) * size::cells() + n] = to_pop(post[i + 1]);
                });
        }

        // ---- BC helpers: write/read a single physical direction Q at node (x,y,z) for time t_phys ----
        // This is needed because in Esoteric Pull, some physical directions live in neighbor storage.
        template <class VS, label_t Q>
        __device__ inline scalar_t load_phys(
            const LBMFields &d,
            const label_t x, const label_t y, const label_t z,
            const label_t t_phys) noexcept
        {
            if constexpr (Q == 0)
            {
                const label_t n = device::global3(x, y, z);
                return from_pop(d.f[0 * size::cells() + n]);
            }
            else
            {
                constexpr label_t base = (Q & 1) ? Q : (Q - 1); // odd index i
                constexpr label_t opp = base + 1;               // even index i+1
                const bool odd = (t_phys & 1);

                if constexpr (Q & 1)
                {
                    // physical odd direction stored at center: dir = odd ? Q : Q+1
                    const label_t n = device::global3(x, y, z);
                    const label_t dir = odd ? Q : (Q + 1);
                    return from_pop(d.f[dir * size::cells() + n]);
                }
                else
                {
                    // physical even direction stored at neighbor j[base] with dir = odd ? (base+1) : base
                    const label_t xx = x + static_cast<label_t>(VS::template cx<base>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<base>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<base>());
                    const label_t j = device::global3(xx, yy, zz);

                    const label_t dir = odd ? opp : base;
                    return from_pop(d.f[dir * size::cells() + j]);
                }
            }
        }

        template <class VS, label_t Q>
        __device__ inline void store_phys(
            LBMFields d,
            const label_t x, const label_t y, const label_t z,
            const scalar_t val,
            const label_t t_phys) noexcept
        {
            if constexpr (Q == 0)
            {
                const label_t n = device::global3(x, y, z);
                d.f[0 * size::cells() + n] = to_pop(val);
            }
            else
            {
                constexpr label_t base = (Q & 1) ? Q : (Q - 1);
                constexpr label_t opp = base + 1;
                const bool odd = (t_phys & 1);

                if constexpr (Q & 1)
                {
                    const label_t n = device::global3(x, y, z);
                    const label_t dir = odd ? Q : (Q + 1);
                    d.f[dir * size::cells() + n] = to_pop(val);
                }
                else
                {
                    const label_t xx = x + static_cast<label_t>(VS::template cx<base>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<base>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<base>());
                    const label_t j = device::global3(xx, yy, zz);

                    const label_t dir = odd ? opp : base;
                    d.f[dir * size::cells() + j] = to_pop(val);
                }
            }
        }
    }
}

namespace phase
{
    namespace esopull
    {
        // g is stored as scalar_t (no to_pop/from_pop)
        template <class VS>
        __device__ inline void load_g(
            const LBMFields &d,
            const label_t x, const label_t y, const label_t z,
            scalar_t *__restrict__ pop,
            const label_t t) noexcept
        {
            const label_t n = device::global3(x, y, z);

            pop[0] = d.g[0 * size::cells() + n];

            const bool odd = (t & 1);

            device::constexpr_for<0, (VS::Q() - 1) / 2>(
                [&](auto K)
                {
                    constexpr label_t k = K.value;
                    constexpr label_t i = static_cast<label_t>(2 * k + 1);

                    const label_t xx = x + static_cast<label_t>(VS::template cx<i>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<i>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<i>());

                    const label_t j = device::global3(xx, yy, zz);

                    pop[i] = d.g[(odd ? i : (i + 1)) * size::cells() + n];
                    pop[i + 1] = d.g[(odd ? (i + 1) : i) * size::cells() + j];
                });
        }

        template <class VS>
        __device__ inline void store_g(
            LBMFields d,
            const label_t x, const label_t y, const label_t z,
            const scalar_t *__restrict__ post,
            const label_t t) noexcept
        {
            const label_t n = device::global3(x, y, z);

            d.g[0 * size::cells() + n] = post[0];

            const bool odd = (t & 1);

            device::constexpr_for<0, (VS::Q() - 1) / 2>(
                [&](auto K)
                {
                    constexpr label_t k = K.value;
                    constexpr label_t i = static_cast<label_t>(2 * k + 1);

                    const label_t xx = x + static_cast<label_t>(VS::template cx<i>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<i>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<i>());

                    const label_t j = device::global3(xx, yy, zz);

                    d.g[(odd ? (i + 1) : i) * size::cells() + j] = post[i];
                    d.g[(odd ? i : (i + 1)) * size::cells() + n] = post[i + 1];
                });
        }

        // Optional (needed for BCs that set individual directions like Q=5/Q=6)
        template <class VS, label_t Q>
        __device__ inline scalar_t load_phys(
            const LBMFields &d,
            const label_t x, const label_t y, const label_t z,
            const label_t t_phys) noexcept
        {
            if constexpr (Q == 0)
            {
                const label_t n = device::global3(x, y, z);
                return d.g[0 * size::cells() + n];
            }
            else
            {
                constexpr label_t base = (Q & 1) ? Q : (Q - 1);
                constexpr label_t opp = base + 1;
                const bool odd = (t_phys & 1);

                if constexpr (Q & 1)
                {
                    const label_t n = device::global3(x, y, z);
                    const label_t dir = odd ? Q : (Q + 1);
                    return d.g[dir * size::cells() + n];
                }
                else
                {
                    const label_t xx = x + static_cast<label_t>(VS::template cx<base>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<base>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<base>());
                    const label_t j = device::global3(xx, yy, zz);

                    const label_t dir = odd ? opp : base;
                    return d.g[dir * size::cells() + j];
                }
            }
        }

        template <class VS, label_t Q>
        __device__ inline void store_phys(
            LBMFields d,
            const label_t x, const label_t y, const label_t z,
            const scalar_t val,
            const label_t t_phys) noexcept
        {
            if constexpr (Q == 0)
            {
                const label_t n = device::global3(x, y, z);
                d.g[0 * size::cells() + n] = val;
            }
            else
            {
                constexpr label_t base = (Q & 1) ? Q : (Q - 1);
                constexpr label_t opp = base + 1;
                const bool odd = (t_phys & 1);

                if constexpr (Q & 1)
                {
                    const label_t n = device::global3(x, y, z);
                    const label_t dir = odd ? Q : (Q + 1);
                    d.g[dir * size::cells() + n] = val;
                }
                else
                {
                    const label_t xx = x + static_cast<label_t>(VS::template cx<base>());
                    const label_t yy = y + static_cast<label_t>(VS::template cy<base>());
                    const label_t zz = z + static_cast<label_t>(VS::template cz<base>());
                    const label_t j = device::global3(xx, yy, zz);

                    const label_t dir = odd ? opp : base;
                    d.g[dir * size::cells() + j] = val;
                }
            }
        }
    }
}

#endif