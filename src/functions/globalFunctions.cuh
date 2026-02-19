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
    Global compile-time geometry, math, relaxation, and utility functions shared across the solver

Namespace
    block
    physics
    geometry
    relaxation
    LBM
    math
    size
    sponge (only if JET is defined)

SourceFiles
    globalFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef GLOBALFUNCTIONS_CUH
#define GLOBALFUNCTIONS_CUH

#include "constants.cuh"

namespace block
{
    __device__ __host__ [[nodiscard]] static inline consteval label_t nx() noexcept
    {
        return 32;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t ny() noexcept
    {
        return 4;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t nz() noexcept
    {
        return 4;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t num_x() noexcept
    {
        return (mesh::nx + block::nx() - 1) / block::nx();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t num_y() noexcept
    {
        return (mesh::ny + block::ny() - 1) / block::ny();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t size() noexcept
    {
        return block::nx() * block::ny() * block::nz();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t pad() noexcept
    {
        return 1;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t tile_nx() noexcept
    {
        return block::nx() + 2 * block::pad();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t tile_ny() noexcept
    {
        return block::ny() + 2 * block::pad();
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t tile_nz() noexcept
    {
        return block::nz() + 2 * block::pad();
    }
}

namespace size
{
    __device__ __host__ [[nodiscard]] static inline consteval label_t stride() noexcept
    {
        return mesh::nx * mesh::ny;
    }

    __device__ __host__ [[nodiscard]] static inline consteval label_t cells() noexcept
    {
        return mesh::nx * mesh::ny * mesh::nz;
    }
}

namespace geometry
{
    __device__ __host__ [[nodiscard]] static inline consteval scalar_t R2() noexcept
    {
        return static_cast<scalar_t>(mesh::radius * mesh::radius);
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t center_x() noexcept
    {
        return static_cast<scalar_t>(mesh::nx - 1) * static_cast<scalar_t>(0.5);
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t center_y() noexcept
    {
        return static_cast<scalar_t>(mesh::ny - 1) * static_cast<scalar_t>(0.5);
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t center_z() noexcept
    {
        return static_cast<scalar_t>(mesh::nz - 1) * static_cast<scalar_t>(0.5);
    }
}

namespace math
{
    __device__ __host__ [[nodiscard]] static inline consteval scalar_t two_pi() noexcept
    {
        return static_cast<scalar_t>(2) * static_cast<scalar_t>(CUDART_PI_F);
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t sqrt(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::sqrtf(x);
        }
        else
        {
            return ::sqrt(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t log(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::logf(x);
        }
        else
        {
            return ::log(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t min(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fminf(a, b);
        }
        else
        {
            return ::fmin(a, b);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t max(
        const scalar_t a,
        const scalar_t b) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaxf(a, b);
        }
        else
        {
            return ::fmax(a, b);
        }
    }

    template <scalar_t Lo, scalar_t Hi>
    __device__ __host__ [[nodiscard]] static inline scalar_t clamp(const scalar_t x) noexcept
    {
        static_assert(Lo <= Hi, "Invalid clamp bounds");
        return max(Lo, min(x, Hi));
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t cos(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::cosf(x);
        }
        else
        {
            return ::cos(x);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t tanh(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::tanhf(x);
        }
        else
        {
            return ::tanh(x);
        }
    }

    __device__ __host__ static inline void sincos(
        const scalar_t x,
        scalar_t *s,
        scalar_t *c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            ::sincosf(x, s, c);
        }
        else
        {
            ::sincos(x, s, c);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t fma(
        const scalar_t a,
        const scalar_t b,
        const scalar_t c) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fmaf(a, b, c);
        }
        else
        {
            return ::fma(a, b, c);
        }
    }

    __device__ __host__ [[nodiscard]] static inline scalar_t abs(const scalar_t x) noexcept
    {
        if constexpr (std::is_same_v<scalar_t, float>)
        {
            return ::fabsf(x);
        }
        else
        {
            return ::fabs(x);
        }
    }
}

namespace sponge
{
    __device__ __host__ [[nodiscard]] static inline consteval scalar_t K_gain() noexcept
    {
        return static_cast<scalar_t>(100);
    }

    __device__ __host__ [[nodiscard]] static inline consteval int sponge_cells() noexcept
    {
        return static_cast<int>(mesh::nz / 12);
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t sponge() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(sponge_cells()) / static_cast<double>(mesh::nz - 1));
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t z_start() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(mesh::nz - 1 - sponge_cells()) / static_cast<double>(mesh::nz - 1));
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t inv_nz_m1() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(mesh::nz - 1));
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t inv_sponge() noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(sponge()));
    }
}

namespace relaxation
{

#if defined(JET)

    // __device__ __host__ [[nodiscard]] static inline consteval scalar_t visc_water() noexcept
    // {
    //     return static_cast<scalar_t>(1.71e-4);
    // }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t visc_water() noexcept
    {
        return static_cast<scalar_t>((static_cast<double>(physics::u_inf) * static_cast<double>(mesh::diam)) / static_cast<double>(physics::reynolds_water));
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t visc_oil() noexcept
    {
        return static_cast<scalar_t>((static_cast<double>(physics::u_inf) * static_cast<double>(mesh::diam)) / static_cast<double>(physics::reynolds_oil));
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t visc_ref() noexcept
    {
        return visc_water();
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omega_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / (static_cast<double>(0.5) + static_cast<double>(lbm::velocitySet::as2()) * static_cast<double>(nu)));
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t tau_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(0.5) + static_cast<scalar_t>(lbm::velocitySet::as2()) * nu;
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_water() noexcept
    {
        return omega_from_nu(visc_water());
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_oil() noexcept
    {
        return omega_from_nu(visc_oil());
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_ref() noexcept
    {
        return omega_water();
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t tau_water() noexcept
    {
        return tau_from_nu(visc_water());
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t tau_oil() noexcept
    {
        return tau_from_nu(visc_oil());
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t tau_ref() noexcept
    {
        return tau_water();
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_zmin() noexcept
    {
        return omega_oil();
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omega_zmax(const scalar_t phi) noexcept
    {
        const scalar_t visc_local = (static_cast<scalar_t>(1) - phi) * visc_water() + phi * visc_oil();

        return omega_from_nu(visc_local * (static_cast<scalar_t>(1) + sponge::K_gain()));
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omega_delta(const scalar_t phi) noexcept
    {
        return omega_zmax(phi) - omega_zmin();
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t tau_zmax(const scalar_t phi) noexcept
    {
        const scalar_t visc_local = (static_cast<scalar_t>(1) - phi) * visc_water() + phi * visc_oil();
        const scalar_t visc_sp = visc_local * (static_cast<scalar_t>(1) + sponge::K_gain());

        return tau_from_nu(visc_sp);
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omco_ref() noexcept
    {
        return static_cast<scalar_t>(1) - omega_ref();
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omco_zmin() noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmin();
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omco_zmax(const scalar_t phi) noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmax(phi);
    }

#elif defined(DROPLET)

    // __device__ __host__ [[nodiscard]] static inline consteval scalar_t visc_water() noexcept
    // {
    //     return static_cast<scalar_t>(1.71e-4);
    // }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t tau_water() noexcept
    {
        return static_cast<scalar_t>(0);
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t tau_oil() noexcept
    {
        return static_cast<scalar_t>(0);
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t tau_zmax(const scalar_t phi) noexcept
    {
        return static_cast<scalar_t>(0);
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omega_from_nu(const scalar_t nu) noexcept
    {
        return static_cast<scalar_t>(static_cast<double>(1) / (static_cast<double>(0.5) + static_cast<double>(lbm::velocitySet::as2()) * static_cast<double>(nu)));
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_ref() noexcept
    {
        return omega_from_nu(physics::visc_ref);
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_zmin() noexcept
    {
        return omega_ref();
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_zmax() noexcept
    {
        return omega_ref();
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omega_delta() noexcept
    {
        return omega_zmax() - omega_zmin();
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omco_ref() noexcept
    {
        return static_cast<scalar_t>(1) - omega_ref();
    }

    __device__ __host__ [[nodiscard]] static inline consteval scalar_t omco_zmin() noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmin();
    }

    __device__ __host__ [[nodiscard]] static inline constexpr scalar_t omco_zmax(const scalar_t phi) noexcept
    {
        return static_cast<scalar_t>(1) - omega_zmax();
    }

#endif
}

#endif