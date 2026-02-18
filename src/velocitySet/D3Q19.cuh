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
    D3Q19 velocity set with second-order equilibrium and non-equilibrium plus forcing

SourceFiles
    D3Q19.cuh

\*---------------------------------------------------------------------------*/

#ifndef D3Q19_CUH
#define D3Q19_CUH

#include "VelocitySet.cuh"

namespace lbm
{
    class D3Q19 : private VelocitySet
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval D3Q19(){};

        __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return static_cast<label_t>(Q_);
        }

        __device__ __host__ [[nodiscard]] static inline consteval label_t max_abs_c() noexcept
        {
            return static_cast<label_t>(1);
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t as2() noexcept
        {
            return static_cast<scalar_t>(3);
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(3));
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_0() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(3));
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_1() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(18));
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(36));
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline consteval scalar_t w() noexcept
        {
            if constexpr (Q == 0)
            {
                return w_0();
            }
            else if constexpr (Q >= 1 && Q <= 6)
            {
                return w_1();
            }
            else
            {
                return w_2();
            }
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline consteval int cx() noexcept
        {
            if constexpr (Q == 1 || Q == 7 || Q == 9 || Q == 13 || Q == 15)
            {
                return 1;
            }
            else if constexpr (Q == 2 || Q == 8 || Q == 10 || Q == 14 || Q == 16)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline consteval int cy() noexcept
        {
            if constexpr (Q == 3 || Q == 7 || Q == 11 || Q == 14 || Q == 17)
            {
                return 1;
            }
            else if constexpr (Q == 4 || Q == 8 || Q == 12 || Q == 13 || Q == 18)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline consteval int cz() noexcept
        {
            if constexpr (Q == 5 || Q == 9 || Q == 11 || Q == 16 || Q == 18)
            {
                return 1;
            }
            else if constexpr (Q == 6 || Q == 10 || Q == 12 || Q == 15 || Q == 17)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t f_eq(
            const scalar_t rho,
            const scalar_t uu,
            const scalar_t cu) noexcept
        {
            return w<Q>() * rho * (static_cast<scalar_t>(1) - uu + cu + static_cast<scalar_t>(0.5) * cu * cu) - w<Q>();
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t f_neq(
            const scalar_t pxx,
            const scalar_t pyy,
            const scalar_t pzz,
            const scalar_t pxy,
            const scalar_t pxz,
            const scalar_t pyz,
            const scalar_t /*ux*/,
            const scalar_t /*uy*/,
            const scalar_t /*uz*/) noexcept
        {
            return (w<Q>() * static_cast<scalar_t>(4.5)) *
                   ((cx<Q>() * cx<Q>() - cs2()) * pxx +
                    (cy<Q>() * cy<Q>() - cs2()) * pyy +
                    (cz<Q>() * cz<Q>() - cs2()) * pzz +
                    static_cast<scalar_t>(2) * (cx<Q>() * cy<Q>() * pxy +
                                                cx<Q>() * cz<Q>() * pxz +
                                                cy<Q>() * cz<Q>() * pyz));
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t force(
            const scalar_t cu,
            const scalar_t ux,
            const scalar_t uy,
            const scalar_t uz,
            const scalar_t fsx,
            const scalar_t fsy,
            const scalar_t fsz) noexcept
        {
            return static_cast<scalar_t>(0.5) * w<Q>() *
                   ((as2() * (cx<Q>() - ux) + as2() * cu * cx<Q>()) * fsx +
                    (as2() * (cy<Q>() - uy) + as2() * cu * cy<Q>()) * fsy +
                    (as2() * (cz<Q>() - uz) + as2() * cu * cz<Q>()) * fsz);
        }

    private:
        static constexpr label_t Q_ = 19;
    };
}

#endif