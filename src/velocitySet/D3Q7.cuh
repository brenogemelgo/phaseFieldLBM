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
    D3Q7 velocity set tailored for phase-field advection–diffusion dynamics

SourceFiles
    D3Q7.cuh

\*---------------------------------------------------------------------------*/

#ifndef D3Q7_CUH
#define D3Q7_CUH

#include "VelocitySet.cuh"

namespace LBM
{
    class D3Q7 : private VelocitySet
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval D3Q7(){};

        __device__ __host__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return static_cast<label_t>(Q_);
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t as2() noexcept
        {
            return static_cast<scalar_t>(4);
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(4));
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_0() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(4));
        }

        __device__ __host__ [[nodiscard]] static inline consteval scalar_t w_1() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(8));
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline consteval scalar_t w() noexcept
        {
            if constexpr (Q == 0)
            {
                return w_0();
            }
            else
            {
                return w_1();
            }
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline consteval int cx() noexcept
        {
            if constexpr (Q == 1)
            {
                return 1;
            }
            else if constexpr (Q == 2)
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
            if constexpr (Q == 3)
            {
                return 1;
            }
            else if constexpr (Q == 4)
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
            if constexpr (Q == 5)
            {
                return 1;
            }
            else if constexpr (Q == 6)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t g_eq(
            const scalar_t phi,
            const scalar_t ux,
            const scalar_t uy,
            const scalar_t uz) noexcept
        {
            return w<Q>() * phi * (static_cast<scalar_t>(1) + as2() * (cx<Q>() * ux + cy<Q>() * uy + cz<Q>() * uz));
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t g_neq() noexcept
        {
            return 0;
        }

        template <label_t Q>
        __device__ __host__ [[nodiscard]] static inline constexpr scalar_t anti_diffusion(
            const scalar_t sharp,
            const scalar_t normx,
            const scalar_t normy,
            const scalar_t normz) noexcept
        {
            return w<Q>() * sharp * (cx<Q>() * normx + cy<Q>() * normy + cz<Q>() * normz);
        }

    private:
        static constexpr label_t Q_ = 7;
    };
}

#endif