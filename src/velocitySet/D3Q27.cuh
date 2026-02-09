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
    D3Q27 velocity set class declaration

SourceFiles
    D3Q27.cuh

\*---------------------------------------------------------------------------*/

#ifndef D3Q27_CUH
#define D3Q27_CUH

#include "velocitySet.cuh"

namespace LBM
{
    class D3Q27 : private velocitySet
    {
    public:
        __host__ __device__ [[nodiscard]] inline consteval D3Q27(){};

        __host__ __device__ [[nodiscard]] static inline consteval label_t Q() noexcept
        {
            return static_cast<label_t>(Q_);
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t as2() noexcept
        {
            return static_cast<scalar_t>(3);
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t cs2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(3));
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w_0() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(8) / static_cast<double>(27));
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w_1() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(2) / static_cast<double>(27));
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w_2() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(54));
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w_3() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(216));
        }

        template <label_t Q>
        __host__ __device__ [[nodiscard]] static inline consteval scalar_t w() noexcept
        {
            if constexpr (Q == 0)
            {
                return w_0();
            }
            else if constexpr (Q >= 1 && Q <= 6)
            {
                return w_1();
            }
            else if constexpr (Q >= 7 && Q <= 18)
            {
                return w_2();
            }
            else
            {
                return w_3();
            }
        }

        template <label_t Q>
        __host__ __device__ [[nodiscard]] static inline consteval int cx() noexcept
        {
            if constexpr (Q == 1 || Q == 7 || Q == 9 || Q == 13 || Q == 15 || Q == 19 || Q == 21 || Q == 23 || Q == 26)
            {
                return 1;
            }
            else if constexpr (Q == 2 || Q == 8 || Q == 10 || Q == 14 || Q == 16 || Q == 20 || Q == 22 || Q == 24 || Q == 25)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __host__ __device__ [[nodiscard]] static inline consteval int cy() noexcept
        {
            if constexpr (Q == 3 || Q == 7 || Q == 11 || Q == 14 || Q == 17 || Q == 19 || Q == 21 || Q == 24 || Q == 25)
            {
                return 1;
            }
            else if constexpr (Q == 4 || Q == 8 || Q == 12 || Q == 13 || Q == 18 || Q == 20 || Q == 22 || Q == 23 || Q == 26)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        template <label_t Q>
        __host__ __device__ [[nodiscard]] static inline consteval int cz() noexcept
        {
            if constexpr (Q == 5 || Q == 9 || Q == 11 || Q == 16 || Q == 18 || Q == 19 || Q == 22 || Q == 23 || Q == 25)
            {
                return 1;
            }
            else if constexpr (Q == 6 || Q == 10 || Q == 12 || Q == 15 || Q == 17 || Q == 20 || Q == 21 || Q == 24 || Q == 26)
            {
                return -1;
            }
            else
            {
                return 0;
            }
        }

        __host__ __device__ [[nodiscard]] static inline consteval scalar_t oos() noexcept
        {
            return static_cast<scalar_t>(static_cast<double>(1) / static_cast<double>(6));
        }

        template <label_t Q>
        __host__ __device__ [[nodiscard]] static inline constexpr scalar_t f_eq(
            const scalar_t rho,
            const scalar_t uu,
            const scalar_t cu) noexcept
        {
            return w<Q>() * rho * (static_cast<scalar_t>(1) - uu + cu + static_cast<scalar_t>(0.5) * cu * cu + oos() * cu * cu * cu - uu * cu) - w<Q>();
        }

        template <label_t Q>
        __host__ __device__ [[nodiscard]] static inline constexpr scalar_t f_neq(
            const scalar_t pxx,
            const scalar_t pyy,
            const scalar_t pzz,
            const scalar_t pxy,
            const scalar_t pxz,
            const scalar_t pyz,
            const scalar_t ux,
            const scalar_t uy,
            const scalar_t uz) noexcept
        {
            return (w<Q>() * static_cast<scalar_t>(4.5)) *
                   ((cx<Q>() * cx<Q>() - cs2()) * pxx +
                    (cy<Q>() * cy<Q>() - cs2()) * pyy +
                    (cz<Q>() * cz<Q>() - cs2()) * pzz +
                    static_cast<scalar_t>(2) * (cx<Q>() * cy<Q>() * pxy +
                                                cx<Q>() * cz<Q>() * pxz +
                                                cy<Q>() * cz<Q>() * pyz) +
                    (cx<Q>() * cx<Q>() * cx<Q>() - cx<Q>()) * (as2() * ux * pxx) +
                    (cy<Q>() * cy<Q>() * cy<Q>() - cy<Q>()) * (as2() * uy * pyy) +
                    (cz<Q>() * cz<Q>() * cz<Q>() - cz<Q>()) * (as2() * uz * pzz) +
                    as2() * ((cx<Q>() * cx<Q>() * cy<Q>() - cs2() * cy<Q>()) * (pxx * uy + static_cast<scalar_t>(2) * ux * pxy) +
                             (cx<Q>() * cx<Q>() * cz<Q>() - cs2() * cz<Q>()) * (pxx * uz + static_cast<scalar_t>(2) * ux * pxz) +
                             (cx<Q>() * cy<Q>() * cy<Q>() - cs2() * cx<Q>()) * (pyy * ux + static_cast<scalar_t>(2) * uy * pxy) +
                             (cy<Q>() * cy<Q>() * cz<Q>() - cs2() * cz<Q>()) * (pyy * uz + static_cast<scalar_t>(2) * uy * pyz) +
                             (cx<Q>() * cz<Q>() * cz<Q>() - cs2() * cx<Q>()) * (pzz * ux + static_cast<scalar_t>(2) * uz * pxz) +
                             (cy<Q>() * cz<Q>() * cz<Q>() - cs2() * cy<Q>()) * (pzz * uy + static_cast<scalar_t>(2) * uz * pyz)) +
                    static_cast<scalar_t>(6) * cx<Q>() * cy<Q>() * cz<Q>() * (ux * pyz + uy * pxz + uz * pxy));
        }

        template <label_t Q>
        __host__ __device__ [[nodiscard]] static inline constexpr scalar_t force(
            const scalar_t cu,
            const scalar_t ux,
            const scalar_t uy,
            const scalar_t uz,
            const scalar_t ffx,
            const scalar_t ffy,
            const scalar_t ffz) noexcept
        {
            return static_cast<scalar_t>(0.5) * w<Q>() *
                   ((as2() * (cx<Q>() - ux) + as2() * cu * cx<Q>()) * ffx +
                    (as2() * (cy<Q>() - uy) + as2() * cu * cy<Q>()) * ffy +
                    (as2() * (cz<Q>() - uz) + as2() * cu * cz<Q>()) * ffz);
        }

    private:
        static constexpr label_t Q_ = 27;
    };
}

#endif