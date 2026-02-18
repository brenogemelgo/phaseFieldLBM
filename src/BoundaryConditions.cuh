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
    Unified device-side implementation of inflow, outflow, and periodic LBM boundary conditions

Namespace
    lbm

SourceFiles
    BoundaryConditions.cuh

\*---------------------------------------------------------------------------*/

#ifndef BOUNDARYCONDITIONS_CUH
#define BOUNDARYCONDITIONS_CUH

namespace lbm
{
    class BoundaryConditions
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval BoundaryConditions(){};

        __device__ static inline void applyInflow(
            LBMFields d,
            const label_t t) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            const scalar_t dx = static_cast<scalar_t>(x) - geometry::center_x();
            const scalar_t dy = static_cast<scalar_t>(y) - geometry::center_y();
            const scalar_t r2 = dx * dx + dy * dy;

            if (r2 > geometry::R2())
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, 0);
            const label_t idx3_zp1 = device::global3(x, y, 1);

            constexpr scalar_t sigma_u = static_cast<scalar_t>(0.08) * physics::u_inf;

            const scalar_t zx = white_noise<0xA341316Cu>(x, y, t);
            const scalar_t zy = white_noise<0xC8013EA4u>(x, y, t);

            const scalar_t rho = static_cast<scalar_t>(1);
            const scalar_t phi = static_cast<scalar_t>(1);
            const scalar_t ux = sigma_u * zx;
            const scalar_t uy = sigma_u * zy;
            const scalar_t uz = physics::u_inf;

            d.rho[idx3_bnd] = rho;
            d.phi[idx3_bnd] = phi;
            d.ux[idx3_bnd] = ux;
            d.uy[idx3_bnd] = uy;
            d.uz[idx3_bnd] = uz;

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            const label_t t_store = t + 1;

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cz<Q>() == 1)
                    {
                        const int xx = static_cast<int>(x) + velocitySet::cx<Q>();
                        const int yy = static_cast<int>(y) + velocitySet::cy<Q>();

                        const label_t fluidNode = device::global3(xx, yy, 1);

                        constexpr scalar_t w = velocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        esopull::store_phys<velocitySet, Q>(d, xx, yy, 1, feq + relaxation::omco_zmin() * fneq, t_store);
                    }
                });

            d.g[5 * size::cells() + idx3_zp1] = phase::velocitySet::w<5>() * phi * (static_cast<scalar_t>(1) + phase::velocitySet::as2() * uz);
        }

        __device__ static inline void applyOutflow(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            if (x == 0 || x >= mesh::nx - 1 || y == 0 || y >= mesh::ny - 1)
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, mesh::nz - 1);
            const label_t idx3_zm1 = device::global3(x, y, mesh::nz - 2);

            d.rho[idx3_bnd] = d.rho[idx3_zm1];
            d.phi[idx3_bnd] = d.phi[idx3_zm1];
            d.ux[idx3_bnd] = d.ux[idx3_zm1];
            d.uy[idx3_bnd] = d.uy[idx3_zm1];
            d.uz[idx3_bnd] = d.uz[idx3_zm1];

            const scalar_t rho = d.rho[idx3_bnd];
            const scalar_t phi = d.phi[idx3_bnd];
            const scalar_t ux = d.ux[idx3_bnd];
            const scalar_t uy = d.uy[idx3_bnd];
            const scalar_t uz = d.uz[idx3_bnd];

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cz<Q>() == -1)
                    {
                        const int xx = static_cast<int>(x) + velocitySet::cx<Q>();
                        const int yy = static_cast<int>(y) + velocitySet::cy<Q>();

                        const label_t fluidNode = device::global3(xx, yy, mesh::nz - 2);

                        constexpr scalar_t w = velocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);
                        const scalar_t force = velocitySet::force<Q>(cu, ux, uy, uz, d.fsx[fluidNode], d.fsy[fluidNode], d.fsz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_ref() * fneq + static_cast<scalar_t>(0.5) * force);
                    }
                });

            d.g[6 * size::cells() + idx3_zm1] = phase::velocitySet::w<6>() * phi * (static_cast<scalar_t>(1) - phase::velocitySet::as2() * uz);
        }

        __device__ static inline void applyConvectiveOutflow(LBMFields d) noexcept
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= mesh::nx || y >= mesh::ny)
            {
                return;
            }

            if (x == 0 || x >= mesh::nx - 1 || y == 0 || y >= mesh::ny - 1)
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, mesh::nz - 1);
            const label_t idx3_zm1 = device::global3(x, y, mesh::nz - 2);

            // Interior values
            const scalar_t rho_zm1 = d.rho[idx3_zm1];
            const scalar_t phi_zm1 = d.phi[idx3_zm1];
            const scalar_t ux_zm1 = d.ux[idx3_zm1];
            const scalar_t uy_zm1 = d.uy[idx3_zm1];
            const scalar_t uz_zm1 = d.uz[idx3_zm1];

            // Boundary values
            scalar_t rho = d.rho[idx3_bnd];
            scalar_t phi = d.phi[idx3_bnd];
            scalar_t ux = d.ux[idx3_bnd];
            scalar_t uy = d.uy[idx3_bnd];
            scalar_t uz = d.uz[idx3_bnd];

            const scalar_t Uc = uz_zm1;
            const scalar_t alpha = math::clamp<static_cast<scalar_t>(0), static_cast<scalar_t>(1)>(Uc);

            // Convective update
            rho += alpha * (rho_zm1 - rho);
            phi += alpha * (phi_zm1 - phi);
            ux += alpha * (ux_zm1 - ux);
            uy += alpha * (uy_zm1 - uy);
            uz += alpha * (uz_zm1 - uz);

            // Store updated boundary macros
            d.rho[idx3_bnd] = rho;
            d.phi[idx3_bnd] = phi;
            d.ux[idx3_bnd] = ux;
            d.uy[idx3_bnd] = uy;
            d.uz[idx3_bnd] = uz;

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cz<Q>() == -1)
                    {
                        const int xx = static_cast<int>(x) + velocitySet::cx<Q>();
                        const int yy = static_cast<int>(y) + velocitySet::cy<Q>();

                        const label_t fluidNode = device::global3(xx, yy, mesh::nz - 2);

                        constexpr scalar_t w = velocitySet::w<Q>();
                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_ref() * fneq);
                    }
                });

            d.g[6 * size::cells() + idx3_zm1] = phase::velocitySet::w<6>() * phi * (static_cast<scalar_t>(1) - phase::velocitySet::as2() * uz);

            // =========== recompute rho locally ===========
            // scalar_t rho_new = 0;
            // for all Q : rho_new += from_pop(d.f[Q * size::cells() + idx3_zm1]);
            // const scalar_t drho = rho_zm1 - rho_new;

            // =========== redistribute correction isotropically ===========
            // for all Q : d.f[Q * size::cells() + idx3_zm1] = to_pop(from_pop(d.f[...]) + velocitySet::w<Q>() * drho);
        }

        __device__ static inline void periodicX(LBMFields d)
        {
            const label_t y = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

            if (y <= 0 || y >= mesh::ny - 1 || z <= 0 || z >= mesh::nz - 1)
            {
                return;
            }

            const label_t bL = device::global3(1, y, z);
            const label_t bR = device::global3(mesh::nx - 2, y, z);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cx<Q>() > 0)
                    {
                        d.f[Q * size::cells() + bL] = d.f[Q * size::cells() + bR];
                    }
                    if constexpr (velocitySet::cx<Q>() < 0)
                    {
                        d.f[Q * size::cells() + bR] = d.f[Q * size::cells() + bL];
                    }
                });

            device::constexpr_for<0, phase::velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (phase::velocitySet::cx<Q>() > 0)
                    {
                        d.g[Q * size::cells() + bL] = d.g[Q * size::cells() + bR];
                    }
                    if constexpr (phase::velocitySet::cx<Q>() < 0)
                    {
                        d.g[Q * size::cells() + bR] = d.g[Q * size::cells() + bL];
                    }
                });

            // Copy to ghost layer (periodic wrapping)
            const label_t gL = device::global3(0, y, z);
            const label_t gR = device::global3(mesh::nx - 1, y, z);

            d.phi[gL] = d.phi[bR];
            d.phi[gR] = d.phi[bL];

            d.ux[gL] = d.ux[bR];
            d.ux[gR] = d.ux[bL];

            d.uy[gL] = d.uy[bR];
            d.uy[gR] = d.uy[bL];

            d.uz[gL] = d.uz[bR];
            d.uz[gR] = d.uz[bL];
        }

        __device__ static inline void periodicY(LBMFields d)
        {
            const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
            const label_t z = threadIdx.y + blockIdx.y * blockDim.y;

            if (x <= 0 || x >= mesh::nx - 1 || z <= 0 || z >= mesh::nz - 1)
            {
                return;
            }

            const label_t bB = device::global3(x, 1, z);
            const label_t bT = device::global3(x, mesh::ny - 2, z);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cy<Q>() > 0)
                    {
                        d.f[Q * size::cells() + bB] = d.f[Q * size::cells() + bT];
                    }
                    if constexpr (velocitySet::cy<Q>() < 0)
                    {
                        d.f[Q * size::cells() + bT] = d.f[Q * size::cells() + bB];
                    }
                });

            device::constexpr_for<0, phase::velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (phase::velocitySet::cy<Q>() > 0)
                    {
                        d.g[Q * size::cells() + bB] = d.g[Q * size::cells() + bT];
                    }
                    if constexpr (phase::velocitySet::cy<Q>() < 0)
                    {
                        d.g[Q * size::cells() + bT] = d.g[Q * size::cells() + bB];
                    }
                });

            // Copy to ghost layer (periodic wrapping)
            const label_t gB = device::global3(x, 0, z);
            const label_t gT = device::global3(x, mesh::ny - 1, z);

            d.phi[gB] = d.phi[bT];
            d.phi[gT] = d.phi[bB];

            d.ux[gB] = d.ux[bT];
            d.ux[gT] = d.ux[bB];

            d.uy[gB] = d.uy[bT];
            d.uy[gT] = d.uy[bB];

            d.uz[gB] = d.uz[bT];
            d.uz[gT] = d.uz[bB];
        }

    private:
        __device__ [[nodiscard]] static inline constexpr uint32_t hash32(uint32_t x) noexcept
        {
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;

            return x;
        }

        __device__ [[nodiscard]] static inline constexpr scalar_t uniform01(const uint32_t seed) noexcept
        {
            constexpr scalar_t inv2_32 = static_cast<scalar_t>(2.3283064365386963e-10);

            return (static_cast<scalar_t>(seed) + static_cast<scalar_t>(0.5)) * inv2_32;
        }

        __device__ [[nodiscard]] static inline scalar_t box_muller(
            scalar_t rrx,
            const scalar_t rry) noexcept
        {
            rrx = math::max(rrx, static_cast<scalar_t>(1e-12));
            const scalar_t r = math::sqrt(-static_cast<scalar_t>(2) * math::log(rrx));
            const scalar_t theta = math::two_pi() * rry;

            return r * math::cos(theta);
        }

        template <uint32_t SALT = 0u>
        __device__ [[nodiscard]] static inline constexpr scalar_t white_noise(
            const label_t x,
            const label_t y,
            const label_t STEP) noexcept
        {
            const uint32_t base = (0x9E3779B9u ^ SALT) ^ static_cast<uint32_t>(x) ^ (static_cast<uint32_t>(y) * 0x85EBCA6Bu) ^ (static_cast<uint32_t>(STEP) * 0xC2B2AE35u);

            const scalar_t rrx = uniform01(hash32(base));
            const scalar_t rry = uniform01(hash32(base ^ 0x68BC21EBu));

            return box_muller(rrx, rry);
        }
    };
}

#endif
