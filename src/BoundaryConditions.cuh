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

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cz<Q>() == 1)
                    {
                        const int xx = static_cast<int>(x) + velocitySet::cx<Q>();
                        const int yy = static_cast<int>(y) + velocitySet::cy<Q>();

                        const label_t fluidNode = device::global3(xx, yy, 1);

                        constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                        constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                        constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                        const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                        const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                        const scalar_t fneq = velocitySet::f_neq<Q>(d.pxx[fluidNode], d.pyy[fluidNode], d.pzz[fluidNode],
                                                                    d.pxy[fluidNode], d.pxz[fluidNode], d.pyz[fluidNode],
                                                                    d.ux[fluidNode], d.uy[fluidNode], d.uz[fluidNode]);

                        d.f[Q * size::cells() + fluidNode] = to_pop(feq + relaxation::omco_zmin() * fneq);
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

            if (x == 0 || x == mesh::nx - 1 || y == 0 || y == mesh::ny - 1)
            {
                return;
            }

            const label_t idx3_bnd = device::global3(x, y, mesh::nz - 1);
            const label_t idx3_zm1 = device::global3(x, y, mesh::nz - 2);

            // ----------------------------
            // Dong OBC: use stored boundary macros as u^n (no extra arrays)
            // ----------------------------
            const scalar_t ux_old = d.ux[idx3_bnd];
            const scalar_t uy_old = d.uy[idx3_bnd];
            const scalar_t uz_old = d.uz[idx3_bnd];

            // Keep your existing "copy scalars from interior" behavior
            d.rho[idx3_bnd] = d.rho[idx3_zm1];
            d.phi[idx3_bnd] = d.phi[idx3_zm1];

            const scalar_t rho = d.rho[idx3_bnd];
            const scalar_t phi = d.phi[idx3_bnd];

            // Interior velocity
            const scalar_t ux_i = d.ux[idx3_zm1];
            const scalar_t uy_i = d.uy[idx3_zm1];
            const scalar_t uz_i = d.uz[idx3_zm1];

            // Default: passive outflow (do not "actuate" the domain)
            scalar_t ux = ux_i;
            scalar_t uy = uy_i;
            scalar_t uz = uz_i;

            // Backflow detection: n = +ez, so un = uz
            const scalar_t un_i = uz_i;

            if (un_i < static_cast<scalar_t>(0))
            {
                // Viscosity in lattice units from BGK relaxation
                constexpr scalar_t nu = velocitySet::cs2() * (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) - relaxation::omco_ref()) - static_cast<scalar_t>(0.5));

                // Convection speed scale Uc = 1/D0 (no reductions => default to u_inf)
                const scalar_t U0 = physics::u_inf;
                const scalar_t Uc = physics::u_inf;
                const scalar_t D0 = static_cast<scalar_t>(1) / Uc;

                // Theta_0(n,u): use boundary-stored u^n for the smooth switch
                constexpr scalar_t delta = static_cast<scalar_t>(0.1);
                const scalar_t un = uz_old;
                const scalar_t Theta = static_cast<scalar_t>(0.5) * (static_cast<scalar_t>(1) - tanh(un / (delta * U0)));

                // E(n,u*) with u* = u^n_b
                const scalar_t u2 = ux_old * ux_old + uy_old * uy_old + uz_old * uz_old;
                const scalar_t Ex = static_cast<scalar_t>(0.5) * (un * ux_old) * Theta;
                const scalar_t Ey = static_cast<scalar_t>(0.5) * (un * uy_old) * Theta;
                const scalar_t Ez = static_cast<scalar_t>(0.5) * (u2 + un * un) * Theta;

                // (nu(D0+1)) u_b^{n+1} = nu D0 u_b^n + nu u_i + E
                const scalar_t denom = nu * (D0 + static_cast<scalar_t>(1));

                ux = (nu * D0 * ux_old + nu * ux_i + Ex) / denom;
                uy = (nu * D0 * uy_old + nu * uy_i + Ey) / denom;
                uz = (nu * D0 * uz_old + nu * uz_i + Ez) / denom;
            }

            // Store boundary macros (used as u^n next step)
            d.ux[idx3_bnd] = ux;
            d.uy[idx3_bnd] = uy;
            d.uz[idx3_bnd] = uz;

            const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

            device::constexpr_for<0, velocitySet::Q()>(
                [&](const auto Q)
                {
                    if constexpr (velocitySet::cz<Q>() == -1)
                    {
                        int xx = static_cast<int>(x) + velocitySet::cx<Q>();
                        int yy = static_cast<int>(y) + velocitySet::cy<Q>();

                        xx = device::periodic_wrap<mesh::nx>(xx);
                        yy = device::periodic_wrap<mesh::ny>(yy);

                        const label_t fluidNode = device::global3(xx, yy, mesh::nz - 2);

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
