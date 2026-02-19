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
    Core LBM kernels for moment computation, collision–streaming, forcing, and coupled phase-field transport

Namespace
    lbm

SourceFiles
    lbm.cuh

\*---------------------------------------------------------------------------*/

#ifndef LBM_CUH
#define LBM_CUH

#include "LBMIncludes.cuh"

namespace lbm
{
    template <class VS>
    __global__ void encodeStandardToEsoteric_t0(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (device::guard(x, y, z))
            return;

        const label_t n = device::global3(x, y, z);

        // read physical pops from standard layout
        scalar_t phys[VS::Q()];
        device::constexpr_for<0, VS::Q()>(
            [&](const auto Q)
            {
                phys[Q] = from_pop(d.f[Q * size::cells() + n]);
            });

        // write Esoteric Pull storage so that load_f(..., t=0) reconstructs phys[]
        // t=0 => odd=false
        d.f[0 * size::cells() + n] = to_pop(phys[0]);

        device::constexpr_for<0, (VS::Q() - 1) / 2>(
            [&](const auto K)
            {
                constexpr label_t k = K.value;
                constexpr label_t i = static_cast<label_t>(2 * k + 1);

                const label_t xx = x + static_cast<label_t>(VS::template cx<i>());
                const label_t yy = y + static_cast<label_t>(VS::template cy<i>());
                const label_t zz = z + static_cast<label_t>(VS::template cz<i>());
                const label_t j = device::global3(xx, yy, zz);

                // From Listing A5 load for even t:
                // phys[i]   = f[i+1] at n  => store f[i+1](n) = phys[i]
                // phys[i+1] = f[i]   at j  => store f[i](j)   = phys[i+1]
                d.f[(i + 1) * size::cells() + n] = to_pop(phys[i]);
                d.f[i * size::cells() + j] = to_pop(phys[i + 1]);
            });
    }

    __global__ void computeMoments(LBMFields d, const label_t STEP)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (device::guard(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t pop[velocitySet::Q()];
        esopull::load_f<velocitySet>(d, x, y, z, pop, STEP);

        scalar_t rho = static_cast<scalar_t>(0);
        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                rho += pop[Q];
            });

        rho += static_cast<scalar_t>(1);
        d.rho[idx3] = rho;

        const scalar_t fsx = d.fsx[idx3];
        const scalar_t fsy = d.fsy[idx3];
        const scalar_t fsz = d.fsz[idx3];

        const scalar_t invRho = static_cast<scalar_t>(1) / rho;

        scalar_t ux = static_cast<scalar_t>(0);
        scalar_t uy = static_cast<scalar_t>(0);
        scalar_t uz = static_cast<scalar_t>(0);

        if constexpr (velocitySet::Q() == 19)
        {
            ux = invRho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]);
            uy = invRho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]);
            uz = invRho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]);
        }
        else if constexpr (velocitySet::Q() == 27)
        {
            ux = invRho * (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] + pop[26] - pop[25]);
            uy = invRho * (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] + pop[24] - pop[23] + pop[25] - pop[26]);
            uz = invRho * (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17] + pop[19] - pop[20] + pop[22] - pop[21] + pop[23] - pop[24] + pop[25] - pop[26]);
        }

        ux += fsx * static_cast<scalar_t>(0.5) * invRho;
        uy += fsy * static_cast<scalar_t>(0.5) * invRho;
        uz += fsz * static_cast<scalar_t>(0.5) * invRho;

        d.ux[idx3] = ux;
        d.uy[idx3] = uy;
        d.uz[idx3] = uz;

        scalar_t pxx = static_cast<scalar_t>(0), pyy = static_cast<scalar_t>(0), pzz = static_cast<scalar_t>(0);
        scalar_t pxy = static_cast<scalar_t>(0), pxz = static_cast<scalar_t>(0), pyz = static_cast<scalar_t>(0);

        const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);
        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                const scalar_t force = velocitySet::force<Q>(cu, ux, uy, uz, fsx, fsy, fsz);
                const scalar_t fneq = pop[Q] - feq + static_cast<scalar_t>(0.5) * force;

                pxx += fneq * cx * cx;
                pyy += fneq * cy * cy;
                pzz += fneq * cz * cz;
                pxy += fneq * cx * cy;
                pxz += fneq * cx * cz;
                pyz += fneq * cy * cz;
            });

        const scalar_t trace = pxx + pyy + pzz;
        pxx -= velocitySet::cs2() * trace;
        pyy -= velocitySet::cs2() * trace;
        pzz -= velocitySet::cs2() * trace;

        d.pxx[idx3] = pxx;
        d.pyy[idx3] = pyy;
        d.pzz[idx3] = pzz;
        d.pxy[idx3] = pxy;
        d.pxz[idx3] = pxz;
        d.pyz[idx3] = pyz;
    }

    __global__ void streamCollide(LBMFields d, const label_t STEP)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (device::guard(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t rho = d.rho[idx3];
        const scalar_t ux = d.ux[idx3];
        const scalar_t uy = d.uy[idx3];
        const scalar_t uz = d.uz[idx3];
        const scalar_t pxx = d.pxx[idx3];
        const scalar_t pyy = d.pyy[idx3];
        const scalar_t pzz = d.pzz[idx3];
        const scalar_t pxy = d.pxy[idx3];
        const scalar_t pxz = d.pxz[idx3];
        const scalar_t pyz = d.pyz[idx3];
        const scalar_t fsx = d.fsx[idx3];
        const scalar_t fsy = d.fsy[idx3];
        const scalar_t fsz = d.fsz[idx3];

        const scalar_t phi = d.phi[idx3];
        scalar_t omega = static_cast<scalar_t>(0);

        if constexpr (flowCase::jet_case())
        {
            // const scalar_t tau_phi = (static_cast<scalar_t>(1) - phi) * relaxation::tau_water() + phi * relaxation::tau_oil();
            // const scalar_t r = device::sponge_ramp(z);
            // const scalar_t tau_eff = tau_phi + r * (relaxation::tau_zmax(phi) - tau_phi);
            // omega = static_cast<scalar_t>(1) / tau_eff;
            omega = relaxation::omega_ref();
        }
        else
        {
            omega = relaxation::omega_ref();
        }

        const scalar_t omco = static_cast<scalar_t>(1) - omega;
        const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);

        scalar_t post[velocitySet::Q()];

        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                const scalar_t cu = velocitySet::as2() * (cx * ux + cy * uy + cz * uz);
                const scalar_t feq = velocitySet::f_eq<Q>(rho, uu, cu);
                const scalar_t force = velocitySet::force<Q>(cu, ux, uy, uz, fsx, fsy, fsz);
                const scalar_t fneq = velocitySet::f_neq<Q>(pxx, pyy, pzz, pxy, pxz, pyz, ux, uy, uz);

                post[Q] = feq + omco * fneq + static_cast<scalar_t>(0.5) * force;
            });

        esopull::store_f<velocitySet>(d, x, y, z, post, STEP);

        const scalar_t normx = d.normx[idx3];
        const scalar_t normy = d.normy[idx3];
        const scalar_t normz = d.normz[idx3];
        const scalar_t sharp = physics::gamma * phi * (static_cast<scalar_t>(1) - phi);

        scalar_t gpost[phase::velocitySet::Q()];

        device::constexpr_for<0, phase::velocitySet::Q()>(
            [&](auto Q)
            {
                const scalar_t geq = phase::velocitySet::g_eq<Q>(phi, ux, uy, uz);
                const scalar_t hi = phase::velocitySet::anti_diffusion<Q>(sharp, normx, normy, normz);
                gpost[Q] = geq + hi;
            });
    }

    __global__ void callInflow(LBMFields d, const label_t t)
    {
        BoundaryConditions::applyInflow(d, t);
    }

    __global__ void callOutflow(LBMFields d)
    {
        // BoundaryConditions::applyOutflow(d);
        // BoundaryConditions::applyConvectiveOutflow(d);
    }

    __global__ void callPeriodicX(LBMFields d)
    {
        // BoundaryConditions::periodicX(d);
    }

    __global__ void callPeriodicY(LBMFields d)
    {
        // BoundaryConditions::periodicY(d);
    }
}

#endif