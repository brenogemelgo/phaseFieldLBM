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
    Initial condition kernels for jet and droplet setups, density initialization, and equilibrium distribution assignment

Namespace
    LBM

SourceFiles
    initialConditions.cu

\*---------------------------------------------------------------------------*/

#ifndef INITIALCONDITIONS_CUH
#define INITIALCONDITIONS_CUH

#include "include/LBMIncludes.cuh"

namespace LBM
{
    __global__ void setJet(LBMFields d)
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

        const label_t idx3_in = device::global3(x, y, 0);

        d.phi[idx3_in] = static_cast<scalar_t>(1);
        d.uz[idx3_in] = physics::u_inf;
    }

    __global__ void setDroplet(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
            x == 0 || x == mesh::nx - 1 ||
            y == 0 || y == mesh::ny - 1 ||
            z == 0 || z == mesh::nz - 1)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t dx = (static_cast<scalar_t>(x) - geometry::center_x()) / static_cast<scalar_t>(2);
        const scalar_t dy = static_cast<scalar_t>(y) - geometry::center_y();
        const scalar_t dz = static_cast<scalar_t>(z) - geometry::center_z();
        const scalar_t L2 = math::sqrt(dx * dx + dy * dy + dz * dz);

        const scalar_t arg = static_cast<scalar_t>((static_cast<double>(mesh::radius) - static_cast<double>(L2)) / static_cast<double>(physics::interface_width));
        const scalar_t phi = static_cast<scalar_t>(0.5) + static_cast<scalar_t>(0.5) * math::tanh(static_cast<scalar_t>(2) * arg);

        d.phi[idx3] = phi;
    }

    __global__ void setInitialDensity(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        d.rho[idx3] = static_cast<scalar_t>(1);
    }

    __global__ void setDistros(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t ux = d.ux[idx3];
        const scalar_t uy = d.uy[idx3];
        const scalar_t uz = d.uz[idx3];

        const scalar_t uu = static_cast<scalar_t>(1.5) * (ux * ux + uy * uy + uz * uz);
        device::constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q)
            {
                constexpr scalar_t cx = static_cast<scalar_t>(VelocitySet::cx<Q>());
                constexpr scalar_t cy = static_cast<scalar_t>(VelocitySet::cy<Q>());
                constexpr scalar_t cz = static_cast<scalar_t>(VelocitySet::cz<Q>());

                const scalar_t cu = VelocitySet::as2() * (cx * ux + cy * uy + cz * uz);

                const scalar_t feq = VelocitySet::f_eq<Q>(d.rho[idx3], uu, cu);

                d.f[Q * size::cells() + idx3] = to_pop(feq);
            });

        device::constexpr_for<0, Phase::VelocitySet::Q()>(
            [&](const auto Q)
            {
                // const label_t xx = x + static_cast<label_t>(Phase::VelocitySet::cx<Q>());
                // const label_t yy = y + static_cast<label_t>(Phase::VelocitySet::cy<Q>());
                // const label_t zz = z + static_cast<label_t>(Phase::VelocitySet::cz<Q>());

                d.g[device::global4(x, y, z, Q)] = Phase::VelocitySet::g_eq<Q>(d.phi[idx3], ux, uy, uz);
            });
    }
}

#endif
