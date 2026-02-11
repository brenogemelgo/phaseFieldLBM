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
    High-order phase-field kernels computing order parameter, interface normals, curvature, and surface-tension forcing using D3Q19/D3Q27 stencils

Namespace
    Phase

SourceFiles
    phaseField.cuh

\*---------------------------------------------------------------------------*/

#ifndef PHASEFIELD_CUH
#define PHASEFIELD_CUH

namespace phase
{
    __global__ void computePhase(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (device::guard(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t phi = static_cast<scalar_t>(0);
        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                phi += d.g[Q * size::cells() + idx3];
            });

        d.phi[idx3] = phi;
    }

    __global__ void computeNormals(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (device::guard(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t gx = static_cast<scalar_t>(0), gy = static_cast<scalar_t>(0), gz = static_cast<scalar_t>(0);

        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                const label_t xx = x + static_cast<label_t>(velocitySet::cx<Q>());
                const label_t yy = y + static_cast<label_t>(velocitySet::cy<Q>());
                const label_t zz = z + static_cast<label_t>(velocitySet::cz<Q>());

                gx += lbm::velocitySet::as2() * lbm::velocitySet::w<Q>() * cx * d.phi[device::global3(xx, yy, zz)];
                gy += lbm::velocitySet::as2() * lbm::velocitySet::w<Q>() * cy * d.phi[device::global3(xx, yy, zz)];
                gz += lbm::velocitySet::as2() * lbm::velocitySet::w<Q>() * cz * d.phi[device::global3(xx, yy, zz)];
            });

        const scalar_t ind = math::sqrt(gx * gx + gy * gy + gz * gz);
        const scalar_t invInd = static_cast<scalar_t>(1) / (ind + static_cast<scalar_t>(1e-9));

        const scalar_t normX = gx * invInd;
        const scalar_t normY = gy * invInd;
        const scalar_t normZ = gz * invInd;

        d.ind[idx3] = ind;
        d.normx[idx3] = normX;
        d.normy[idx3] = normY;
        d.normz[idx3] = normZ;
    }

    __global__ void computeForces(LBMFields d)
    {
        const label_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const label_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const label_t z = threadIdx.z + blockIdx.z * blockDim.z;

        if (device::guard(x, y, z))
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t kappa = static_cast<scalar_t>(0);

        device::constexpr_for<0, velocitySet::Q()>(
            [&](const auto Q)
            {
                constexpr scalar_t cx = static_cast<scalar_t>(velocitySet::cx<Q>());
                constexpr scalar_t cy = static_cast<scalar_t>(velocitySet::cy<Q>());
                constexpr scalar_t cz = static_cast<scalar_t>(velocitySet::cz<Q>());

                const label_t xx = x + static_cast<label_t>(velocitySet::cx<Q>());
                const label_t yy = y + static_cast<label_t>(velocitySet::cy<Q>());
                const label_t zz = z + static_cast<label_t>(velocitySet::cz<Q>());

                kappa += lbm::velocitySet::as2() * lbm::velocitySet::w<Q>() * (cx * d.normx[device::global3(xx, yy, zz)] + cy * d.normy[device::global3(xx, yy, zz)] + cz * d.normz[device::global3(xx, yy, zz)]);
            });

        const scalar_t stCurv = -physics::sigma * kappa * d.ind[idx3];
        d.fsx[idx3] = stCurv * d.normx[idx3];
        d.fsy[idx3] = stCurv * d.normy[idx3];
        d.fsz[idx3] = stCurv * d.normz[idx3];
    }
}

#endif