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
    CUDA kernels for low-order gradient (D3Q7) phase field

Namespace
    phase

SourceFiles
    phaseField.cuh

\*---------------------------------------------------------------------------*/

#ifndef PHASEFIELD_CUH
#define PHASEFIELD_CUH

namespace Phase
{
    __global__ void computePhase(LBMFields d)
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

        scalar_t phi = static_cast<scalar_t>(0);
        device::constexpr_for<0, VelocitySet::Q()>(
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

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
            x == 0 || x == mesh::nx - 1 ||
            y == 0 || y == mesh::ny - 1 ||
            z == 0 || z == mesh::nz - 1)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        scalar_t gx = static_cast<scalar_t>(0.5) * (d.phi[device::global3(x + 1, y, z)] - d.phi[device::global3(x - 1, y, z)]);
        scalar_t gy = static_cast<scalar_t>(0.5) * (d.phi[device::global3(x, y + 1, z)] - d.phi[device::global3(x, y - 1, z)]);
        scalar_t gz = static_cast<scalar_t>(0.5) * (d.phi[device::global3(x, y, z + 1)] - d.phi[device::global3(x, y, z - 1)]);

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

        if (x >= mesh::nx || y >= mesh::ny || z >= mesh::nz ||
            x == 0 || x == mesh::nx - 1 ||
            y == 0 || y == mesh::ny - 1 ||
            z == 0 || z == mesh::nz - 1)
        {
            return;
        }

        const label_t idx3 = device::global3(x, y, z);

        const scalar_t curvature = static_cast<scalar_t>(0.5) *
                                   ((d.normx[device::global3(x + 1, y, z)] - d.normx[device::global3(x - 1, y, z)]) +
                                    (d.normy[device::global3(x, y + 1, z)] - d.normy[device::global3(x, y - 1, z)]) +
                                    (d.normz[device::global3(x, y, z + 1)] - d.normz[device::global3(x, y, z - 1)]));

        const scalar_t stCurv = -physics::sigma * curvature * d.ind[idx3];
        d.ffx[idx3] = stCurv * d.normx[idx3];
        d.ffy[idx3] = stCurv * d.normy[idx3];
        d.ffz[idx3] = stCurv * d.normz[idx3];
    }
}

#endif