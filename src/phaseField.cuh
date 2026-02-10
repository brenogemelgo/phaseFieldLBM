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
    High-order phase-field kernels computing order parameter, interface normals, curvature, and surface-tension forcing using D3Q19/D3Q27 stencils

Namespace
    Phase

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

        const scalar_t phi_xp1_yp1_z = d.phi[device::global3(x + 1, y + 1, z)];
        const scalar_t phi_xp1_y_zp1 = d.phi[device::global3(x + 1, y, z + 1)];
        const scalar_t phi_xp1_ym1_z = d.phi[device::global3(x + 1, y - 1, z)];
        const scalar_t phi_xp1_y_zm1 = d.phi[device::global3(x + 1, y, z - 1)];
        const scalar_t phi_xm1_ym1_z = d.phi[device::global3(x - 1, y - 1, z)];
        const scalar_t phi_xm1_y_zm1 = d.phi[device::global3(x - 1, y, z - 1)];
        const scalar_t phi_xm1_yp1_z = d.phi[device::global3(x - 1, y + 1, z)];
        const scalar_t phi_xm1_y_zp1 = d.phi[device::global3(x - 1, y, z + 1)];
        const scalar_t phi_x_yp1_zp1 = d.phi[device::global3(x, y + 1, z + 1)];
        const scalar_t phi_x_yp1_zm1 = d.phi[device::global3(x, y + 1, z - 1)];
        const scalar_t phi_x_ym1_zm1 = d.phi[device::global3(x, y - 1, z - 1)];
        const scalar_t phi_x_ym1_zp1 = d.phi[device::global3(x, y - 1, z + 1)];

        scalar_t sgx = LBM::velocitySet::w_1() * (d.phi[device::global3(x + 1, y, z)] - d.phi[device::global3(x - 1, y, z)]) +
                       LBM::velocitySet::w_2() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                  phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                  phi_xp1_ym1_z - phi_xm1_yp1_z +
                                                  phi_xp1_y_zm1 - phi_xm1_y_zp1);

        scalar_t sgy = LBM::velocitySet::w_1() * (d.phi[device::global3(x, y + 1, z)] - d.phi[device::global3(x, y - 1, z)]) +
                       LBM::velocitySet::w_2() * (phi_xp1_yp1_z - phi_xm1_ym1_z +
                                                  phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                  phi_xm1_yp1_z - phi_xp1_ym1_z +
                                                  phi_x_yp1_zm1 - phi_x_ym1_zp1);

        scalar_t sgz = LBM::velocitySet::w_1() * (d.phi[device::global3(x, y, z + 1)] - d.phi[device::global3(x, y, z - 1)]) +
                       LBM::velocitySet::w_2() * (phi_xp1_y_zp1 - phi_xm1_y_zm1 +
                                                  phi_x_yp1_zp1 - phi_x_ym1_zm1 +
                                                  phi_xm1_y_zp1 - phi_xp1_y_zm1 +
                                                  phi_x_ym1_zp1 - phi_x_yp1_zm1);

        if constexpr (LBM::velocitySet::Q() == 27)
        {
            const scalar_t phi_xp1_yp1_zp1 = d.phi[device::global3(x + 1, y + 1, z + 1)];
            const scalar_t phi_xp1_yp1_zm1 = d.phi[device::global3(x + 1, y + 1, z - 1)];
            const scalar_t phi_xp1_ym1_zp1 = d.phi[device::global3(x + 1, y - 1, z + 1)];
            const scalar_t phi_xp1_ym1_zm1 = d.phi[device::global3(x + 1, y - 1, z - 1)];
            const scalar_t phi_xm1_ym1_zm1 = d.phi[device::global3(x - 1, y - 1, z - 1)];
            const scalar_t phi_xm1_ym1_zp1 = d.phi[device::global3(x - 1, y - 1, z + 1)];
            const scalar_t phi_xm1_yp1_zm1 = d.phi[device::global3(x - 1, y + 1, z - 1)];
            const scalar_t phi_xm1_yp1_zp1 = d.phi[device::global3(x - 1, y + 1, z + 1)];

            sgx += LBM::D3Q27::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                        phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                        phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                        phi_xp1_ym1_zm1 - phi_xm1_yp1_zp1);

            sgy += LBM::D3Q27::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                        phi_xp1_yp1_zm1 - phi_xm1_ym1_zp1 +
                                        phi_xm1_yp1_zm1 - phi_xp1_ym1_zp1 +
                                        phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);

            sgz += LBM::D3Q27::w_3() * (phi_xp1_yp1_zp1 - phi_xm1_ym1_zm1 +
                                        phi_xm1_ym1_zp1 - phi_xp1_yp1_zm1 +
                                        phi_xp1_ym1_zp1 - phi_xm1_yp1_zm1 +
                                        phi_xm1_yp1_zp1 - phi_xp1_ym1_zm1);
        }

        const scalar_t gx = LBM::velocitySet::as2() * sgx;
        const scalar_t gy = LBM::velocitySet::as2() * sgy;
        const scalar_t gz = LBM::velocitySet::as2() * sgz;

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

        const label_t xp1_yp1_z = device::global3(x + 1, y + 1, z);
        const label_t xp1_y_zp1 = device::global3(x + 1, y, z + 1);
        const label_t xp1_ym1_z = device::global3(x + 1, y - 1, z);
        const label_t xp1_y_zm1 = device::global3(x + 1, y, z - 1);
        const label_t xm1_ym1_z = device::global3(x - 1, y - 1, z);
        const label_t xm1_y_zm1 = device::global3(x - 1, y, z - 1);
        const label_t xm1_yp1_z = device::global3(x - 1, y + 1, z);
        const label_t xm1_y_zp1 = device::global3(x - 1, y, z + 1);
        const label_t x_yp1_zp1 = device::global3(x, y + 1, z + 1);
        const label_t x_yp1_zm1 = device::global3(x, y + 1, z - 1);
        const label_t x_ym1_zm1 = device::global3(x, y - 1, z - 1);
        const label_t x_ym1_zp1 = device::global3(x, y - 1, z + 1);

        scalar_t scx = LBM::velocitySet::w_1() * (d.normx[device::global3(x + 1, y, z)] - d.normx[device::global3(x - 1, y, z)]) +
                       LBM::velocitySet::w_2() * (d.normx[xp1_yp1_z] - d.normx[xm1_ym1_z] +
                                                  d.normx[xp1_y_zp1] - d.normx[xm1_y_zm1] +
                                                  d.normx[xp1_ym1_z] - d.normx[xm1_yp1_z] +
                                                  d.normx[xp1_y_zm1] - d.normx[xm1_y_zp1]);

        scalar_t scy = LBM::velocitySet::w_1() * (d.normy[device::global3(x, y + 1, z)] - d.normy[device::global3(x, y - 1, z)]) +
                       LBM::velocitySet::w_2() * (d.normy[xp1_yp1_z] - d.normy[xm1_ym1_z] +
                                                  d.normy[x_yp1_zp1] - d.normy[x_ym1_zm1] +
                                                  d.normy[xm1_yp1_z] - d.normy[xp1_ym1_z] +
                                                  d.normy[x_yp1_zm1] - d.normy[x_ym1_zp1]);

        scalar_t scz = LBM::velocitySet::w_1() * (d.normz[device::global3(x, y, z + 1)] - d.normz[device::global3(x, y, z - 1)]) +
                       LBM::velocitySet::w_2() * (d.normz[xp1_y_zp1] - d.normz[xm1_y_zm1] +
                                                  d.normz[x_yp1_zp1] - d.normz[x_ym1_zm1] +
                                                  d.normz[xm1_y_zp1] - d.normz[xp1_y_zm1] +
                                                  d.normz[x_ym1_zp1] - d.normz[x_yp1_zm1]);

        if constexpr (LBM::velocitySet::Q() == 27)
        {
            const label_t xp1_yp1_zp1 = device::global3(x + 1, y + 1, z + 1);
            const label_t xp1_yp1_zm1 = device::global3(x + 1, y + 1, z - 1);
            const label_t xp1_ym1_zp1 = device::global3(x + 1, y - 1, z + 1);
            const label_t xp1_ym1_zm1 = device::global3(x + 1, y - 1, z - 1);
            const label_t xm1_ym1_zm1 = device::global3(x - 1, y - 1, z - 1);
            const label_t xm1_ym1_zp1 = device::global3(x - 1, y - 1, z + 1);
            const label_t xm1_yp1_zm1 = device::global3(x - 1, y + 1, z - 1);
            const label_t xm1_yp1_zp1 = device::global3(x - 1, y + 1, z + 1);

            scx += LBM::D3Q27::w_3() * (d.normx[xp1_yp1_zp1] - d.normx[xm1_ym1_zm1] +
                                        d.normx[xp1_yp1_zm1] - d.normx[xm1_ym1_zp1] +
                                        d.normx[xp1_ym1_zp1] - d.normx[xm1_yp1_zm1] +
                                        d.normx[xp1_ym1_zm1] - d.normx[xm1_yp1_zp1]);

            scy += LBM::D3Q27::w_3() * (d.normy[xp1_yp1_zp1] - d.normy[xm1_ym1_zm1] +
                                        d.normy[xp1_yp1_zm1] - d.normy[xm1_ym1_zp1] +
                                        d.normy[xm1_yp1_zm1] - d.normy[xp1_ym1_zp1] +
                                        d.normy[xm1_yp1_zp1] - d.normy[xp1_ym1_zm1]);

            scz += LBM::D3Q27::w_3() * (d.normz[xp1_yp1_zp1] - d.normz[xm1_ym1_zm1] +
                                        d.normz[xm1_ym1_zp1] - d.normz[xp1_yp1_zm1] +
                                        d.normz[xp1_ym1_zp1] - d.normz[xm1_yp1_zm1] +
                                        d.normz[xm1_yp1_zp1] - d.normz[xp1_ym1_zm1]);
        }

        const scalar_t curvature = LBM::velocitySet::as2() * (scx + scy + scz);

        // const scalar_t indicator = std::abs(phi - static_cast<scalar_t>(0.5));
        // const scalar_t expontential_indicator = std::exp(-std::abs(phi - static_cast<scalar_t>(0.5)));

        // // This is arbitrary
        // constexpr const scalar_t sigma = static_cast<scalar_t>(1);

        // const scalar_t Gaussian_indicator = std::exp(-std::pow(std::abs(phi - static_cast<scalar_t>(0.5)), 2) / sigma);

        const scalar_t stCurv = -physics::sigma * curvature * d.ind[idx3];
        d.ffx[idx3] = stCurv * d.normx[idx3];
        d.ffy[idx3] = stCurv * d.normy[idx3];
        d.ffz[idx3] = stCurv * d.normz[idx3];
    }
}

#endif