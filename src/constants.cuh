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
   Centralized compile-time configuration of velocity sets, flow cases, mesh, and physical parameters

Namespace
    lbm
    phase

SourceFiles
    constants.cuh

\*---------------------------------------------------------------------------*/

#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include "LBMIncludes.cuh"
#include "cuda/precision.cuh"
#include "cuda/errorCheck.cuh"
#include "functions/constexprFor.cuh"
#include "velocitySet/VelocitySet.cuh"
#include "flowCase/FlowCase.cuh"

namespace lbm
{
#if defined(VS_D3Q19)
    using velocitySet = D3Q19;
#elif defined(VS_D3Q27)
    using velocitySet = D3Q27;
#endif

#if defined(DROPLET)
    using flowCase = Droplet;
#elif defined(JET)
    using flowCase = Jet;
#endif
}

namespace phase
{
    using velocitySet = lbm::D3Q7;
}

static_assert(lbm::velocitySet::max_abs_c() == phase::velocitySet::max_abs_c(), "Hydrodynamic and phase velocity sets must have identical stencil radius.");

#define RUN_MODE
// #define SAMPLE_MODE
// #define PROFILE_MODE

#if defined(RUN_MODE)

static constexpr int MACRO_SAVE = 1000;
static constexpr int NSTEPS = 200000;

#elif defined(SAMPLE_MODE)

static constexpr int MACRO_SAVE = 100;
static constexpr int NSTEPS = 10000;

#elif defined(PROFILE_MODE)

static constexpr int MACRO_SAVE = 1;
static constexpr int NSTEPS = 0;

#endif

#if defined(JET)

namespace mesh
{
    static constexpr label_t res = 128;
    static constexpr label_t nx = res;
    static constexpr label_t ny = res;
    static constexpr label_t nz = res * 2;
    static constexpr int diam = 20;
    static constexpr int radius = diam / 2;
}

namespace physics
{
    static constexpr scalar_t u_inf = static_cast<scalar_t>(0.05);

    static constexpr scalar_t reynolds_water = static_cast<scalar_t>(5000);
    static constexpr scalar_t reynolds_oil = static_cast<scalar_t>(5000);

    static constexpr scalar_t weber = static_cast<scalar_t>(500);

    static constexpr scalar_t sigma = (u_inf * u_inf * mesh::diam) / static_cast<scalar_t>(weber);

    static constexpr scalar_t interface_width = static_cast<scalar_t>(4); // continuum interface width. discretization may change it a little

    static constexpr scalar_t tau_g = static_cast<scalar_t>(1);                                            // phase field relaxation time
    static constexpr scalar_t diff_int = phase::velocitySet::cs2() * (tau_g - static_cast<scalar_t>(0.5)); // interfacial diffusivity
    static constexpr scalar_t kappa = static_cast<scalar_t>(4) * diff_int / interface_width;               // sharpening parameter
    static constexpr scalar_t gamma = kappa / phase::velocitySet::cs2();
}

#elif defined(DROPLET)

namespace mesh
{
    static constexpr label_t res = 256;
    static constexpr label_t nx = res;
    static constexpr label_t ny = res;
    static constexpr label_t nz = res;
    static constexpr int radius = 35;
    static constexpr int diam = 2 * radius;
}

namespace physics
{

    static constexpr scalar_t u_inf = static_cast<scalar_t>(0);
    static constexpr int reynolds_water = 0;
    static constexpr int reynolds_oil = 0;
    static constexpr int weber = 0;
    static constexpr scalar_t sigma = static_cast<scalar_t>(0.03);

    static constexpr scalar_t interface_width = static_cast<scalar_t>(4); // continuum interface width. discretization may change it a little

    static constexpr scalar_t tau_g = static_cast<scalar_t>(1);                                            // phase field relaxation time
    static constexpr scalar_t diff_int = phase::velocitySet::cs2() * (tau_g - static_cast<scalar_t>(0.5)); // interfacial diffusivity
    static constexpr scalar_t kappa = static_cast<scalar_t>(4) * diff_int / interface_width;               // sharpening parameter
    static constexpr scalar_t gamma = kappa / phase::velocitySet::cs2();

    static constexpr scalar_t tau = static_cast<scalar_t>(0.55);
    static constexpr scalar_t visc_ref = (tau - static_cast<scalar_t>(0.5)) / lbm::velocitySet::as2();
}

#endif

#endif