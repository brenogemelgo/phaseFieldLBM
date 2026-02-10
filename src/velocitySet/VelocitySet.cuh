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
    Common interface and aggregation header for LBM velocity set definitions

SourceFiles
    VelocitySet.cuh

\*---------------------------------------------------------------------------*/

#ifndef VELOCITYSET_CUH
#define VELOCITYSET_CUH

#include "cuda/utils.cuh"

namespace lbm
{
    class VelocitySet
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval VelocitySet() noexcept {};
    };
}

#include "D3Q7.cuh"
#include "D3Q19.cuh"
#include "D3Q27.cuh"

#endif
