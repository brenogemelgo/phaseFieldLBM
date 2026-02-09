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
    Header file for the velocity set classes

SourceFiles
    velocitySet.cuh

\*---------------------------------------------------------------------------*/

#ifndef VELOCITYSET_CUH
#define VELOCITYSET_CUH

#include "cuda/utils.cuh"

namespace LBM
{
    class velocitySet
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval velocitySet() noexcept {};
    };
}

#include "D3Q7.cuh"
#include "D3Q19.cuh"
#include "D3Q27.cuh"

#endif
