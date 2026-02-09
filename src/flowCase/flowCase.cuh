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
    Header file for the flow case classes

Namespace
    LBM

SourceFiles
    flowCase.cuh

\*---------------------------------------------------------------------------*/

#ifndef FLOWCASE_CUH
#define FLOWCASE_CUH

#include "cuda/utils.cuh"
#include "include/LBMIncludes.cuh"

namespace LBM
{
    class flowCase
    {
    public:
        __host__ __device__ [[nodiscard]] inline consteval flowCase() noexcept {};
    };
}

#include "droplet.cuh"
#include "jet.cuh"

#endif