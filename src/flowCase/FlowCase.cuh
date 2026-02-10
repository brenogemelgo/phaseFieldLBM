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
    Base interface for compile-time flow case definitions

Namespace
    LBM

SourceFiles
    FlowCase.cuh

\*---------------------------------------------------------------------------*/

#ifndef FLOWCASE_CUH
#define FLOWCASE_CUH

#include "cuda/utils.cuh"
#include "include/LBMIncludes.cuh"

namespace LBM
{
    class FlowCase
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval FlowCase() noexcept {};
    };
}

#include "Droplet.cuh"
#include "Jet.cuh"

#endif