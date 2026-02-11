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
    Base interface for compile-time flow case definitions

Namespace
    lbm

SourceFiles
    FlowCase.cuh

\*---------------------------------------------------------------------------*/

#ifndef FLOWCASE_CUH
#define FLOWCASE_CUH

#include "cuda/utils.cuh"
#include "LBMIncludes.cuh"

namespace lbm
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