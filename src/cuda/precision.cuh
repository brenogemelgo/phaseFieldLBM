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
    CUDA-level precision control

SourceFiles
    precision.cuh

\*---------------------------------------------------------------------------*/

#ifndef PRECISION_CUH
#define PRECISION_CUH

using label_t = uint32_t;
using scalar_t = float;

#if ENABLE_FP16

#include <cuda_fp16.h>
using pop_t = __half;

__device__ [[nodiscard]] inline pop_t to_pop(const scalar_t x) noexcept
{
    return __float2half(x);
}

__device__ [[nodiscard]] inline scalar_t from_pop(const pop_t x) noexcept
{
    return __half2float(x);
}

#else

using pop_t = scalar_t;

__device__ [[nodiscard]] inline constexpr pop_t to_pop(const scalar_t x) noexcept
{
    return x;
}

__device__ [[nodiscard]] inline constexpr scalar_t from_pop(const pop_t x) noexcept
{
    return x;
}

#endif

#endif