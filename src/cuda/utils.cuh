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
    CUDA utility functions and definitions

Namespace
    block

SourceFiles
    utils.cuh

\*---------------------------------------------------------------------------*/

#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <math_constants.h>
#include <builtin_types.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <unordered_map>
#include <array>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <math.h>
#include <stdlib.h>

namespace block
{
    static constexpr unsigned nx = 32;
    static constexpr unsigned ny = 4;
    static constexpr unsigned nz = 4;

    static constexpr int pad = 1;
    static constexpr int tile_nx = static_cast<int>(nx) + 2 * pad;
    static constexpr int tile_ny = static_cast<int>(ny) + 2 * pad;
    static constexpr int tile_nz = static_cast<int>(nz) + 2 * pad;
}

using label_t = uint64_t;
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

#define checkCudaErrors(err) __checkCudaErrors((err), #err, __FILE__, __LINE__)
#define checkCudaErrorsOutline(err) __checkCudaErrorsOutline((err), #err, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError((msg), __FILE__, __LINE__)
#define getLastCudaErrorOutline(msg) __getLastCudaErrorOutline((msg), __FILE__, __LINE__)

__host__ static void __checkCudaErrorsOutline(
    cudaError_t err,
    const char *const func,
    const char *const file,
    const int line) noexcept
{
    if (err != cudaSuccess)
    {
        fprintf(
            stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n", file, line, func, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        std::abort();
    }
}

__host__ static void __getLastCudaErrorOutline(
    const char *const errorMessage,
    const char *const file,
    const int line) noexcept
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s. Context: %s\n",
                file, line, (int)err, cudaGetErrorString(err), errorMessage);
        fflush(stderr);
        std::abort();
    }
}

__host__ static inline void __checkCudaErrors(
    cudaError_t err,
    const char *const func,
    const char *const file,
    const int line) noexcept
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        std::abort();
    }
}

__host__ static inline void __getLastCudaError(
    const char *const errorMessage,
    const char *const file,
    const int line) noexcept
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s. Context: %s\n",
                file, line, (int)err, cudaGetErrorString(err), errorMessage);
        fflush(stderr);
        std::abort();
    }
}

#endif