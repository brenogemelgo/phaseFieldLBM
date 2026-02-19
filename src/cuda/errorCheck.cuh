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
    CUDA utilities for error handling

SourceFiles
    errorCheck.cuh

\*---------------------------------------------------------------------------*/

#ifndef ERRORCHECK_CUH
#define ERRORCHECK_CUH

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
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n", file, line, func, static_cast<int>(err), cudaGetErrorString(err));
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
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s. Context: %s\n", file, line, static_cast<int>(err), cudaGetErrorString(err), errorMessage);
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
        fprintf(stderr, "CUDA error at %s(%d) \"%s\": [%d] %s.\n", file, line, func, static_cast<int>(err), cudaGetErrorString(err));
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
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s. Context: %s\n", file, line, (int)err, cudaGetErrorString(err), errorMessage);
        fflush(stderr);
        std::abort();
    }
}

#endif