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
    Droplet flow case definition with initialization and boundary hooks

Namespace
    LBM

SourceFiles
    droplet.cuh

\*---------------------------------------------------------------------------*/

#ifndef DROPLET_CUH
#define DROPLET_CUH

#include "flowCase.cuh"

namespace LBM
{
    class droplet : private flowCase
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval droplet(){};

        __device__ __host__ [[nodiscard]] static inline consteval bool droplet_case() noexcept
        {
            return true;
        }

        __device__ __host__ [[nodiscard]] static inline consteval bool jet_case() noexcept
        {
            return false;
        }

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void initialConditions(
            const LBMFields &fields,
            const cudaStream_t queue)
        {
            setDroplet<<<grid, block, dynamic, queue>>>(fields);
        }

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void boundaryConditions(
            const LBMFields &fields,
            const cudaStream_t queue,
            const label_t STEP)
        {
            // Full periodicity with wrap in streaming
        }

    private:
        // No private methods
    };
}

#endif