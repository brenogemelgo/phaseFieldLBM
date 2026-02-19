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
    Droplet flow case definition with initialization and boundary hooks

Namespace
    lbm

SourceFiles
    Droplet.cuh

\*---------------------------------------------------------------------------*/

#ifndef DROPLET_CUH
#define DROPLET_CUH

#include "FlowCase.cuh"

namespace lbm
{
    class Droplet : private FlowCase
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval Droplet(){};

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

        template <dim3 gridX, dim3 blockX, dim3 gridY, dim3 blockY, dim3 gridZ, dim3 blockZ, size_t dynamic>
        __host__ static inline void boundaryConditions(
            const LBMFields &fields,
            const cudaStream_t queue,
            const label_t t)
        {
            // Full periodicity with wrap in streaming
        }

    private:
        // No private methods
    };
}

#endif