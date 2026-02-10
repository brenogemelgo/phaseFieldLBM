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
    Jet flow case definition with inflow, outflow, and periodic boundary conditions

Namespace
    LBM

SourceFiles
    Jet.cuh

\*---------------------------------------------------------------------------*/

#ifndef JET_CUH
#define JET_CUH

#include "FlowCase.cuh"

namespace LBM
{
    class Jet : private FlowCase
    {
    public:
        __device__ __host__ [[nodiscard]] inline consteval Jet(){};

        __device__ __host__ [[nodiscard]] static inline consteval bool droplet_case() noexcept
        {
            return false;
        }

        __device__ __host__ [[nodiscard]] static inline consteval bool jet_case() noexcept
        {
            return true;
        }

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void initialConditions(
            const LBMFields &fields,
            const cudaStream_t queue)
        {
            setJet<<<grid, block, dynamic, queue>>>(fields);
        }

        template <dim3 gridX, dim3 blockX, dim3 gridY, dim3 blockY, dim3 gridZ, dim3 blockZ, size_t dynamic>
        __host__ static inline void boundaryConditions(
            const LBMFields &fields,
            const cudaStream_t queue,
            const label_t STEP)
        {
            callInflow<<<gridZ, blockZ, dynamic, queue>>>(fields, STEP);
            callOutflow<<<gridZ, blockZ, dynamic, queue>>>(fields);
            callPeriodicX<<<gridX, blockX, dynamic, queue>>>(fields);
            callPeriodicY<<<gridY, blockY, dynamic, queue>>>(fields);
        }

    private:
        // No private methods
    };
}

#endif