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
    Jet flow case definition with inflow, outflow, and periodic boundary conditions

Namespace
    lbm

SourceFiles
    Jet.cuh

\*---------------------------------------------------------------------------*/

#ifndef JET_CUH
#define JET_CUH

#include "FlowCase.cuh"

namespace lbm
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
            const label_t t)
        {
            callInflow<<<gridZ, blockZ, dynamic, queue>>>(fields, t);
            callOutflow<<<gridZ, blockZ, dynamic, queue>>>(fields, t);
            callPeriodicX<<<gridX, blockX, dynamic, queue>>>(fields, t);
            callPeriodicY<<<gridY, blockY, dynamic, queue>>>(fields, t);
        }

    private:
        // No private methods
    };
}

#endif