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
    Passive scalar advection–diffusion kernel and derived field registration

Namespace
    lbm
    derived
    passive

SourceFiles
    passiveScalar.cuh

\*---------------------------------------------------------------------------*/

#ifndef PASSIVESCALAR_CUH
#define PASSIVESCALAR_CUH

#include "functions/ioFields.cuh"

#if PASSIVE_SCALAR

namespace lbm
{
    __global__ void advectDiffuse(LBMFields d)
    {
        printf("Passive scalar not implemented yet!\n");
        asm volatile("trap;");
    }
}

namespace derived
{
    namespace passive
    {
        constexpr std::array<host::FieldConfig, 1> fields{{
            {host::FieldID::C, "c", host::FieldDumpShape::Grid3D, true},
        }};

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ static inline void launch(
            cudaStream_t queue,
            LBMFields d) noexcept
        {
#if PASSIVE_SCALAR
            lbm::advectDiffuse<<<grid, block, dynamic, queue>>>(d);
#endif
        }

        __host__ static inline void free(LBMFields &d) noexcept
        {
#if PASSIVE_SCALAR
            if (d.c)
            {
                cudaFree(d.c);
                d.c = nullptr;
            }
#endif
        }
    }
}

#endif

#endif
