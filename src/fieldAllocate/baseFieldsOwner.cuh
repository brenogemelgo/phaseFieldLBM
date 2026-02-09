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
    RAII owner responsible for allocation and lifetime of base LBM fields

Namespace
    host

SourceFiles
    baseFieldsOwner.cuh

\*---------------------------------------------------------------------------*/

#ifndef BASEFIELDSOWNER_CUH
#define BASEFIELDSOWNER_CUH

#include "cuda/utils.cuh"
#include "fieldAllocate/fieldAllocate.cuh"

namespace host
{
    template <size_t NScalar>
    class BaseFieldsOwner
    {
    public:
        BaseFieldsOwner(
            LBMFields &f,
            const std::array<host::FieldDescription<scalar_t>, NScalar> &scalarGrid,
            const host::FieldDescription<pop_t> &fDist,
            const host::FieldDescription<scalar_t> &gDist)
            : f_(&f),
              scalarGrid_(scalarGrid),
              fDist_(fDist),
              gDist_(gDist)
        {
            host::FieldAllocate A;
            A.resetByteCounter();

            A.alloc_many(*f_, scalarGrid_);
            A.alloc(*f_, fDist_);
            A.alloc(*f_, gDist_);

            getLastCudaErrorOutline("BaseFieldsOwner: allocate");
        }

        BaseFieldsOwner(const BaseFieldsOwner &) = delete;
        BaseFieldsOwner &operator=(const BaseFieldsOwner &) = delete;

        ~BaseFieldsOwner() noexcept
        {
            if (!f_)
            {
                return;
            }

            host::FieldAllocate A;

            A.free(*f_, fDist_);
            A.free(*f_, gDist_);
            A.free_many(*f_, scalarGrid_);

            getLastCudaErrorOutline("BaseFieldsOwner: free");
        }

    private:
        LBMFields *f_ = nullptr;

        const std::array<host::FieldDescription<scalar_t>, NScalar> &scalarGrid_;
        host::FieldDescription<pop_t> fDist_;
        host::FieldDescription<scalar_t> gDist_;
    };

    template <size_t NScalar>
    BaseFieldsOwner(
        LBMFields &,
        const std::array<host::FieldDescription<scalar_t>, NScalar> &,
        const host::FieldDescription<pop_t> &,
        const host::FieldDescription<scalar_t> &)
        -> BaseFieldsOwner<NScalar>;
}

#endif
