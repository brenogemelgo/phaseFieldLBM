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
    A class to handle field construction and destruction

Namespace
    host

SourceFiles
    fieldAllocator.cuh

\*---------------------------------------------------------------------------*/

#ifndef FIELDALLOCATOR_CUH
#define FIELDALLOCATOR_CUH

#include "cuda/utils.cuh"

namespace host
{
    template <typename T>
    struct FieldDescription
    {
        const char *name;
        T *LBMFields::*member;
        size_t bytes;
        bool zero;
    };

    class FieldAllocate
    {
    public:
        FieldAllocate() = default;
        FieldAllocate(const FieldAllocate &) = delete;
        FieldAllocate &operator=(const FieldAllocate &) = delete;

        template <typename T>
        __host__ void alloc(LBMFields &f, const FieldDescription<T> &d)
        {
            T *&ptr = f.*(d.member);
            if (ptr != nullptr)
            {
                return;
            }

            checkCudaErrorsOutline(cudaMalloc(&ptr, d.bytes));

            if (d.zero)
            {
                checkCudaErrorsOutline(cudaMemset(ptr, 0, d.bytes));
            }

            bytes_allocated_ += d.bytes;
        }

        template <typename T>
        __host__ void free(LBMFields &f, const FieldDescription<T> &d) noexcept
        {
            T *&ptr = f.*(d.member);
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }

        template <typename T, size_t N>
        __host__ void alloc_many(LBMFields &f, const std::array<FieldDescription<T>, N> &descs)
        {
            for (const auto &d : descs)
            {
                alloc(f, d);
            }
        }

        template <typename T, size_t N>
        __host__ void free_many(LBMFields &f, const std::array<FieldDescription<T>, N> &descs) noexcept
        {
            for (const auto &d : descs)
            {
                free(f, d);
            }
        }

        __host__ [[nodiscard]] size_t bytesAllocated() const noexcept { return bytes_allocated_; }
        __host__ void resetByteCounter() noexcept { bytes_allocated_ = 0; }

    private:
        size_t bytes_allocated_ = 0;
    };
}

#endif
