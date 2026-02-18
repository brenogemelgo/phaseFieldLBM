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
    Descriptor-driven allocator for CUDA device fields with null-safe free

Namespace
    host

SourceFiles
    FieldAllocate.cuh

\*---------------------------------------------------------------------------*/

#ifndef FIELDALLOCATOR_CUH
#define FIELDALLOCATOR_CUH

#include "LBMIncludes.cuh"

namespace host
{
    template <typename T>
    struct FieldDescription
    {
        const char *name = nullptr;
        T *LBMFields::*member = nullptr;
        size_t bytes = 0;
        bool zero = false;
    };

    class FieldAllocate
    {
    public:
        FieldAllocate() = default;

        template <size_t NScalar>
        __host__ FieldAllocate(
            LBMFields &f,
            const std::array<FieldDescription<scalar_t>, NScalar> &scalarGrid,
            const FieldDescription<pop_t> &fDist,
            const FieldDescription<scalar_t> &gDist)
            : owned_(&f),
              owning_(true),
              scalarOwned_(scalarGrid.begin(), scalarGrid.end()),
              fOwned_(fDist),
              gOwned_(gDist)
        {
            resetByteCounter();

            alloc_many(*owned_, scalarGrid);
            alloc(*owned_, fOwned_);
            alloc(*owned_, gOwned_);

            getLastCudaErrorOutline("FieldAllocate: own+alloc");
        }

        FieldAllocate(const FieldAllocate &) = delete;
        FieldAllocate &operator=(const FieldAllocate &) = delete;

        FieldAllocate(FieldAllocate &&) = delete;
        FieldAllocate &operator=(FieldAllocate &&) = delete;

        __host__ ~FieldAllocate() noexcept
        {
            if (!owning_ || owned_ == nullptr)
            {
                return;
            }

            free(*owned_, fOwned_);
            free(*owned_, gOwned_);

            for (const auto &d : scalarOwned_)
            {
                free(*owned_, d);
            }

            getLastCudaErrorOutline("FieldAllocate: own+free");
        }

        __host__ void release() noexcept
        {
            owning_ = false;
            owned_ = nullptr;
            scalarOwned_.clear();
        }

        __host__ void resetByteCounter() noexcept { bytes_allocated_ = 0; }
        __host__ [[nodiscard]] size_t bytesAllocated() const noexcept { return bytes_allocated_; }

        template <typename T>
        __host__ void alloc(LBMFields &f, const FieldDescription<T> &d)
        {
            using MemberT = std::remove_reference_t<decltype(f.*(d.member))>;
            static_assert(std::is_same_v<MemberT, T *>, "FieldDescription member type must be T*");

            T *&ptr = f.*(d.member);
            if (ptr != nullptr)
            {
                return;
            }

            checkCudaErrorsOutline(cudaMalloc(&ptr, d.bytes));

            if (d.zero && d.bytes > 0)
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

    private:
        size_t bytes_allocated_ = 0;

        LBMFields *owned_ = nullptr;
        bool owning_ = false;

        std::vector<FieldDescription<scalar_t>> scalarOwned_;
        FieldDescription<pop_t> fOwned_{};
        FieldDescription<scalar_t> gOwned_{};
    };

}

#endif
