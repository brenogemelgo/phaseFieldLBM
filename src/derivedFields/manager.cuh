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
    Derived fields RAII manager (allocation + automatic free)

Namespace
    Derived

Source files
    manager.cuh

\*---------------------------------------------------------------------------*/

#ifndef MANAGER_CUH
#define MANAGER_CUH

#include "derivedFields/registry.cuh"
#include "cuda/utils.cuh"

namespace Derived
{
    class Manager
    {
    public:
        Manager() = default;
        Manager(const Manager &) = delete;
        Manager &operator=(const Manager &) = delete;

        ~Manager() noexcept
        {
            if (attached_)
            {
                Derived::freeAll(*attached_);
            }
        }

        __host__ void attach(LBMFields &d) noexcept
        {
            attached_ = &d;
        }

        __host__ void allocate(LBMFields &d)
        {
            attach(d);

            constexpr size_t NCELLS =
                static_cast<size_t>(mesh::nx) *
                static_cast<size_t>(mesh::ny) *
                static_cast<size_t>(mesh::nz);

            constexpr size_t SIZE = NCELLS * sizeof(scalar_t);

            static_assert(NCELLS > 0, "Empty grid?");
            static_assert(SIZE / sizeof(scalar_t) == NCELLS, "SIZE overflow");

#if TIME_AVERAGE
            if (Selection::anyEnabled(TimeAverage::fields))
            {
                if (Selection::enabledName("avg_phi"))
                {
                    mallocZero_(d.avg_phi, SIZE);
                }
                if (Selection::enabledName("avg_ux"))
                {
                    mallocZero_(d.avg_ux, SIZE);
                }
                if (Selection::enabledName("avg_uy"))
                {
                    mallocZero_(d.avg_uy, SIZE);
                }
                if (Selection::enabledName("avg_uz"))
                {
                    mallocZero_(d.avg_uz, SIZE);
                }
            }
#endif

#if REYNOLDS_MOMENTS
            if (Selection::anyEnabled(ReynoldsMoments::fields))
            {
                if (Selection::enabledName("avg_uxux"))
                {
                    mallocZero_(d.avg_uxux, SIZE);
                }
                if (Selection::enabledName("avg_uyuy"))
                {
                    mallocZero_(d.avg_uyuy, SIZE);
                }
                if (Selection::enabledName("avg_uzuz"))
                {
                    mallocZero_(d.avg_uzuz, SIZE);
                }
                if (Selection::enabledName("avg_uxuy"))
                {
                    mallocZero_(d.avg_uxuy, SIZE);
                }
                if (Selection::enabledName("avg_uxuz"))
                {
                    mallocZero_(d.avg_uxuz, SIZE);
                }
                if (Selection::enabledName("avg_uyuz"))
                {
                    mallocZero_(d.avg_uyuz, SIZE);
                }
            }
#endif

#if VORTICITY_FIELDS
            if (Selection::anyEnabled(VorticityFields::fields))
            {
                if (Selection::enabledName("vort_x"))
                {
                    mallocZero_(d.vort_x, SIZE);
                }
                if (Selection::enabledName("vort_y"))
                {
                    mallocZero_(d.vort_y, SIZE);
                }
                if (Selection::enabledName("vort_z"))
                {
                    mallocZero_(d.vort_z, SIZE);
                }
                if (Selection::enabledName("vort_mag"))
                {
                    mallocZero_(d.vort_mag, SIZE);
                }
            }
#endif

#if PASSIVE_SCALAR
            if (Selection::anyEnabled(PassiveScalar::fields))
            {
                mallocZero_(d.c, SIZE);
            }
#endif

            getLastCudaErrorOutline("Derived::Manager::allocate");
        }

    private:
        LBMFields *attached_ = nullptr;

        __host__ static inline void mallocZero_(scalar_t *&ptr, const size_t bytes)
        {
            if (ptr != nullptr)
            {
                return;
            }

            checkCudaErrorsOutline(cudaMalloc(&ptr, bytes));
            checkCudaErrorsOutline(cudaMemset(ptr, 0, bytes));
        }
    };
}

#endif
