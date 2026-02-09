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
#include "fieldAllocate/fieldAllocate.cuh"
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

            constexpr size_t S_BYTES = host::bytesScalarGrid3D();

            host::FieldAllocate A;
            A.resetByteCounter();

#if TIME_AVERAGE
            if (Selection::anyEnabled(TimeAverage::fields))
            {
                if (Selection::enabledName("avg_phi"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_phi", &LBMFields::avg_phi, S_BYTES, true});
                }
                if (Selection::enabledName("avg_ux"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_ux", &LBMFields::avg_ux, S_BYTES, true});
                }
                if (Selection::enabledName("avg_uy"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uy", &LBMFields::avg_uy, S_BYTES, true});
                }
                if (Selection::enabledName("avg_uz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uz", &LBMFields::avg_uz, S_BYTES, true});
                }
            }
#endif

#if REYNOLDS_MOMENTS
            if (Selection::anyEnabled(ReynoldsMoments::fields))
            {
                if (Selection::enabledName("avg_uxux"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uxux", &LBMFields::avg_uxux, S_BYTES, true});
                }
                if (Selection::enabledName("avg_uyuy"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uyuy", &LBMFields::avg_uyuy, S_BYTES, true});
                }
                if (Selection::enabledName("avg_uzuz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uzuz", &LBMFields::avg_uzuz, S_BYTES, true});
                }
                if (Selection::enabledName("avg_uxuy"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uxuy", &LBMFields::avg_uxuy, S_BYTES, true});
                }
                if (Selection::enabledName("avg_uxuz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uxuz", &LBMFields::avg_uxuz, S_BYTES, true});
                }
                if (Selection::enabledName("avg_uyuz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uyuz", &LBMFields::avg_uyuz, S_BYTES, true});
                }
            }
#endif

#if VORTICITY_FIELDS
            if (Selection::anyEnabled(VorticityFields::fields))
            {
                if (Selection::enabledName("vort_x"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_x", &LBMFields::vort_x, S_BYTES, true});
                }
                if (Selection::enabledName("vort_y"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_y", &LBMFields::vort_y, S_BYTES, true});
                }
                if (Selection::enabledName("vort_z"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_z", &LBMFields::vort_z, S_BYTES, true});
                }
                if (Selection::enabledName("vort_mag"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_mag", &LBMFields::vort_mag, S_BYTES, true});
                }
            }
#endif

#if PASSIVE_SCALAR
            if (Selection::anyEnabled(PassiveScalar::fields))
            {
                if (Selection::enabledName("c"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"c", &LBMFields::c, S_BYTES, true});
                }
            }
#endif

            getLastCudaErrorOutline("Derived::Manager::allocate");
        }

    private:
        LBMFields *attached_ = nullptr;
    };
}

#endif
