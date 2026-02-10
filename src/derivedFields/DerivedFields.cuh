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
    Manages allocation, execution, and output configuration of optional derived fields computed from the primary LBM solution

Namespace
    derived

Source files
    DerivedFields.cuh

\*---------------------------------------------------------------------------*/

#ifndef DERIVEDFIELDS_CUH
#define DERIVEDFIELDS_CUH

#include "cuda/utils.cuh"
#include "functions/ioFields.cuh"
#include "fieldAllocate/FieldAllocate.cuh"

#include "operators/timeAverage.cuh"
#include "operators/reynoldsMoments.cuh"
#include "operators/vorticityFields.cuh"
#include "operators/passiveScalar.cuh"

namespace derived
{
    class DerivedFields
    {
    public:
        static inline constexpr std::array<const char *, 0> kEnabledNames = {{}};

        DerivedFields() = default;
        DerivedFields(const DerivedFields &) = delete;
        DerivedFields &operator=(const DerivedFields &) = delete;

        ~DerivedFields() noexcept
        {
            if (attached_)
            {
                free(*attached_);
            }
        }

        __host__ void allocate(LBMFields &d)
        {
            attached_ = &d;

            host::FieldAllocate A;
            A.resetByteCounter();

#if TIME_AVERAGE
            if (anyEnabled(average::fields))
            {
                if (enabledName("avg_phi"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_phi", &LBMFields::avg_phi, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_ux"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_ux", &LBMFields::avg_ux, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_uy"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uy", &LBMFields::avg_uy, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_uz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uz", &LBMFields::avg_uz, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
            }
#endif

#if REYNOLDS_MOMENTS
            if (anyEnabled(reynolds::fields))
            {
                if (enabledName("avg_uxux"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uxux", &LBMFields::avg_uxux, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_uyuy"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uyuy", &LBMFields::avg_uyuy, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_uzuz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uzuz", &LBMFields::avg_uzuz, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_uxuy"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uxuy", &LBMFields::avg_uxuy, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_uxuz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uxuz", &LBMFields::avg_uxuz, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("avg_uyuz"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"avg_uyuz", &LBMFields::avg_uyuz, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
            }
#endif

#if VORTICITY_FIELDS
            if (anyEnabled(vorticity::fields))
            {
                if (enabledName("vort_x"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_x", &LBMFields::vort_x, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("vort_y"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_y", &LBMFields::vort_y, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("vort_z"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_z", &LBMFields::vort_z, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
                if (enabledName("vort_mag"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"vort_mag", &LBMFields::vort_mag, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
            }
#endif

#if PASSIVE_SCALAR
            if (anyEnabled(passive::fields))
            {
                if (enabledName("c"))
                {
                    A.alloc(d, host::FieldDescription<scalar_t>{"c", &LBMFields::c, static_cast<size_t>(size::cells()) * sizeof(scalar_t), true});
                }
            }
#endif
        }

        template <dim3 grid, dim3 block, size_t dynamic>
        __host__ inline void launch(cudaStream_t queue, LBMFields d, const label_t step) const noexcept
        {
#if TIME_AVERAGE
            if (anyEnabled(average::fields))
            {
                average::launch<grid, block, dynamic>(queue, d, step);
            }
#endif
#if REYNOLDS_MOMENTS
            if (anyEnabled(reynolds::fields))
            {
                reynolds::launch<grid, block, dynamic>(queue, d, step);
            }
#endif
#if VORTICITY_FIELDS
            if (anyEnabled(vorticity::fields))
            {
                vorticity::launch<grid, block, dynamic>(queue, d);
            }
#endif
#if PASSIVE_SCALAR
            if (anyEnabled(passive::fields))
            {
                passive::launch<grid, block, dynamic>(queue, d);
            }
#endif
        }

        __host__ [[nodiscard]] inline std::vector<host::FieldConfig> makeOutputFields() const
        {
            std::vector<host::FieldConfig> out;

            out.reserve(
#if TIME_AVERAGE
                average::fields.size() +
#endif
#if REYNOLDS_MOMENTS
                reynolds::fields.size() +
#endif
#if VORTICITY_FIELDS
                vorticity::fields.size() +
#endif
#if PASSIVE_SCALAR
                passive::fields.size() +
#endif
                0u);

#if TIME_AVERAGE
            appendEnabled(out, average::fields);
#endif
#if REYNOLDS_MOMENTS
            appendEnabled(out, reynolds::fields);
#endif
#if VORTICITY_FIELDS
            appendEnabled(out, vorticity::fields);
#endif
#if PASSIVE_SCALAR
            appendEnabled(out, passive::fields);
#endif

            return out;
        }

        __host__ inline void free(LBMFields &d) noexcept
        {
#if TIME_AVERAGE
            if (d.avg_phi)
            {
                cudaFree(d.avg_phi);
                d.avg_phi = nullptr;
            }
            if (d.avg_ux)
            {
                cudaFree(d.avg_ux);
                d.avg_ux = nullptr;
            }
            if (d.avg_uy)
            {
                cudaFree(d.avg_uy);
                d.avg_uy = nullptr;
            }
            if (d.avg_uz)
            {
                cudaFree(d.avg_uz);
                d.avg_uz = nullptr;
            }
#endif

#if REYNOLDS_MOMENTS
            if (d.avg_uxux)
            {
                cudaFree(d.avg_uxux);
                d.avg_uxux = nullptr;
            }
            if (d.avg_uyuy)
            {
                cudaFree(d.avg_uyuy);
                d.avg_uyuy = nullptr;
            }
            if (d.avg_uzuz)
            {
                cudaFree(d.avg_uzuz);
                d.avg_uzuz = nullptr;
            }
            if (d.avg_uxuy)
            {
                cudaFree(d.avg_uxuy);
                d.avg_uxuy = nullptr;
            }
            if (d.avg_uxuz)
            {
                cudaFree(d.avg_uxuz);
                d.avg_uxuz = nullptr;
            }
            if (d.avg_uyuz)
            {
                cudaFree(d.avg_uyuz);
                d.avg_uyuz = nullptr;
            }
#endif

#if VORTICITY_FIELDS
            if (d.vort_x)
            {
                cudaFree(d.vort_x);
                d.vort_x = nullptr;
            }
            if (d.vort_y)
            {
                cudaFree(d.vort_y);
                d.vort_y = nullptr;
            }
            if (d.vort_z)
            {
                cudaFree(d.vort_z);
                d.vort_z = nullptr;
            }
            if (d.vort_mag)
            {
                cudaFree(d.vort_mag);
                d.vort_mag = nullptr;
            }
#endif

#if PASSIVE_SCALAR
            if (d.c)
            {
                cudaFree(d.c);
                d.c = nullptr;
            }
#endif

            if (attached_ == &d)
            {
                attached_ = nullptr;
            }
        }

    private:
        LBMFields *attached_ = nullptr;

        __host__ static inline constexpr bool streq(const char *a, const char *b) noexcept
        {
            while (*a && (*a == *b))
            {
                ++a;
                ++b;
            }
            return (*a == *b);
        }

        __host__ static inline constexpr bool enabledName(const char *name) noexcept
        {
            if constexpr (kEnabledNames.size() == 0)
            {
                (void)name;
                return true;
            }
            else
            {
                for (const char *s : kEnabledNames)
                {
                    if (streq(s, name))
                    {
                        return true;
                    }
                }
                return false;
            }
        }

        template <typename FieldArray>
        __host__ static inline constexpr bool anyEnabled(const FieldArray &arr) noexcept
        {
            if constexpr (kEnabledNames.size() == 0)
            {
                (void)arr;
                return true;
            }
            else
            {
                for (const auto &cfg : arr)
                {
                    if (enabledName(cfg.name))
                    {
                        return true;
                    }
                }
                return false;
            }
        }

        template <size_t N>
        __host__ static inline void appendEnabled(std::vector<host::FieldConfig> &out,
                                                  const std::array<host::FieldConfig, N> &arr)
        {
            for (const auto &cfg : arr)
            {
                if (enabledName(cfg.name))
                {
                    out.push_back(cfg);
                }
            }
        }
    };
}

#endif
