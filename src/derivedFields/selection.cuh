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
    Compile-time selection list for derived fields

Namespace
    Derived
    Selection

\*---------------------------------------------------------------------------*/

#ifndef SELECTION_CUH
#define SELECTION_CUH

#include "cuda/utils.cuh"

namespace Derived
{
    namespace Selection
    {

        /**
         * @brief Derived fields selector
         **/
        inline constexpr std::array<const char *, 1> kEnabledNames = {{
            // "avg_phi",
            // "avg_uz",
            "avg_uxux",
            // "vort_mag",
        }};

        __host__ [[nodiscard]] static inline bool enabledName(const char *name) noexcept
        {
            if constexpr (kEnabledNames.size() == 0)
            {
                return true;
            }

            if (name == nullptr)
            {
                return false;
            }

            for (auto s : kEnabledNames)
            {
                if (s != nullptr && std::strcmp(s, name) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        template <typename FieldArray>
        __host__ [[nodiscard]] static inline bool anyEnabled(const FieldArray &arr) noexcept
        {
            if constexpr (kEnabledNames.size() == 0)
            {
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
    }
}

#endif
