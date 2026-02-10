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
    Common post-processing types and VTK scalar metadata utilities

Namespace
    host

SourceFiles
    postCommon.cuh

\*---------------------------------------------------------------------------*/

#ifndef POSTCOMMON_CUH
#define POSTCOMMON_CUH

#include "functions/globalFunctions.cuh"

namespace host
{
    namespace detail
    {
        template <typename T>
        struct VTScalarTypeName;

        template <>
        struct VTScalarTypeName<float>
        {
            static constexpr const char *value() noexcept { return "Float32"; }
        };

        template <>
        struct VTScalarTypeName<double>
        {
            static constexpr const char *value() noexcept { return "Float64"; }
        };

        struct AppendedArray
        {
            std::string name;
            std::filesystem::path path;
            std::uint64_t nbytes;
            std::uint64_t offset;
            bool isPoints;
        };
    }
}

#endif
