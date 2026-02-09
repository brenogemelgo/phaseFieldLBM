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
    Field configuration and save helpers (binary output)

Namespace
    host

SourceFiles
    ioFields.cuh

\*---------------------------------------------------------------------------*/

#ifndef IOFIELDS_CUH
#define IOFIELDS_CUH

#include "globalFunctions.cuh"

namespace host
{
    enum class FieldID : std::uint8_t
    {
        Rho,
        Ux,
        Uy,
        Uz,
        Pxx,
        Pyy,
        Pzz,
        Pxy,
        Pxz,
        Pyz,
        Phi,
        NormX,
        NormY,
        NormZ,
        Ind,
        Ffx,
        Ffy,
        Ffz,
        Avg_phi,
        // Avg_ux,
        // Avg_uy,
        Avg_uz,
        Avg_uxux,
        Avg_uyuy,
        Avg_uzuz,
        Avg_uxuy,
        Avg_uxuz,
        Avg_uyuz,
        Vort_x,
        Vort_y,
        Vort_z,
        Vort_mag,
        C,
    };

    enum class FieldDumpShape : std::uint8_t
    {
        Grid3D,
        Plane2D
    };

    struct FieldConfig
    {
        FieldID id;
        const char *name;
        FieldDumpShape shape;
        bool includeInPost;
    };

    __host__ [[nodiscard]] static inline scalar_t *getDeviceFieldPointer(const FieldID id) noexcept
    {
        switch (id)
        {
        case FieldID::Rho:
            return fields.rho;
        case FieldID::Ux:
            return fields.ux;
        case FieldID::Uy:
            return fields.uy;
        case FieldID::Uz:
            return fields.uz;
        case FieldID::Pxx:
            return fields.pxx;
        case FieldID::Pyy:
            return fields.pyy;
        case FieldID::Pzz:
            return fields.pzz;
        case FieldID::Pxy:
            return fields.pxy;
        case FieldID::Pxz:
            return fields.pxz;
        case FieldID::Pyz:
            return fields.pyz;
        case FieldID::Phi:
            return fields.phi;
        case FieldID::NormX:
            return fields.normx;
        case FieldID::NormY:
            return fields.normy;
        case FieldID::NormZ:
            return fields.normz;
        case FieldID::Ind:
            return fields.ind;
        case FieldID::Ffx:
            return fields.ffx;
        case FieldID::Ffy:
            return fields.ffy;
        case FieldID::Ffz:
            return fields.ffz;

#if TIME_AVERAGE

        case FieldID::Avg_phi:
            return fields.avg_phi;
        // case FieldID::Avg_ux:
        //     return fields.avg_ux;
        // case FieldID::Avg_uy:
        //     return fields.avg_uy;
        case FieldID::Avg_uz:
            return fields.avg_uz;

#endif

#if REYNOLDS_MOMENTS

        case FieldID::Avg_uxux:
            return fields.avg_uxux;
        case FieldID::Avg_uyuy:
            return fields.avg_uyuy;
        case FieldID::Avg_uzuz:
            return fields.avg_uzuz;
        case FieldID::Avg_uxuy:
            return fields.avg_uxuy;
        case FieldID::Avg_uxuz:
            return fields.avg_uxuz;
        case FieldID::Avg_uyuz:
            return fields.avg_uyuz;

#endif

#if VORTICITY_FIELDS

        case FieldID::Vort_x:
            return fields.vort_x;
        case FieldID::Vort_y:
            return fields.vort_y;
        case FieldID::Vort_z:
            return fields.vort_z;
        case FieldID::Vort_mag:
            return fields.vort_mag;

#endif

#if PASSIVE_SCALAR

        case FieldID::C:
            return fields.c;

#endif

        default:
            return nullptr;
        }
    }

    template <typename Container>
    __host__ static inline void saveConfiguredFields(
        const Container &fieldsCfg,
        const std::string &SIM_DIR,
        const label_t STEP)
    {
        for (const auto &cfg : fieldsCfg)
        {
            scalar_t *d_ptr = getDeviceFieldPointer(cfg.id);
            if (d_ptr == nullptr)
            {
                std::cerr << "saveConfiguredFields: null pointer for field " << cfg.name << ", skipping.\n";

                continue;
            }

            switch (cfg.shape)
            {
            case FieldDumpShape::Grid3D:
                host::copyAndSaveToBinary<size::cells()>(d_ptr, SIM_DIR, STEP, cfg.name);

                break;

            case FieldDumpShape::Plane2D:
                host::copyAndSaveToBinary<size::stride()>(d_ptr, SIM_DIR, STEP, cfg.name);

                break;
            }
        }
    }
}

#endif