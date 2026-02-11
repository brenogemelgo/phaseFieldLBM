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
   Field configuration, device-pointer access, and binary output helpers

Namespace
    host

SourceFiles
    ioFields.cuh

\*---------------------------------------------------------------------------*/

#ifndef IOFIELDS_CUH
#define IOFIELDS_CUH

#include "functions/globalFunctions.cuh"

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
        Avg_ux,
        Avg_uy,
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

    __host__ [[nodiscard]] static inline const scalar_t *getDeviceFieldPointer(
        const LBMFields &f,
        const FieldID id) noexcept
    {
        switch (id)
        {
        case FieldID::Rho:
            return f.rho;
        case FieldID::Ux:
            return f.ux;
        case FieldID::Uy:
            return f.uy;
        case FieldID::Uz:
            return f.uz;
        case FieldID::Pxx:
            return f.pxx;
        case FieldID::Pyy:
            return f.pyy;
        case FieldID::Pzz:
            return f.pzz;
        case FieldID::Pxy:
            return f.pxy;
        case FieldID::Pxz:
            return f.pxz;
        case FieldID::Pyz:
            return f.pyz;
        case FieldID::Phi:
            return f.phi;
        case FieldID::NormX:
            return f.normx;
        case FieldID::NormY:
            return f.normy;
        case FieldID::NormZ:
            return f.normz;
        case FieldID::Ind:
            return f.ind;
        case FieldID::Ffx:
            return f.Fx;
        case FieldID::Ffy:
            return f.Fy;
        case FieldID::Ffz:
            return f.Fz;

#if TIME_AVERAGE

        case FieldID::Avg_phi:
            return f.avg_phi;
        case FieldID::Avg_ux:
            return f.avg_ux;
        case FieldID::Avg_uy:
            return f.avg_uy;
        case FieldID::Avg_uz:
            return f.avg_uz;

#endif

#if REYNOLDS_MOMENTS

        case FieldID::Avg_uxux:
            return f.avg_uxux;
        case FieldID::Avg_uyuy:
            return f.avg_uyuy;
        case FieldID::Avg_uzuz:
            return f.avg_uzuz;
        case FieldID::Avg_uxuy:
            return f.avg_uxuy;
        case FieldID::Avg_uxuz:
            return f.avg_uxuz;
        case FieldID::Avg_uyuz:
            return f.avg_uyuz;

#endif

#if VORTICITY_FIELDS

        case FieldID::Vort_x:
            return f.vort_x;
        case FieldID::Vort_y:
            return f.vort_y;
        case FieldID::Vort_z:
            return f.vort_z;
        case FieldID::Vort_mag:
            return f.vort_mag;

#endif

#if PASSIVE_SCALAR

        case FieldID::C:
            return f.c;

#endif

        default:
            return nullptr;
        }
    }
}

#endif