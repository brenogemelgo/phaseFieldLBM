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
    Unified device field container for primary, phase-field, and optional derived LBM quantities

SourceFiles
    LBMFields.cuh

\*---------------------------------------------------------------------------*/

#ifndef STRUCTS_CUH
#define STRUCTS_CUH

struct LBMFields
{
    scalar_t *rho;
    scalar_t *ux;
    scalar_t *uy;
    scalar_t *uz;
    scalar_t *pxx;
    scalar_t *pyy;
    scalar_t *pzz;
    scalar_t *pxy;
    scalar_t *pxz;
    scalar_t *pyz;

    scalar_t *phi;
    scalar_t *normx;
    scalar_t *normy;
    scalar_t *normz;
    scalar_t *ind;
    scalar_t *Fx;
    scalar_t *Fy;
    scalar_t *Fz;

    pop_t *f;
    scalar_t *g;

#if TIME_AVERAGE

    scalar_t *avg_phi; // phi time average
    scalar_t *avg_ux;  // x velocity time average
    scalar_t *avg_uy;  // y velocity time average
    scalar_t *avg_uz;  // z velocity time average

#endif

#if REYNOLDS_MOMENTS

    scalar_t *avg_uxux; // xx
    scalar_t *avg_uyuy; // yy
    scalar_t *avg_uzuz; // zz
    scalar_t *avg_uxuy; // xy
    scalar_t *avg_uxuz; // xz
    scalar_t *avg_uyuz; // yz

#endif

#if VORTICITY_FIELDS

    scalar_t *vort_x;
    scalar_t *vort_y;
    scalar_t *vort_z;
    scalar_t *vort_mag;

#endif

#if PASSIVE_SCALAR

    scalar_t *c;

#endif
};

LBMFields fields{};

#endif