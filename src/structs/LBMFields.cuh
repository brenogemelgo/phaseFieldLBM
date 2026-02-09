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

License
    This file is part of MULTIC-TS-LBM.

    MULTIC-TS-LBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    LBMFields struct declaration

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
    scalar_t *ffx;
    scalar_t *ffy;
    scalar_t *ffz;

    pop_t *f;
    scalar_t *g;

#if TIME_AVERAGE

    scalar_t *avg_phi; // phi time average
    // scalar_t *avg_ux;  // x velocity time average
    // scalar_t *avg_uy;  // y velocity time average
    scalar_t *avg_uz; // z velocity time average

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