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
    Centralized kernel declarations for LBM initialization, core routines, and boundary conditions

Namespace
    lbm

SourceFiles
    LBMIncludes.cuh

\*---------------------------------------------------------------------------*/

#ifndef LBMINCLUDES_CUH
#define LBMINCLUDES_CUH

namespace lbm
{
    // Initial conditions
    __global__ void setInitialDensity(LBMFields d);
    __global__ void setDroplet(LBMFields d);
    __global__ void setJet(LBMFields d);
    __global__ void setDistros(LBMFields d);

    // Moments and core routines
    __global__ void computeMoments(LBMFields d);
    __global__ void streamCollide(LBMFields d);

    // Boundary conditions
    __global__ void callInflow(LBMFields d, const label_t t);
    __global__ void callOutflow(LBMFields d);
    __global__ void callPeriodicX(LBMFields d);
    __global__ void callPeriodicY(LBMFields d);
}

#endif