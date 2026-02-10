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
    CUDA Graph capture of the core phase-field and hydrodynamic LBM kernel sequence

Namespace
    graph

SourceFiles
    CUDAGraph.cuh

\*---------------------------------------------------------------------------*/

#ifndef CUDAGRAPH_CUH
#define CUDAGRAPH_CUH

#include "phaseField.cuh"
#include "include/LBMIncludes.cuh"

namespace graph
{
    template <dim3 grid, dim3 block, size_t dynamic>
    __host__ inline void captureGraph(
        cudaGraph_t &graph,
        cudaGraphExec_t &graphExec,
        const LBMFields &fields,
        const cudaStream_t queue)
    {
        checkCudaErrorsOutline(cudaStreamBeginCapture(queue, cudaStreamCaptureModeGlobal));

        // Phase field
        Phase::computePhase<<<grid, block, dynamic, queue>>>(fields);
        Phase::computeNormals<<<grid, block, dynamic, queue>>>(fields);
        Phase::computeForces<<<grid, block, dynamic, queue>>>(fields);

        // Hydrodynamics
        lbm::computeMoments<<<grid, block, dynamic, queue>>>(fields);
        lbm::streamCollide<<<grid, block, dynamic, queue>>>(fields);

        // NOTE: We intentionally DO NOT include boundary conditions or
        // derived fields here, because they depend on STEP and/or other
        // time-varying parameters. We launch them after the graph each step.

        checkCudaErrorsOutline(cudaStreamEndCapture(queue, &graph));
        checkCudaErrorsOutline(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }
}

#endif
