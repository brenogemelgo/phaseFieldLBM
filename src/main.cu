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
    Main program file

SourceFiles
    main.cu

\*---------------------------------------------------------------------------*/

#include "functions/deviceFunctions.cuh"
#include "fieldAllocate/baseFieldsOwner.cuh"
#include "functions/hostFunctions.cuh"
#include "functions/ioFields.cuh"
#include "functions/vtsWriter.cuh"
#include "functions/vtiWriter.cuh"
#include "cuda/CUDAGraph.cuh"
#include "initialConditions.cu"
#include "boundaryConditions.cuh"
#include "phaseField.cuh"
#include "derivedFields/registry.cuh"
#include "derivedFields/manager.cuh"
#include "lbm.cu"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Error: Usage: " << argv[0] << " <flow case> <velocity set> <ID>\n";

        return 1;
    }

    const std::string FLOW_CASE = argv[1];
    const std::string VELOCITY_SET = argv[2];
    const std::string SIM_ID = argv[3];
    const std::string SIM_DIR = host::createSimulationDirectory(FLOW_CASE, VELOCITY_SET, SIM_ID);

    // Get device from pipeline argument
    if (host::setDeviceFromEnv() < 0)
    {
        return 1;
    }

    // Size variables
    constexpr size_t S_BYTES = host::bytesScalarGrid3D();
    constexpr size_t F_BYTES = host::bytesFDistros();
    constexpr size_t G_BYTES = host::bytesGDistros();

    // Device fields
    static constexpr std::array<host::FieldDescription<scalar_t>, 18> BASE_SCALAR_GRID = {{
        {"rho", &LBMFields::rho, S_BYTES, true},
        {"ux", &LBMFields::ux, S_BYTES, true},
        {"uy", &LBMFields::uy, S_BYTES, true},
        {"uz", &LBMFields::uz, S_BYTES, true},
        {"pxx", &LBMFields::pxx, S_BYTES, true},
        {"pyy", &LBMFields::pyy, S_BYTES, true},
        {"pzz", &LBMFields::pzz, S_BYTES, true},
        {"pxy", &LBMFields::pxy, S_BYTES, true},
        {"pxz", &LBMFields::pxz, S_BYTES, true},
        {"pyz", &LBMFields::pyz, S_BYTES, true},
        {"phi", &LBMFields::phi, S_BYTES, true},
        {"normx", &LBMFields::normx, S_BYTES, true},
        {"normy", &LBMFields::normy, S_BYTES, true},
        {"normz", &LBMFields::normz, S_BYTES, true},
        {"ind", &LBMFields::ind, S_BYTES, true},
        {"ffx", &LBMFields::ffx, S_BYTES, true},
        {"ffy", &LBMFields::ffy, S_BYTES, true},
        {"ffz", &LBMFields::ffz, S_BYTES, true},
    }};

    static constexpr host::FieldDescription<pop_t> F_DIST = {"f", &LBMFields::f, F_BYTES, true};
    static constexpr host::FieldDescription<scalar_t> G_DIST = {"g", &LBMFields::g, G_BYTES, true};

    // Allocate device fields
    host::BaseFieldsOwner baseOwner(fields, BASE_SCALAR_GRID, F_DIST, G_DIST);

    // Construct derived fields manager
    Derived::Manager derived;
    derived.allocate(fields);

    // Block-wise configuration
    constexpr dim3 block3D(block::nx, block::ny, block::nz);
    constexpr dim3 grid3D(host::divUp(mesh::nx, block3D.x),
                          host::divUp(mesh::ny, block3D.y),
                          host::divUp(mesh::nz, block3D.z));

    // Periodic x-direction
    constexpr dim3 blockX(block::ny, block::nz, 1u);
    constexpr dim3 gridX(host::divUp(mesh::ny, blockX.x), host::divUp(mesh::nz, blockX.y), 1u);

    // Periodic y-direction
    constexpr dim3 blockY(block::nx, block::nz, 1u);
    constexpr dim3 gridY(host::divUp(mesh::nx, blockY.x), host::divUp(mesh::nz, blockY.y), 1u);

    // Inlet and outlet
    constexpr dim3 blockZ(block::nx, block::ny, 1u);
    constexpr dim3 gridZ(host::divUp(mesh::nx, blockZ.x), host::divUp(mesh::ny, blockZ.y), 1u);

    // Dynamic shared memory size
    constexpr size_t dynamic = 0;

    // Stream setup
    cudaStream_t queue{};
    checkCudaErrorsOutline(cudaStreamCreate(&queue));

    // Initial conditions
    LBM::setInitialDensity<<<grid3D, block3D, dynamic, queue>>>(fields);
    LBM::FlowCase::initialConditions<grid3D, block3D, dynamic>(fields, queue);
    LBM::setDistros<<<grid3D, block3D, dynamic, queue>>>(fields);

    // Make sure everything is initialized
    checkCudaErrorsOutline(cudaDeviceSynchronize());

    // Generate info file and print diagnostics
    host::generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET);
    host::printDiagnostics(VELOCITY_SET);

#if !BENCHMARK

    // Initialize thread for asynchronous VTS generation
    std::vector<std::thread> vtk_threads;
    vtk_threads.reserve(NSTEPS / MACRO_SAVE + 2);

    // Base fields (always saved)
    constexpr std::array<host::FieldConfig, 2> BASE_FIELDS{
        {{host::FieldID::Phi, "phi", host::FieldDumpShape::Grid3D, true},
         {host::FieldID::Rho, "rho", host::FieldDumpShape::Grid3D, true}}};

    // Derived fields from modules (possibly empty)
    const auto DERIVED_FIELDS = Derived::makeOutputFields();

    // Compose final list in a vector
    std::vector<host::FieldConfig> OUTPUT_FIELDS;
    OUTPUT_FIELDS.reserve(BASE_FIELDS.size() + DERIVED_FIELDS.size());
    OUTPUT_FIELDS.insert(OUTPUT_FIELDS.end(), BASE_FIELDS.begin(), BASE_FIELDS.end());
    OUTPUT_FIELDS.insert(OUTPUT_FIELDS.end(), DERIVED_FIELDS.begin(), DERIVED_FIELDS.end());

    // Ensure post-processing only targets full 3D fields
    for (auto &cfg : OUTPUT_FIELDS)
    {
        cfg.includeInPost = (cfg.shape == host::FieldDumpShape::Grid3D);
    }

#endif

    // Warmup (optional)
    checkCudaErrorsOutline(cudaDeviceSynchronize());

    // Build CUDA Graph
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};
    graph::captureGraph<grid3D, block3D, dynamic>(graph, graphExec, fields, queue);

    // Start clock
    const auto START_TIME = std::chrono::high_resolution_clock::now();

    // Time loop
    for (label_t STEP = 0; STEP <= NSTEPS; ++STEP)
    {
        // Launch captured sequence
        cudaGraphLaunch(graphExec, queue);

        // Flow case specific boundary conditions
        LBM::FlowCase::boundaryConditions<gridX, blockX, gridY, blockY, gridZ, blockZ, dynamic>(fields, queue, STEP);

        // Ensure debug output is complete before host logic
        cudaStreamSynchronize(queue);

        // Derived fields
#if TIME_AVERAGE || REYNOLDS_MOMENTS || VORTICITY_FIELDS || PASSIVE_SCALAR
        Derived::launchAllDerived<grid3D, block3D, dynamic>(queue, fields, STEP);
#endif

#if !BENCHMARK

        const bool isOutputStep = (STEP % MACRO_SAVE == 0) || (STEP == NSTEPS);

        if (isOutputStep)
        {
            checkCudaErrors(cudaStreamSynchronize(queue));

            const auto step_copy = STEP;

            host::saveConfiguredFields(OUTPUT_FIELDS, SIM_DIR, step_copy, fields);

            vtk_threads.emplace_back(
                [step_copy,
                 fieldsCfg = OUTPUT_FIELDS,
                 sim_dir = SIM_DIR]
                {
                    host::writeImageData(fieldsCfg, sim_dir, step_copy);
                });

            std::cout << "Step " << STEP << ": bins in " << SIM_DIR << "\n";
        }

#endif
    }

#if !BENCHMARK

    for (auto &t : vtk_threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

#endif

    // Make sure everything is done on the GPU
    cudaStreamSynchronize(queue);
    const auto END_TIME = std::chrono::high_resolution_clock::now();

    // Destroy CUDA Graph resources
    checkCudaErrorsOutline(cudaGraphExecDestroy(graphExec));
    checkCudaErrorsOutline(cudaGraphDestroy(graph));

    // Destroy stream
    checkCudaErrorsOutline(cudaStreamDestroy(queue));

    const std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;

    const double steps = static_cast<double>(NSTEPS + 1);
    const double total_lattice_updates = static_cast<double>(mesh::nx) * mesh::ny * mesh::nz * steps;
    const double MLUPS = total_lattice_updates / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " s\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n\n";

    getLastCudaErrorOutline("Final sync");

    return 0;
}
