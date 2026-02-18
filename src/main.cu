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
    Main driver orchestrating initialization, CUDA Graph execution, time stepping, boundary handling, and post-processing output

SourceFiles
    main.cu

\*---------------------------------------------------------------------------*/

#include "functions/deviceFunctions.cuh"
#include "fieldAllocate/FieldAllocate.cuh"
#include "functions/hostFunctions.cuh"
#include "fileIO/fields.cuh"
#include "postProcess/PostProcess.cuh"
#include "cuda/CUDAGraph.cuh"
#include "initialConditions.cu"
#include "BoundaryConditions.cuh"
#include "phaseField.cuh"
#include "derivedFields/DerivedFields.cuh"
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

    // Device 3D fields
    static constexpr auto scalar = std::to_array<host::FieldDescription<scalar_t>>({
        {"rho", &LBMFields::rho, host::bytesScalar(), true},
        {"ux", &LBMFields::ux, host::bytesScalar(), true},
        {"uy", &LBMFields::uy, host::bytesScalar(), true},
        {"uz", &LBMFields::uz, host::bytesScalar(), true},
        {"pxx", &LBMFields::pxx, host::bytesScalar(), true},
        {"pyy", &LBMFields::pyy, host::bytesScalar(), true},
        {"pzz", &LBMFields::pzz, host::bytesScalar(), true},
        {"pxy", &LBMFields::pxy, host::bytesScalar(), true},
        {"pxz", &LBMFields::pxz, host::bytesScalar(), true},
        {"pyz", &LBMFields::pyz, host::bytesScalar(), true},
        {"phi", &LBMFields::phi, host::bytesScalar(), true},
        {"normx", &LBMFields::normx, host::bytesScalar(), true},
        {"normy", &LBMFields::normy, host::bytesScalar(), true},
        {"normz", &LBMFields::normz, host::bytesScalar(), true},
        {"ind", &LBMFields::ind, host::bytesScalar(), true},
        {"fsx", &LBMFields::fsx, host::bytesScalar(), true},
        {"fsy", &LBMFields::fsy, host::bytesScalar(), true},
        {"fsz", &LBMFields::fsz, host::bytesScalar(), true},
    });

    // Device distribution functions
    static constexpr host::FieldDescription<pop_t> f = {"f", &LBMFields::f, host::bytesF(), true};
    static constexpr host::FieldDescription<scalar_t> g = {"g", &LBMFields::g, host::bytesG(), true};

    // Allocate all device fields
    host::FieldAllocate baseOwner(fields, scalar, f, g);

    // Construct derived fields
    derived::DerivedFields dfields;
    dfields.allocate(fields);

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
    lbm::setInitialDensity<<<grid3D, block3D, dynamic, queue>>>(fields);
    lbm::flowCase::initialConditions<grid3D, block3D, dynamic>(fields, queue);
    lbm::setDistros<<<grid3D, block3D, dynamic, queue>>>(fields);

    // Make sure everything is initialized
    checkCudaErrorsOutline(cudaDeviceSynchronize());

    // Generate info file and print diagnostics
    host::generateSimulationInfoFile(SIM_DIR, SIM_ID, VELOCITY_SET);
    host::printDiagnostics(VELOCITY_SET);

#if !BENCHMARK
    // Post-processing instance
    host::PostProcess write;

    // Base fields (always saved)
    static constexpr auto BASE_FIELDS = std::to_array<host::FieldConfig>({
        {host::FieldID::Rho, "rho", host::FieldDumpShape::Grid3D, true},
        {host::FieldID::Uz, "uz", host::FieldDumpShape::Grid3D, true},
    });

    // Derived fields from modules (possibly empty)
    const auto DERIVED_FIELDS = dfields.makeOutputFields();

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
        lbm::flowCase::boundaryConditions<gridX, blockX, gridY, blockY, gridZ, blockZ, dynamic>(fields, queue, STEP);

        // Ensure debug output is complete before host logic
        cudaStreamSynchronize(queue);

        // Derived fields
        dfields.launch<grid3D, block3D, dynamic>(queue, fields, STEP);

#if !BENCHMARK
        const bool isOutputStep = (STEP % MACRO_SAVE == 0) || (STEP == NSTEPS);

        if (isOutputStep)
        {
            checkCudaErrors(cudaStreamSynchronize(queue));
            std::cout << "Step " << STEP << ": bins in " << SIM_DIR << "\n";
            write.bin(OUTPUT_FIELDS, SIM_DIR, STEP, fields);
            write.vti(OUTPUT_FIELDS, SIM_DIR, STEP);
        }
#endif
    }

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
