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
   Host-side I/O, diagnostics, and utility routines for simulation setup and data output

Namespace
    host

SourceFiles
    hostFunctions.cuh

\*---------------------------------------------------------------------------*/

#ifndef HOSTFUNCTIONS_CUH
#define HOSTFUNCTIONS_CUH

#include "globalFunctions.cuh"

namespace host
{
    __host__ [[nodiscard]] static inline std::string createSimulationDirectory(
        const std::string &FLOW_CASE,
        const std::string &VELOCITY_SET,
        const std::string &SIM_ID)
    {
        std::filesystem::path BASE_DIR = std::filesystem::current_path();

        std::filesystem::path SIM_DIR = BASE_DIR / "bin" / FLOW_CASE / VELOCITY_SET / SIM_ID;

        std::error_code EC;
        std::filesystem::create_directories(SIM_DIR, EC);

        return SIM_DIR.string() + std::string(1, std::filesystem::path::preferred_separator);
    }

    __host__ [[gnu::cold]] static inline void generateSimulationInfoFile(
        const std::string &SIM_DIR,
        const std::string &SIM_ID,
        const std::string &VELOCITY_SET)
    {
        std::filesystem::path INFO_PATH = std::filesystem::path(SIM_DIR) / (SIM_ID + "_info.txt");

        try
        {
            std::ofstream file(INFO_PATH, std::ios::out | std::ios::trunc);
            if (!file.is_open())
            {
                std::cerr << "Error opening file: " << INFO_PATH.string() << std::endl;

                return;
            }

            file << "---------------------------- SIMULATION METADATA ----------------------------\n"
                 << "ID:                    " << SIM_ID << '\n'
                 << "Velocity set:          " << VELOCITY_SET << '\n'
                 << "Reference velocity:    " << physics::u_inf << '\n'
                 << "Dispersed phase Re:    " << physics::reynolds_oil << '\n'
                 << "Continuous phase Re:   " << physics::reynolds_water << '\n'
                 << "Weber number:          " << physics::weber << '\n'
                 << "Surface tension:       " << physics::sigma << '\n'
                 << "NX:                    " << mesh::nx << '\n'
                 << "NY:                    " << mesh::ny << '\n'
                 << "NZ:                    " << mesh::nz << '\n'
                 << "Diameter:              " << mesh::diam << '\n'
                 << "Timesteps:             " << NSTEPS << '\n'
                 << "Output interval:       " << MACRO_SAVE << '\n'
                 << "-----------------------------------------------------------------------------\n";

            file.close();

            std::cout << "Simulation information file created in: " << INFO_PATH.string() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error generating information file: " << e.what() << std::endl;
        }
    }

    __host__ [[gnu::cold]] static inline void printDiagnostics(const std::string &VELOCITY_SET) noexcept
    {
        const double Ma = static_cast<double>(physics::u_inf) * static_cast<double>(lbm::velocitySet::as2());
        const double nu_d = static_cast<double>(physics::u_inf) * static_cast<double>(mesh::diam) / static_cast<double>(physics::reynolds_oil);
        const double nu_c = static_cast<double>(physics::u_inf) * static_cast<double>(mesh::diam) / static_cast<double>(physics::reynolds_water);
        const double tau_d = static_cast<double>(0.5) + static_cast<double>(nu_d) * static_cast<double>(lbm::velocitySet::as2());
        const double tau_c = static_cast<double>(0.5) + static_cast<double>(nu_c) * static_cast<double>(lbm::velocitySet::as2());

        std::cout << "\n---------------------------- SIMULATION METADATA ----------------------------\n"
                  << "Velocity set:          " << VELOCITY_SET << '\n'
                  << "Reference velocity:    " << physics::u_inf << '\n'
                  << "Dispersed phase Re:    " << physics::reynolds_oil << '\n'
                  << "Continuous phase Re:   " << physics::reynolds_water << '\n'
                  << "Weber number:          " << physics::weber << '\n'
                  << "Surface tension:       " << physics::sigma << '\n'
                  << "NX:                    " << mesh::nx << '\n'
                  << "NY:                    " << mesh::ny << '\n'
                  << "NZ:                    " << mesh::nz << '\n'
                  << "Diameter:              " << mesh::diam << '\n'
                  << "Total domain size:     " << size::cells() << '\n'
                  << "Mach:                  " << Ma << '\n'
                  << "Dispersed phase tau:   " << tau_d << '\n'
                  << "Continuous phase tau:  " << tau_c << '\n'
                  << "-----------------------------------------------------------------------------\n\n";
    }

    __host__ [[nodiscard]] [[gnu::cold]] static inline int setDeviceFromEnv() noexcept
    {
        int dev = 0;
        if (const char *env = std::getenv("GPU_INDEX"))
        {
            char *end = nullptr;
            long v = std::strtol(env, &end, 10);

            if (end != env && v >= 0)
            {
                dev = static_cast<int>(v);
            }
        }

        cudaError_t err = cudaSetDevice(dev);
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));

            return -1;
        }

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        {
            std::printf("Using GPU %d: %s (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);
        }
        else
        {
            std::printf("Using GPU %d (unknown properties)\n", dev);
        }

        return dev;
    }

    __host__ [[nodiscard]] static inline constexpr unsigned divUp(
        const unsigned a,
        const unsigned b) noexcept
    {
        return (a + b - 1u) / b;
    }

    __host__ [[nodiscard]] static inline constexpr size_t bytesScalar() noexcept
    {
        return static_cast<size_t>(size::cells()) * sizeof(scalar_t);
    }

    __host__ [[nodiscard]] static inline constexpr size_t bytesF() noexcept
    {
        return static_cast<size_t>(size::cells()) * static_cast<size_t>(lbm::velocitySet::Q()) * sizeof(pop_t);
    }

    __host__ [[nodiscard]] static inline constexpr size_t bytesG() noexcept
    {
        return static_cast<size_t>(size::cells()) * static_cast<size_t>(phase::velocitySet::Q()) * sizeof(scalar_t);
    }
}

#endif