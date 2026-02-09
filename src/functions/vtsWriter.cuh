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
    VTS (StructuredGrid) writer for exporting scalar fields with explicit grid geometry

Namespace
    host

SourceFiles
    vtsWriter.cuh

\*---------------------------------------------------------------------------*/

#ifndef VTSWRITER_CUH
#define VTSWRITER_CUH

#include "include/postCommon.cuh"

namespace host
{
    template <typename Container>
    __host__ static inline void writeStructuredGrid(
        const Container &fieldsCfg,
        const std::string &SIM_DIR,
        const label_t STEP)
    {
        const uint64_t NX = static_cast<uint64_t>(mesh::nx);
        const uint64_t NY = static_cast<uint64_t>(mesh::ny);
        const uint64_t NZ = static_cast<uint64_t>(mesh::nz);
        const uint64_t NCELLS = NX * NY * NZ;
        const uint64_t cellBytes = NCELLS * static_cast<uint64_t>(sizeof(scalar_t));

        std::ostringstream stepStr;
        stepStr << std::setw(6) << std::setfill('0') << STEP;
        const std::string stepSuffix = stepStr.str();

        const std::string vtsFileName = "step_" + stepSuffix + ".vts";
        const std::filesystem::path vtsPath = std::filesystem::path(SIM_DIR) / vtsFileName;

        std::vector<detail::AppendedArray> arrays;
        arrays.reserve(fieldsCfg.size() + 1);

        uint64_t currentOffset = 0;
        constexpr uint64_t headerSize = sizeof(std::uint32_t);

        bool hasFieldArrays = false;

        try
        {
            for (const auto &cfg : fieldsCfg)
            {
                if (!cfg.includeInPost ||
                    cfg.shape != FieldDumpShape::Grid3D)
                {
                    continue;
                }

                const std::filesystem::path binPath =
                    std::filesystem::path(SIM_DIR) / (std::string(cfg.name) + stepSuffix + ".bin");

                std::error_code ec_fs;
                const uint64_t fs =
                    static_cast<uint64_t>(std::filesystem::file_size(binPath, ec_fs));
                if (ec_fs)
                {
                    std::cerr << "VTS writer: could not get size for " << binPath.string() << " (" << ec_fs.message() << "), skipping variable " << cfg.name << ".\n";

                    continue;
                }

                if (fs != cellBytes)
                {
                    std::cerr << "VTS writer: file " << binPath.string() << " has size " << fs << " bytes, expected " << cellBytes << " for full 3D field. " << "Skipping variable " << cfg.name << ".\n";

                    continue;
                }

                detail::AppendedArray arr;
                arr.name = cfg.name;
                arr.path = binPath;
                arr.nbytes = fs;
                arr.offset = currentOffset;
                arr.isPoints = false;

                arrays.push_back(arr);
                currentOffset += headerSize + arr.nbytes;
                hasFieldArrays = true;
            }

            if (!hasFieldArrays)
            {
                std::cerr << "VTS writer: no valid 3D fields to write for step " << STEP << ", aborting VTS.\n";

                return;
            }

            const uint64_t NPOINTS = NCELLS;

            detail::AppendedArray pointsArr;
            pointsArr.name = "Points";
            pointsArr.path.clear();
            pointsArr.nbytes = NPOINTS * 3u * sizeof(float);
            pointsArr.offset = currentOffset;
            pointsArr.isPoints = true;

            arrays.push_back(pointsArr);
            currentOffset += headerSize + pointsArr.nbytes;

            std::ofstream vts(vtsPath, std::ios::binary | std::ios::trunc);
            if (!vts)
            {
                std::cerr << "Error opening VTS file " << vtsPath.string() << " for writing.\n";

                return;
            }

            const char *scalarTypeName = detail::vtScalarTypeName<scalar_t>::value();

            vts << R"(<?xml version="1.0"?>)" << '\n';
            vts << R"(<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">)" << '\n';
            vts << "  <StructuredGrid WholeExtent=\"0 " << (mesh::nx - 1)
                << " 0 " << (mesh::ny - 1)
                << " 0 " << (mesh::nz - 1) << "\">\n";
            vts << "    <Piece Extent=\"0 " << (mesh::nx - 1)
                << " 0 " << (mesh::ny - 1)
                << " 0 " << (mesh::nz - 1) << "\">\n";

            vts << "      <PointData Scalars=\"" << arrays.front().name << "\">\n";
            for (const auto &arr : arrays)
            {
                if (arr.isPoints)
                {
                    continue;
                }

                vts << "        <DataArray type=\"" << scalarTypeName
                    << "\" Name=\"" << arr.name
                    << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
                    << arr.offset << "\"/>\n";
            }
            vts << "      </PointData>\n";

            vts << "      <CellData/>\n";

            const auto &pts = arrays.back();
            vts << "      <Points>\n";
            vts << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" "
                   "format=\"appended\" offset=\""
                << pts.offset << "\"/>\n";
            vts << "      </Points>\n";

            vts << "    </Piece>\n";
            vts << "  </StructuredGrid>\n";

            vts << "  <AppendedData encoding=\"raw\">\n";
            vts << '_';

            for (const auto &arr : arrays)
            {
                const std::uint32_t nbytes32 = static_cast<std::uint32_t>(arr.nbytes);
                vts.write(reinterpret_cast<const char *>(&nbytes32), sizeof(nbytes32));

                if (!arr.isPoints)
                {
                    std::ifstream in(arr.path, std::ios::binary);
                    if (!in)
                    {
                        std::cerr << "Error opening binary field file " << arr.path.string() << " when writing VTS. Writing zeros instead.\n";
                        std::vector<char> zeros(arr.nbytes, 0);
                        vts.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));

                        continue;
                    }

                    std::vector<char> buffer(1 << 20);
                    while (in)
                    {
                        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                        const std::streamsize got = in.gcount();

                        if (got > 0)
                        {
                            vts.write(buffer.data(), got);
                        }
                    }
                }
                else
                {
                    for (uint64_t k = 0; k < NZ; ++k)
                    {
                        for (uint64_t j = 0; j < NY; ++j)
                        {
                            for (uint64_t i = 0; i < NX; ++i)
                            {
                                const float x = static_cast<float>(i);
                                const float y = static_cast<float>(j);
                                const float z = static_cast<float>(k);
                                vts.write(reinterpret_cast<const char *>(&x), sizeof(float));
                                vts.write(reinterpret_cast<const char *>(&y), sizeof(float));
                                vts.write(reinterpret_cast<const char *>(&z), sizeof(float));
                            }
                        }
                    }
                }
            }

            vts << "\n  </AppendedData>\n";
            vts << "</VTKFile>\n";
            vts.close();

            std::cout << "VTS file written to: " << vtsPath.string() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error while writing VTS file: " << e.what() << std::endl;
        }
    }
}

#endif