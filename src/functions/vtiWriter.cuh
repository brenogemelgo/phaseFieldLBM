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
    VTI (ImageData) writer function

Namespace
    host

SourceFiles
    vtkWriter.cuh

\*---------------------------------------------------------------------------*/

#ifndef VTIWRITER_CUH
#define VTIWRITER_CUH

#include "include/postCommon.cuh"

namespace host
{
    template <typename Container>
    __host__ static inline void writeImageData(
        const Container &fieldsCfg,
        const std::string &SIM_DIR,
        const label_t STEP)
    {
        const std::uint64_t NX = static_cast<std::uint64_t>(mesh::nx);
        const std::uint64_t NY = static_cast<std::uint64_t>(mesh::ny);
        const std::uint64_t NZ = static_cast<std::uint64_t>(mesh::nz);

        const std::uint64_t NNODES = NX * NY * NZ;
        const std::uint64_t nodeBytes = NNODES * static_cast<std::uint64_t>(sizeof(scalar_t));

        std::ostringstream stepStr;
        stepStr << std::setw(6) << std::setfill('0') << STEP;
        const std::string stepSuffix = stepStr.str();

        const std::string vtiFileName = "step_" + stepSuffix + ".vti";
        const std::filesystem::path vtiPath = std::filesystem::path(SIM_DIR) / vtiFileName;

        std::vector<detail::AppendedArray> arrays;
        arrays.reserve(fieldsCfg.size());

        std::uint64_t currentOffset = 0;
        constexpr std::uint64_t headerSize = sizeof(std::uint32_t);

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
                const std::uint64_t fs =
                    static_cast<std::uint64_t>(std::filesystem::file_size(binPath, ec_fs));

                if (ec_fs)
                {
                    std::cerr << "VTI writer: could not get size for " << binPath.string() << " (" << ec_fs.message() << "), skipping variable " << cfg.name << ".\n";

                    continue;
                }

                if (fs != nodeBytes)
                {
                    std::cerr << "VTI writer: file " << binPath.string() << " has size " << fs << " bytes, expected " << nodeBytes << " for full 3D field. " << "Skipping variable " << cfg.name << ".\n";

                    continue;
                }

                detail::AppendedArray arr;
                arr.name = cfg.name;
                arr.path = binPath;
                arr.nbytes = fs;
                arr.offset = currentOffset;

                arrays.push_back(arr);
                currentOffset += headerSize + arr.nbytes;
                hasFieldArrays = true;
            }

            if (!hasFieldArrays)
            {
                std::cerr << "VTI writer: no valid 3D fields to write for step " << STEP << ", aborting VTI.\n";

                return;
            }

            std::ofstream vti(vtiPath, std::ios::binary | std::ios::trunc);
            if (!vti)
            {
                std::cerr << "Error opening VTI file " << vtiPath.string() << " for writing.\n";

                return;
            }

            const char *scalarTypeName = detail::vtScalarTypeName<scalar_t>::value();

            constexpr double ox = 0.0, oy = 0.0, oz = 0.0;
            constexpr double sx = 1.0, sy = 1.0, sz = 1.0;

            vti << R"(<?xml version="1.0"?>)" << '\n';
            vti << R"(<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">)" << '\n';

            vti << "  <ImageData WholeExtent=\"0 " << (mesh::nx - 1)
                << " 0 " << (mesh::ny - 1)
                << " 0 " << (mesh::nz - 1)
                << "\" Origin=\"" << ox << " " << oy << " " << oz
                << "\" Spacing=\"" << sx << " " << sy << " " << sz << "\">\n";

            vti << "    <Piece Extent=\"0 " << (mesh::nx - 1)
                << " 0 " << (mesh::ny - 1)
                << " 0 " << (mesh::nz - 1) << "\">\n";

            vti << "      <PointData Scalars=\"" << arrays.front().name << "\">\n";
            for (const auto &arr : arrays)
            {
                vti << "        <DataArray type=\"" << scalarTypeName
                    << "\" Name=\"" << arr.name
                    << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
                    << arr.offset << "\"/>\n";
            }
            vti << "      </PointData>\n";

            vti << "      <CellData/>\n";
            vti << "    </Piece>\n";
            vti << "  </ImageData>\n";

            vti << "  <AppendedData encoding=\"raw\">\n";
            vti << '_';

            for (const auto &arr : arrays)
            {
                const std::uint32_t nbytes32 = static_cast<std::uint32_t>(arr.nbytes);
                vti.write(reinterpret_cast<const char *>(&nbytes32), sizeof(nbytes32));

                std::ifstream in(arr.path, std::ios::binary);
                if (!in)
                {
                    std::cerr << "Error opening binary field file " << arr.path.string() << " when writing VTI. Writing zeros instead.\n";
                    std::vector<char> zeros(arr.nbytes, 0);
                    vti.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));

                    continue;
                }

                std::vector<char> buffer(1 << 20);
                while (in)
                {
                    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                    const std::streamsize got = in.gcount();

                    if (got > 0)
                    {
                        vti.write(buffer.data(), got);
                    }
                }
            }

            vti << "\n  </AppendedData>\n";
            vti << "</VTKFile>\n";
            vti.close();

            std::cout << "VTI file written to: " << vtiPath.string() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error while writing VTI file: " << e.what() << std::endl;
        }
    }
}

#endif
