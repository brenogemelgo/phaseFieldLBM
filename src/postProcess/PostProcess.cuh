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
    Post-processing utilities

Namespace
    host

SourceFiles
    PostProcess.cuh

\*---------------------------------------------------------------------------*/

#ifndef POSTPROCESS_CUH
#define POSTPROCESS_CUH

#include "LBMIncludes.cuh"
#include "fileIO/fields.cuh"

namespace host
{
    class PostProcess
    {
    public:
        PostProcess() = default;

        ~PostProcess() = default;
        PostProcess(const PostProcess &) = delete;
        PostProcess &operator=(const PostProcess &) = delete;

        template <typename Container>
        __host__ bool bin(
            const Container &fieldsCfg,
            const std::string &SIM_DIR,
            const label_t t,
            const LBMFields &fields) const
        {
            bool ok = true;

            const std::string stepSuffix = stepSuffix6(t);

            for (const auto &cfg : fieldsCfg)
            {
                if (!cfg.includeInPost)
                {
                    continue;
                }

                const size_t n = (cfg.shape == FieldDumpShape::Grid3D) ? static_cast<size_t>(size::cells()) : static_cast<size_t>(size::stride());

                const scalar_t *d_ptr = getDeviceFieldPointer(fields, cfg.id);
                if (d_ptr == nullptr)
                {
                    if (verbose_)
                    {
                        std::cerr << "PostProcess::bin: null device ptr for field " << cfg.name << ", skipping.\n";
                    }

                    continue;
                }

                const std::filesystem::path outPath = std::filesystem::path(SIM_DIR) / (std::string(cfg.name) + stepSuffix + ".bin");

                static thread_local std::vector<scalar_t> h;
                if (h.size() != n)
                {
                    h.resize(n);
                }

                checkCudaErrors(cudaMemcpy(h.data(), d_ptr, n * sizeof(scalar_t), cudaMemcpyDeviceToHost));

                std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
                if (!out)
                {
                    std::cerr << "PostProcess::bin: could not open " << outPath.string() << " for writing.\n";
                    ok = false;

                    continue;
                }

                out.write(reinterpret_cast<const char *>(h.data()), static_cast<std::streamsize>(h.size() * sizeof(scalar_t)));
            }

            return ok;
        }

        template <typename Container>
        __host__ bool vti(
            const Container &fieldsCfg,
            const std::string &SIM_DIR,
            const label_t t) const
        {
            struct ArrayDesc
            {
                std::string name;
                std::filesystem::path path;
                std::uint64_t nbytes = 0;
                std::uint64_t offset = 0;
            };

            const std::uint64_t NX = static_cast<std::uint64_t>(mesh::nx);
            const std::uint64_t NY = static_cast<std::uint64_t>(mesh::ny);
            const std::uint64_t NZ = static_cast<std::uint64_t>(mesh::nz);

            const std::uint64_t NNODES = NX * NY * NZ;
            const std::uint64_t nodeBytes = NNODES * static_cast<std::uint64_t>(sizeof(scalar_t));

            const std::string stepSuffix = stepSuffix6(t);

            const std::filesystem::path vtiPath = std::filesystem::path(SIM_DIR) / ("step_" + stepSuffix + ".vti");

            std::vector<ArrayDesc> arrays;
            arrays.reserve(fieldsCfg.size());

            std::uint64_t currentOffset = 0;
            constexpr std::uint64_t headerSize = sizeof(std::uint32_t);

            for (const auto &cfg : fieldsCfg)
            {
                if (!cfg.includeInPost || cfg.shape != FieldDumpShape::Grid3D)
                {
                    continue;
                }

                const std::filesystem::path binPath = std::filesystem::path(SIM_DIR) / (std::string(cfg.name) + stepSuffix + ".bin");

                std::error_code ec{};
                const bool exists = std::filesystem::exists(binPath, ec);
                if (ec || !exists)
                {
                    std::cerr << "PostProcess::vti: missing bin " << binPath.string() << " for field " << cfg.name << " (run bin first).\n";

                    continue;
                }

                std::error_code ec_sz{};
                const std::uint64_t fs =
                    static_cast<std::uint64_t>(std::filesystem::file_size(binPath, ec_sz));
                if (ec_sz)
                {
                    std::cerr << "PostProcess::vti: could not stat " << binPath.string() << " (" << ec_sz.message() << "), skipping " << cfg.name << ".\n";

                    continue;
                }

                if (fs != nodeBytes)
                {
                    std::cerr << "PostProcess::vti: " << binPath.string() << " has " << fs << " bytes; expected " << nodeBytes << " for Grid3D. Skipping " << cfg.name << ".\n";

                    continue;
                }

                ArrayDesc a{};
                a.name = cfg.name;
                a.path = binPath;
                a.nbytes = fs;
                a.offset = currentOffset;

                arrays.push_back(a);
                currentOffset += headerSize + a.nbytes;
            }

            if (arrays.empty())
            {
                std::cerr << "PostProcess::vti: no valid Grid3D bins found for step " << t << ". Not writing VTI.\n";

                return false;
            }

            std::ofstream vti(vtiPath, std::ios::binary | std::ios::trunc);
            if (!vti)
            {
                std::cerr << "PostProcess::vti: could not open " << vtiPath.string() << " for writing.\n";

                return false;
            }

            const char *scalarTypeName = vtkScalarTypeName();

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
            for (const auto &a : arrays)
            {
                vti << "        <DataArray type=\"" << scalarTypeName
                    << "\" Name=\"" << a.name
                    << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\""
                    << a.offset << "\"/>\n";
            }
            vti << "      </PointData>\n";
            vti << "      <CellData/>\n";
            vti << "    </Piece>\n";
            vti << "  </ImageData>\n";

            vti << "  <AppendedData encoding=\"raw\">\n";
            vti << '_';

            for (const auto &a : arrays)
            {
                const std::uint32_t nbytes32 = static_cast<std::uint32_t>(a.nbytes);
                vti.write(reinterpret_cast<const char *>(&nbytes32), sizeof(nbytes32));

                std::ifstream in(a.path, std::ios::binary);
                if (!in)
                {
                    std::cerr << "PostProcess::vti: could not open " << a.path.string() << " while writing VTI. Writing zeros for " << a.name << ".\n";
                    std::vector<char> zeros(a.nbytes, 0);
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

            if (verbose_)
            {
                std::cout << "VTI file written to: " << vtiPath.string() << "\n";
            }

            return true;
        }

    private:
        bool verbose_ = true;

        __host__ static inline std::string stepSuffix6(const label_t t)
        {
            std::ostringstream ss;
            ss << std::setw(6) << std::setfill('0') << t;
            return ss.str();
        }

        __host__ static inline const char *vtkScalarTypeName() noexcept
        {
            if constexpr (std::is_same_v<scalar_t, float>)
            {
                return "Float32";
            }
            else if constexpr (std::is_same_v<scalar_t, double>)
            {
                return "Float64";
            }
            else
            {
                return "Float32";
            }
        }
    };
}

#endif
