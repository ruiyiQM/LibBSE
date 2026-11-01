#include "io/bse_files.h"
#include "io/fhi_aims_adapter.h"

#include <librpa_file_reader.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace
{

void write_wc(const fs::path &file, int iat, int jat, double value)
{
    std::ofstream output(file);
    if (!output) throw std::runtime_error("cannot create Wc test file");
    output << "%%MatrixMarket matrix coordinate complex general\n"
           << "% atom pair " << iat << ' ' << jat << " R = (0 0 0)\n"
           << "1 1 1\n"
           << "1 1 " << value << " 0\n";
}

void test_coarse_qp_reader(const fs::path &input_dir)
{
    std::ofstream output(input_dir / "energy_qp");
    if (!output) throw std::runtime_error("cannot create energy_qp test file");
    output << "  state     occ_num        e_gs(Ha)        e_qp(Ha)\n"
           << "--------------------------------------------------\n"
           << "  K_point    1 :           0.0000          0.0000          0.0000\n"
           << "--------------------------------------------------\n"
           << "       1    2.0000   -1.1000000000E+00   -1.0000000000E+00\n"
           << "       2    2.0000   -3.0000000000E-01   -2.0000000000E-01\n"
           << "       3    0.0000    2.0000000000E-01    3.0000000000E-01\n"
           << "--------------------------------------------------\n\n"
           << "  K_point    2 :           0.1429          0.0000          0.0000\n"
           << "--------------------------------------------------\n"
           << "       1    2.0000   -1.0000000000E+00   -9.0000000000E-01\n"
           << "       2    2.0000   -2.0000000000E-01   -1.0000000000E-01\n"
           << "       3    0.0000    2.0000000000E-01    2.5000000000E-01\n"
           << "--------------------------------------------------\n";
    output.close();

    librpa_int::Dataset dataset(MPI_COMM_WORLD);
    dataset.mf_band = librpa_int::MeanField(1, 2, 3, 1);
    dataset.kfrac_band_list = {{0.0, 0.0, 0.0}, {1.0 / 7.0, 0.0, 0.0}};

    libbse::InputParameters options;
    options.input_dir = input_dir.string();
    options.nocc = 1;
    options.nvirt = 1;
    options.bse_use_fine_kgrid = 0;
    const auto qp = libbse::read_qp_bands(options, dataset);
    const std::vector<double> expected{-0.4, 0.6, -0.2, 0.5};
    if (qp.ncore != 1 || qp.nk != 2 || qp.nbands != 2
        || qp.energies_ry.size() != expected.size())
        throw std::runtime_error("coarse-grid quasiparticle dimensions are incorrect");
    for (std::size_t i = 0; i != expected.size(); ++i)
        if (std::abs(qp.energies_ry[i] - expected[i]) > 1.0e-13)
            throw std::runtime_error("coarse-grid quasiparticle energy is incorrect");
    if (std::abs(qp.direct_gap_ry - 0.7) > 1.0e-13
        || std::abs(qp.indirect_gap_ry - 0.7) > 1.0e-13)
        throw std::runtime_error("coarse-grid quasiparticle gap is incorrect");
}

void test_uniform_qp_lookup(const fs::path &input_dir)
{
    librpa_int::Dataset dataset(MPI_COMM_WORLD);
    dataset.mf_band = librpa_int::MeanField(1, 8, 3, 1);
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                dataset.kfrac_band_list.push_back({.1 + .5*x, .2 + .5*y, .3 + .5*z});
    std::reverse(dataset.kfrac_band_list.begin(), dataset.kfrac_band_list.end());
    libbse::InputParameters options;
    options.input_dir = input_dir.string();
    options.qp_data = input_dir.string();
    options.qp_format = "energy_qp";
    options.nocc = options.nvirt = 1;
    const std::vector<int> order{7,6,5,4,3,2,1,0};
    const auto write = [&](const std::vector<int> &indices, double displacement = 5.e-6) {
        std::ofstream output(input_dir / "energy_qp");
        output << std::setprecision(17);
        for (const int i : indices)
        {
            const auto &k = dataset.kfrac_band_list[i];
            output << "K_point " << i+1 << " : " << k.x-1+displacement << ' ' << k.y-1 << ' ' << k.z-1 << '\n'
                   << "1 2 -1 -1\n2 2 -.2 " << -.2-.01*i
                   << "\n3 0 .3 " << .3+.01*i << "\n---\n";
        }
    };
    const auto verify = [](const libbse::QuasiparticleBands &qp) {
        if (qp.ncore != 1 || qp.nk != 8) throw std::runtime_error("uniform QP dimensions differ");
        for (int i = 0; i < 8; ++i)
            if (std::abs(qp.energies_ry[2*i] - (-.4-.02*i)) > 1.e-13
                || std::abs(qp.energies_ry[2*i+1] - (.6+.02*i)) > 1.e-13)
                throw std::runtime_error("shifted/permuted/periodic QP lookup changed energies");
    };
    write(order);
    verify(libbse::read_qp_bands(options, dataset));
    {
        std::ofstream output(input_dir / "GW_band_spin_1.dat");
        output << std::setprecision(17);
        for (const int i : order)
        {
            const auto &k = dataset.kfrac_band_list[i];
            const double ev_per_ha = 2 * libbse::PARAM.constants.ry_to_ev;
            output << i+1 << ' ' << k.x-1 << ' ' << k.y-1 << ' ' << k.z-1
                   << " .25 " << -ev_per_ha << " .25 " << (-.2-.01*i)*ev_per_ha
                   << " 0 " << (.3+.01*i)*ev_per_ha << '\n';
        }
    }
    options.qp_format = "fine_band";
    verify(libbse::read_qp_bands(options, dataset));
    options.qp_format = "energy_qp";
    const auto reject = [&](const librpa_int::Dataset &data, const std::string &message) {
        try { (void)libbse::read_qp_bands(options, data); }
        catch (const std::runtime_error &error) {
            if (std::string(error.what()).find(message) != std::string::npos) return;
            throw;
        }
        throw std::runtime_error("invalid QP input was accepted: " + message);
    };
    auto duplicate = order;
    duplicate[0] = duplicate[1];
    write(duplicate); reject(dataset, "duplicate");
    auto missing = order;
    missing.pop_back();
    write(missing); reject(dataset, "missing");
    write(order, .1); reject(dataset, "absent");
    dataset.kfrac_band_list[0] = dataset.kfrac_band_list[1];
    write(order); reject(dataset, "ambiguous");

    librpa_int::Dataset dense(MPI_COMM_WORLD);
    dense.mf_band = librpa_int::MeanField(1, 10000, 3, 1);
    for (int k = 0; k < 10000; ++k) dense.kfrac_band_list.push_back({k/10000.,0.,0.});
    {
        std::ofstream output(input_dir / "energy_qp");
        output << "K_point 1 : 0.00005 0 0\n1 2 -1 -1\n2 2 -.2 -.2\n3 0 .3 .3\n---\n";
    }
    reject(dense, "ambiguous");
}

void test_aims_reader_view(const fs::path &input_dir, const fs::path &output_dir)
{
    {
        std::ofstream basis(input_dir / "basis_out");
        basis << "1 1 1 aims\n";
        std::ofstream band(input_dir / "band_out");
        band << "2\n1\n3\n1\n";
        std::ofstream coulomb(input_dir / "coulomb_cut_0.txt");
        coulomb << "test\n";
    }
    {
        std::ofstream velocity(input_dir / "velocity_matrix",
                               std::ios::binary);
        velocity << "canonical velocity fixture\n";
    }
    libbse::InputParameters options;
    options.input_dir = input_dir.string();
    options.output_dir = output_dir.string();
    libbse::resolve_input_format(options);
    if (options.input_format != "fhi_aims"
        || options.wavefunction_gauge != "native")
        throw std::runtime_error("FHI-aims input auto-detection failed");
    const fs::path view = libbse::prepare_fhi_aims_reader_view(
        MPI_COMM_WORLD, options);
    if (!fs::is_symlink(view / "coulomb_unshrinked_cut_0.txt")
        || !fs::is_symlink(view / "velocity_matrix")
        || fs::file_size(view / "velocity_matrix") == 0)
        throw std::runtime_error("FHI-aims reader compatibility view is incomplete");
}

void test_aims_reader_rejects_missing_velocity(
    const fs::path &input_dir, const fs::path &output_dir)
{
    fs::create_directories(input_dir);
    {
        std::ofstream basis(input_dir / "basis_out");
        basis << "1 1 1 aims\n";
        std::ofstream band(input_dir / "band_out");
        band << "1\n1\n2\n1\n";
    }
    libbse::InputParameters options;
    options.input_dir = input_dir.string();
    options.output_dir = output_dir.string();
    bool rejected = false;
    try
    {
        (void)libbse::prepare_fhi_aims_reader_view(MPI_COMM_WORLD, options);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error(
            "FHI-aims input without velocity data was not rejected");
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int status = 0;
    const fs::path root = fs::temp_directory_path()
                          / ("libbse_wc_local_"
                             + std::to_string(static_cast<long long>(::getpid())));
    try
    {
        const fs::path input_dir = root / "input";
        const fs::path wc_dir = root / "librpa.d";
        fs::create_directories(input_dir);
        fs::create_directories(wc_dir);

        test_coarse_qp_reader(input_dir);
        test_uniform_qp_lookup(input_dir);
        test_aims_reader_view(input_dir, root / "adapter_output");
        test_aims_reader_rejects_missing_velocity(
            root / "missing_velocity_input",
            root / "missing_velocity_output");

        libbse::TensorMap<libbse::Complex> bare;
        const libbse::Cell cell{0, 0, 0};
        for (int iat = 0; iat < 2; ++iat)
            for (int jat = 0; jat < 2; ++jat)
            {
                RI::Tensor<libbse::Complex> tensor({1, 1});
                tensor(0, 0) = 1.0;
                bare[iat][{jat, cell}] = std::move(tensor);
            }

        // Only this local LibRI atom pair is present. Any attempt to read a
        // nonlocal pair therefore makes the test fail with a missing file.
        write_wc(wc_dir / "Wc_Mu_0_Nu_1_iR_0_ifreq_0.mtx", 0, 1, 2.0);

        libbse::InputParameters options;
        options.input_dir = input_dir.string();
        const auto screened = libbse::read_screened_interaction(
            options, bare, 1, std::vector<int>{0}, std::vector<int>{1});
        if (screened.size() != 1 || screened.at(0).size() != 1
            || std::abs(screened.at(0).at({1, cell})(0, 0)
                        - libbse::Complex(3.0, 0.0)) > 1.0e-13)
            throw std::runtime_error(
                "local-pair screened interaction is incorrect");

        fs::remove_all(root);
    }
    catch (const std::exception &error)
    {
        fs::remove_all(root);
        std::cerr << error.what() << '\n';
        status = 1;
    }
    MPI_Finalize();
    return status;
}
