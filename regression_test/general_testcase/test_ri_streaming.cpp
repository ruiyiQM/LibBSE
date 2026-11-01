// Integration check: run in a small, complete FHI-aims LibBSE input directory.
// Compare all distributed matrix entries, before eigensolver/phase conventions.
#include "bse/molecular_lri.h"
#include "io/aims_screened.h"
#include "io/chi0_screening.h"
#include "io/bse_files.h"
#include "io/fhi_aims_adapter.h"
#include "interface/librpa_api.h"
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <functional>
#include <filesystem>
#include <memory>
#include <fstream>
#include <cstdlib>
int main(int argc,char **argv) {
    int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&provided);
    LibRPA_API::initialize();
    try {
        if(provided<MPI_THREAD_MULTIPLE)throw std::runtime_error("LibRI requires MPI_THREAD_MULTIPLE");
        auto &options=libbse::PARAM.inp;
        libbse::PARAM.read();libbse::resolve_input_format(options);
        const bool selected=argc==2 && std::string(argv[1])=="--benchmark-apply";
        const bool benchmark=selected || (argc==2 && std::string(argv[1])=="--benchmark");
        if(argc!=1 && !benchmark)throw std::invalid_argument("expected no arguments, --benchmark or --benchmark-apply");
        if(benchmark)options.bse_solver="haydock";
        LibRPA_API::ReaderOptions ro;
        ro.input_dir=options.input_dir;ro.output_dir=options.output_dir;
        ro.input_format=options.input_format;ro.read_band_data=options.bse_use_fine_kgrid==1;
        auto ds=LibRPA_API::read_dataset(MPI_COMM_WORLD,ro);
        if (std::getenv("LIBBSE_TEST_MIXED_CACHE_BUDGET")) {
            int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            const auto pairs = ds->atoms.size() * ds->atoms.size();
            if (pairs && (rank / pairs) % 2) options.bse_kmpi_cache_mb = 0;
        }
        const auto qp=libbse::read_qp_bands(options,*ds);
        libbse::apply_wavefunction_gauge(*ds,options,qp);
        auto bare=LibRPA_API::build_bare_coulomb(*ds);
        libbse::MolecularLri molecular(*ds,options,qp);
        libbse::Chi0Screening dielectric(options,*ds);
        auto screened=options.screened_format=="librpa_wc"
            ? libbse::read_screened_interaction(options,bare,ds->pbc.Rlist.size(),
                molecular.local_i_atoms(),molecular.local_j_atoms())
            : libbse::read_aims_screened_interaction(options,*ds,
                molecular.local_i_atoms(),molecular.local_j_atoms(),&dielectric);
        const bool synthetic_complex_w = std::getenv("LIBBSE_TEST_COMPLEX_W") != nullptr;
        const auto rotate_screened = [&](auto &map) {
            if (!synthetic_complex_w) return;
            for (auto &[atom, pairs] : map)
                for (auto &[key, tensor] : pairs)
                    for (std::size_t i=0;i<tensor.shape.get_shape_all();++i)
                        tensor.ptr()[i] *= libbse::Complex(.8,.3);
        };
        rotate_screened(screened);
        auto coefficients=libbse::convert_lri_coefficients(*ds);
        // Preserve raw cell keys for reading the independent reference Wc + V.
        decltype(bare) reference_bare;
        if (!selected) reference_bare = bare;
        libbse::remap_to_nearest_bvk_cell(coefficients,*ds);
        libbse::remap_to_nearest_bvk_cell(bare,*ds);
        libbse::remap_to_nearest_bvk_cell(screened,*ds);
        // initialize consumes all three tensor maps.
        decltype(coefficients) reference_coefficients;
        if (!selected) reference_coefficients = coefficients;
        molecular.initialize(coefficients,bare,screened);
        const int d=qp.nk*options.nocc*options.nvirt;
        auto reference_options = options;
        reference_options.bse_solver = "elpa";
        reference_options.bse_haydock_apply = "realspace";
        reference_options.output_dir = options.output_dir + "/cartesian_reference";
        std::unique_ptr<libbse::MolecularLri> reference_ptr;
        if (!selected) {
        // Explicit and streamed references require the Cartesian atom/k partition.
        std::filesystem::create_directories(reference_options.output_dir);
        reference_ptr = std::make_unique<libbse::MolecularLri>(*ds, reference_options, qp);
        auto &reference_molecular = *reference_ptr;
        auto reference_screened = options.screened_format == "librpa_wc"
            ? libbse::read_screened_interaction(reference_options, reference_bare, ds->pbc.Rlist.size(),
                reference_molecular.local_i_atoms(), reference_molecular.local_j_atoms())
            : libbse::read_aims_screened_interaction(reference_options, *ds,
                reference_molecular.local_i_atoms(), reference_molecular.local_j_atoms(), &dielectric);
        rotate_screened(reference_screened);
        libbse::remap_to_nearest_bvk_cell(reference_screened, *ds);
        libbse::remap_to_nearest_bvk_cell(reference_bare, *ds);
        reference_molecular.initialize(reference_coefficients, reference_bare, reference_screened);
        }
        if(synthetic_complex_w)std::cout<<"RI_SYNTHETIC_COMPLEX_W rotated=true hermitian_check=false\n";
        if(benchmark) {
            std::vector<libbse::Complex> x(d),a,b;
            for(int i=0;i<d;++i)x[i]={std::sin(.7*(i+1)),std::cos(.3*(i+2))};
            double error=0.;
            if(!selected) {
                molecular.apply_tda_real_space(x,a,"singlet");
                reference_ptr->apply_tda_streamed(x,b,"singlet");
                for(int i=0;i<d;++i)error=std::max(error,std::abs(a[i]-b[i]));
                if(error>5e-12)throw std::runtime_error("benchmark operators differ");
            }
            int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            const auto measure=[&](const std::string &name,const std::function<void()> &apply){
                const auto sample=[&](int repeats){
                    MPI_Barrier(MPI_COMM_WORLD);
                    const double start=MPI_Wtime();
                    for(int i=0;i<repeats;++i)apply();
                    const double local=(MPI_Wtime()-start)/repeats;
                    double maximum=0.;
                    MPI_Allreduce(&local,&maximum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                    return maximum;
                };
                const double trial=sample(1);
                if(selected) {
                    if(a.size()!=static_cast<std::size_t>(d))throw std::runtime_error("benchmark result has wrong dimension");
                    for(const auto value:a)
                        if(!std::isfinite(std::abs(value)))throw std::runtime_error("nonfinite benchmark result");
                }
                // One expensive Hx per batch suffices; retain three independent batches.
                const int repeats=std::max(selected?1:3,std::min(2000,static_cast<int>(std::ceil(.1/std::max(trial,1.e-9)))));
                std::vector<double> samples;
                for(int i=0;i<3;++i)samples.push_back(sample(repeats));
                std::sort(samples.begin(),samples.end());
                if(rank==0) {
                    std::cout<<std::setprecision(16)<<"RI_APPLY_BENCHMARK nk="<<qp.nk<<" implementation="<<name
                    <<" seconds="<<samples[1]<<" min="<<samples.front()<<" max="<<samples.back()
                    <<" repeats="<<repeats<<" reference_checked="<<!selected<<" trial_seconds="<<trial;
                    if(!selected)std::cout<<" max_abs="<<error;
                    std::cout<<'\n';
                }
            };
            if(selected) {
                // Large-grid correctness uses the separate ELPA/spectrum checks.
                measure(options.bse_haydock_apply,[&]{molecular.apply_tda(x,a,"singlet");});
                if(rank==0) {
                    std::ofstream out("hx.bin",std::ios::binary);
                    out.write(reinterpret_cast<const char *>(a.data()),a.size()*sizeof(libbse::Complex));
                    out.close();if(!out)throw std::runtime_error("could not save Hx validation vector");
                }
            } else {
            measure("real_space",[&]{molecular.apply_tda_real_space(x,a,"singlet");});
            if(libbse::FftBatch::available()) {
                reference_ptr->apply_tda_fft(x,a,"singlet");
                for(int i=0;i<d;++i)error=std::max(error,std::abs(a[i]-b[i]));
                if(error>5e-12)throw std::runtime_error("FFT benchmark operator differs");
                measure("fft",[&]{reference_ptr->apply_tda_fft(x,a,"singlet");});
            }
            measure("streamed",[&]{reference_ptr->apply_tda_streamed(x,b,"singlet");});
            }
        } else {
        auto &reference_molecular = *reference_ptr;
        librpa_int::ArrayDesc desc(ds->blacs_h);
        if(desc.init(d,d,3,3,0,0)) throw std::runtime_error("descriptor failed");
        const auto local=static_cast<std::size_t>(desc.lld())*desc.n_loc();
        const auto add=[&](int which,std::vector<libbse::Complex> &mat){
            reference_options.bse_memory_optimized = options.bse_memory_optimized;
            reference_options.bse_ri_batch_blocks = options.bse_ri_batch_blocks;
            if(which==0)reference_molecular.add_hartree_a(mat,desc,1.);
            if(which==1)reference_molecular.add_screened_a(mat,desc,1.);
            if(which==2)reference_molecular.add_hartree_b(mat,desc,1.);
            if(which==3)reference_molecular.add_screened_b(mat,desc,1.);
        };
        int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        int ranks = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        const auto pairs = ds->atoms.size() * ds->atoms.size();
        if (options.bse_haydock_apply == "realspace_kmpi" && pairs > 0
            && static_cast<std::size_t>(ranks) > pairs
            && (static_cast<std::size_t>(ranks) - 1) / pairs < static_cast<std::size_t>(qp.nk)) {
            std::vector<libbse::Complex> matrix(local), x(d), output;
            const auto rejected = [](const std::function<void()> &action) {
                try { action(); }
                catch (const std::logic_error &) { return; }
                throw std::runtime_error("distributed-k object silently accepted an incompatible reference action");
            };
            rejected([&]{ molecular.add_hartree_a(matrix,desc,1.); });
            rejected([&]{ molecular.add_hartree_b(matrix,desc,1.); });
            rejected([&]{ molecular.add_screened_a(matrix,desc,1.); });
            rejected([&]{ molecular.add_screened_b(matrix,desc,1.); });
            rejected([&]{ molecular.apply_tda_streamed(x,output,"singlet"); });
            if (libbse::FftBatch::available())
                rejected([&]{ molecular.apply_tda_fft(x,output,"singlet"); });
            if (rank==0) std::cout << "RI_KMPI_REFERENCE_GUARDS checked="
                                  << (libbse::FftBatch::available() ? 6 : 5) << '\n';
        }
        std::vector<libbse::Complex> hartree_reference,screened_reference;
        for(int which=0;which<4;++which) {
            std::vector<libbse::Complex> reference(local),candidate(local);
            options.bse_memory_optimized=false;add(which,reference);
            if(which==0)hartree_reference=reference;
            if(which==1)screened_reference=reference;
            for(int batch:{1,3,64}) {
                std::fill(candidate.begin(),candidate.end(),libbse::Complex{});
                options.bse_memory_optimized=true;options.bse_ri_batch_blocks=batch;
                add(which,candidate);
                double error=0.,all_error=0.;
                for(std::size_t i=0;i<local;++i)error=std::max(error,std::abs(candidate[i]-reference[i]));
                MPI_Allreduce(&error,&all_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                if(rank==0)std::cout<<"RI_MATRIX_COMPARE kind="<<which<<" batch="<<batch<<" max_abs="<<all_error<<'\n';
                if(all_error>5e-12)throw std::runtime_error("streamed RI matrix differs from unbatched result");
            }
        }
        for(const std::string spin:{"singlet","rpa","triplet"}) {
        std::vector<libbse::Complex> matrix(local);
        const auto weights=libbse::interaction_coefficients(spin);
        for(std::size_t i=0;i<local;++i)matrix[i]=weights.hartree*hartree_reference[i]+weights.screened*screened_reference[i];
        std::vector<libbse::Complex> previous_x,previous_hx;
        for(int probe=0;probe<3;++probe) {
            std::vector<libbse::Complex> x(d),reference(d),partial(d),candidate;
            for(int i=0;i<d;++i)x[i]={std::sin(0.7*(i+1)*(probe+1)),std::cos(0.3*(i+2)*(probe+2))};
            for(int j=0;j<desc.n_loc();++j)
                for(int i=0;i<desc.m_loc();++i)
                    partial[desc.indx_l2g_r(i)]+=matrix[i+static_cast<std::size_t>(j)*desc.lld()]*x[desc.indx_l2g_c(j)];
            MPI_Allreduce(partial.data(),reference.data(),d,MPI_C_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);
            for(int k=0;k<qp.nk;++k)
                for(int i=0;i<options.nocc;++i)
                    for(int a=0;a<options.nvirt;++a) {
                        const int p=(k*options.nocc+i)*options.nvirt+a;
                        const int offset=k*qp.nbands;
                        reference[p]+=(qp.energies_ry[offset+options.nocc+a]-qp.energies_ry[offset+i])*x[p];
                    }
            for(int batch:{1,3,64}) {
                options.bse_ri_batch_blocks=batch;
                reference_options.bse_ri_batch_blocks = options.bse_ri_batch_blocks;
                reference_molecular.apply_tda_streamed(x,candidate,spin);
                double error=0.;
                for(int p=0;p<d;++p)error=std::max(error,std::abs(candidate[p]-reference[p]));
                if(rank==0)std::cout<<"RI_APPLY_COMPARE spin="<<spin<<" probe="<<probe<<" batch="<<batch<<" max_abs="<<error<<'\n';
                if(error>5e-12)throw std::runtime_error("LRI apply differs from explicit TDA H*x");
            }
            molecular.apply_tda_real_space(x,candidate,spin);
            double error=0.;
            for(int p=0;p<d;++p)error=std::max(error,std::abs(candidate[p]-reference[p]));
            if(rank==0)std::cout<<"RI_REAL_SPACE_COMPARE spin="<<spin<<" probe="<<probe<<" max_abs="<<error<<'\n';
            if(error>5e-12)throw std::runtime_error("real-space LRI apply differs from explicit TDA H*x");
            if(!previous_x.empty() && !synthetic_complex_w) {
                libbse::Complex lhs{},rhs{};double nx=0.,ny=0.;
                for(int p=0;p<d;++p) {
                    lhs+=std::conj(previous_x[p])*candidate[p];
                    rhs+=std::conj(previous_hx[p])*x[p];
                    nx+=std::norm(previous_x[p]);ny+=std::norm(x[p]);
                }
                const double adjoint_error=std::abs(lhs-rhs)/std::sqrt(nx*ny);
                if(rank==0)std::cout<<"RI_REAL_SPACE_ADJOINT spin="<<spin<<" probe="<<probe<<" max_abs="<<adjoint_error<<'\n';
                if(adjoint_error>5e-12)throw std::runtime_error("real-space LRI adjoint relation differs");
            }
            previous_x=x;previous_hx=candidate;
            if(libbse::FftBatch::available()) {
                // Distributed-k objects use a separately constructed Cartesian FFT reference.
                (options.bse_haydock_apply == "realspace_kmpi" ? reference_molecular : molecular)
                    .apply_tda_fft(x,candidate,spin);
                error=0.;
                for(int p=0;p<d;++p)error=std::max(error,std::abs(candidate[p]-reference[p]));
                if(rank==0)std::cout<<"RI_FFT_COMPARE spin="<<spin<<" probe="<<probe<<" max_abs="<<error<<'\n';
                if(error>5e-12)throw std::runtime_error("FFT LRI apply differs from explicit TDA H*x");
            }
        }
        }
        }
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
    LibRPA_API::finalize();MPI_Finalize();return 0;
}
