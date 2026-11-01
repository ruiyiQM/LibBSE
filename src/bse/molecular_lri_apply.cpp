#include "molecular_lri.h"
#include "kmpi_contractions.h"
#include "tr_virtual.h"
#include "w_boundary_phase.h"
#include "utils/memory_views.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <climits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace libbse
{
namespace
{
std::vector<Complex> pack_cs(const Complex *cs, int naux, int nao, int nbands,
                             int first, int count, bool conjugate)
{
    std::vector<Complex> packed(static_cast<std::size_t>(naux) * nao * count);
    for (int row = 0; row < naux * nao; ++row)
        for (int band = 0; band < count; ++band)
        {
            const auto value = cs[row * nbands + first + band];
            packed[row * count + band] = conjugate ? std::conj(value) : value;
        }
    return packed;
}

// [aux, AO, band] -> [AO, aux, band], preserving contiguous band blocks.
void swap_aux_ao(const Complex *source, Complex *target, int naux, int nao, int bands)
{
    for (int ao = 0; ao < nao; ++ao)
        for (int aux = 0; aux < naux; ++aux)
            std::copy_n(source + (static_cast<std::size_t>(aux) * nao + ao) * bands,
                        bands, target + (static_cast<std::size_t>(ao) * naux + aux) * bands);
}


struct PairCommunicator
{
    MPI_Comm value = MPI_COMM_NULL;
    ~PairCommunicator()
    {
        if (value != MPI_COMM_NULL)
            MPI_Comm_free(&value);
    }
};

// Sum local-k densities before projecting only the locally owned output k points.
void add_screened_kmpi(const RI::LR<int, int, 3, Complex> &lr_, const InputParameters &options_,
                       int nk_, int pair_dimension_, const std::vector<Complex> &x,
                       std::vector<Complex> &local, double scale, MPI_Comm pair_comm,
                       const std::vector<int> &partner_slots, std::ofstream &log_)
{
    const int no = options_.nocc, nv = options_.nvirt, nb = no + nv;
    const auto &screened = lr_.lrik.data_pool.at("Ws_").Ds_ab;
    for (const int mu : lr_.list_I)
        for (const int nu : lr_.list_J)
        {
            std::vector<const Complex *> om(nk_), on(nk_), fm(nk_), fn(nk_);
            const auto &cm = lr_.Csk_ao_mo.at(lr_.k_indices.front()).at(mu);
            const auto &cn = lr_.Csk_ao_mo.at(lr_.k_indices.front()).at(nu);
            const int nm = cm.shape[0], nn = cn.shape[0];
            const int ns = cm.shape[1], nt = cn.shape[1];
            for (const int k : lr_.k_indices)
            {
                om[k] = lr_.map_psi.at(k).at(mu).ptr();
                on[k] = lr_.map_psi.at(k).at(nu).ptr();
                fm[k] = lr_.Csk_ao_mo.at(k).at(mu).ptr();
                fn[k] = lr_.Csk_ao_mo.at(k).at(nu).ptr();
            }
            // Band slices are independent of R; pack them once per atom pair.
            std::vector<std::array<std::vector<Complex>, 6>> cs(nk_);
            for (const int k : lr_.k2_indices)
            {
                cs[k][1] = pack_cs(fm[k], nm, ns, nb, 0, no, true);
                cs[k][3] = pack_cs(fn[k], nn, nt, nb, no, nv, false);
                cs[k][4].resize(nt * nn * nv);
                swap_aux_ao(cs[k][3].data(), cs[k][4].data(), nn, nt, nv);
            }
            for (const int k : lr_.k1_indices)
            {
                const auto f = pack_cs(fm[k], nm, ns, nb, 0, no, false);
                cs[k][2].resize(ns * nm * no);
                swap_aux_ao(f.data(), cs[k][2].data(), nm, ns, no);
                cs[k][5] = pack_cs(fn[k], nn, nt, nb, no, nv, true);
            }
            // Share each R density across a block of output k points.
            constexpr std::size_t projection_tile = 64;
            std::vector<std::vector<Complex>> output_f, output_g;
            for (std::size_t begin = 0; begin < lr_.k1_indices.size(); begin += projection_tile)
            {
                const auto count = std::min(projection_tile, lr_.k1_indices.size() - begin);
                output_f.emplace_back(static_cast<std::size_t>(nm) * ns * count * no);
                output_g.emplace_back(static_cast<std::size_t>(nn) * nt * count * nv);
                for (std::size_t slot = 0; slot < count; ++slot)
                {
                    const int k = lr_.k1_indices[begin + slot];
                    for (int row = 0; row < nm * ns; ++row)
                        std::copy_n(fm[k] + row * nb, no,
                                    output_f.back().data() + (row * count + slot) * no);
                    for (int row = 0; row < nn * nt; ++row)
                        std::copy_n(cs[k][5].data() + row * nv, nv,
                                    output_g.back().data() + (row * count + slot) * nv);
                }
            }
            auto output_f_memory = watch_memory("LibRI.kmpi_output_f", output_f);
            auto output_g_memory = watch_memory("LibRI.kmpi_output_g", output_g);
            auto cs_memory = watch_memory("LibRI.apply_R_packed_Cs", cs);
            struct Image
            {
                std::array<int, 3> cell;
                double weight;
                const Complex *w;
            };
            std::vector<Image> images;
            for (const auto &key :
                 RI::Divide_Atoms::traversal_atom_period(std::vector<int>{nu}, lr_.lrik.period))
            {
                const auto &w = RI::Global_Func::find(screened, mu, key);
                if (w.empty())
                    continue;
                if (w.shape[0] != static_cast<std::size_t>(nm) ||
                    w.shape[1] != static_cast<std::size_t>(nn))
                    throw std::runtime_error("inconsistent screened auxiliary dimensions");
                for (const auto &[cell, weight] : w_boundary_images(
                         key.second, lr_.lrik.period, options_.bse_w_boundary_average))
                    images.push_back({cell, weight, w.ptr()});
            }
            if (images.empty())
                continue;

            int threads = 1;
#ifdef _OPENMP
            threads = std::min<std::size_t>(omp_get_max_threads(), images.size());
#endif
            const std::size_t paired_fields = static_cast<std::size_t>(ns) * nt
                * (static_cast<std::size_t>(nm) + nn + 2);
            const std::size_t budget = 256ull * 1024 * 1024;
            const auto density_tile = [&](std::size_t fields) {
                return static_cast<int>(std::min<std::size_t>(threads,
                    std::max<std::size_t>(1, std::min<std::size_t>(INT_MAX / fields,
                        budget / (sizeof(Complex) * fields)))));
            };
            const bool valid_paired_fields = paired_fields > 0 && paired_fields <= INT_MAX;
            const std::size_t cache_budget = static_cast<std::size_t>(options_.bse_kmpi_cache_mb)
                * 1024 * 1024;
            int plan[2] = {threads, lr_.k1_indices == lr_.k2_indices && valid_paired_fields
                && nm <= INT_MAX / 2 && static_cast<std::size_t>(nt) * nv <= INT_MAX / 2
                && kmpi_cache_fits(cache_budget, lr_.k2_indices.size(), nm, nn, ns, nt, nv,
                                   valid_paired_fields ? density_tile(paired_fields) : 1)};
            // Every owner of this atom pair must use the same packet layout and tile.
            MPI_Allreduce(MPI_IN_PLACE, plan, 2, MPI_INT, MPI_MIN, pair_comm);
            threads = plan[0];
            const bool paired = plan[1];
            const std::size_t fields = paired ? paired_fields : 2ull * ns * nt * (nn + 1);
            if (fields == 0 || fields > static_cast<std::size_t>(INT_MAX))
                throw std::overflow_error("one R density exceeds MPI count limit");
            const int tile = density_tile(fields);
            std::vector<Complex> packet(fields * tile);
            auto packet_memory = watch_memory("LibRI.kmpi_packet", packet);
            std::vector<std::vector<Complex>> outputs(threads), weighted_virtuals(paired ? tile : 0);
            const std::size_t cached_elements = paired
                ? lr_.k1_indices.size() * static_cast<std::size_t>(nt) * nm * nv : 0;
            std::exception_ptr allocation_error;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
            for (int worker = 0; worker < threads; ++worker)
            {
                try
                {
                    outputs[worker].resize(lr_.k1_indices.size() * pair_dimension_);
                    if (paired && worker < tile)
                        weighted_virtuals[worker].resize(cached_elements);
                }
                catch (...)
                {
#ifdef _OPENMP
#pragma omp critical(libbse_kmpi_allocation)
#endif
                    {
                        if (!allocation_error)
                            allocation_error = std::current_exception();
                    }
                }
            }
            if (allocation_error)
                std::rethrow_exception(allocation_error);
            auto output_memory = watch_memory("LibRI.kmpi_outputs", outputs);
            auto cache_memory = watch_memory("LibRI.kmpi_weighted_virtuals", weighted_virtuals);
            std::vector<std::vector<double>> real_g(paired ? nk_ : 0);
            std::vector<std::vector<Complex>> projected_x(paired ? nk_ : 0),
                transformed_x(paired ? nk_ : 0);
            if (paired)
            {
                // x products and split virtual vertices are independent of R.
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
                for (std::size_t owned = 0; owned < lr_.k2_indices.size(); ++owned)
                {
                    try
                    {
                        const int k = lr_.k2_indices[owned];
                        const auto &g = cs[k][3];
                        const std::size_t columns = static_cast<std::size_t>(nt) * nv;
                        real_g[k].resize(2 * g.size());
                        for (std::size_t i = 0; i < g.size(); ++i)
                        {
                            const auto offset = (i / columns) * 2 * columns + i % columns;
                            real_g[k][offset] = g[i].real();
                            real_g[k][offset + columns] = g[i].imag();
                        }
                        projected_x[k].resize(static_cast<std::size_t>(ns) * nv);
                        transformed_x[k].resize(static_cast<std::size_t>(ns) * nm * nv);
                        std::vector<Complex> raw(static_cast<std::size_t>(nm) * ns * nv);
                        using RI::Blas_Interface::gemm;
                        gemm('C', 'N', ns, nv, no, Complex(1.), om[k], ns,
                             x.data() + k * pair_dimension_, nv, Complex{}, projected_x[k].data(), nv);
                        gemm('N', 'N', nm * ns, nv, no, Complex(1.), cs[k][1].data(), no,
                             x.data() + k * pair_dimension_, nv, Complex{}, raw.data(), nv);
                        swap_aux_ao(raw.data(), transformed_x[k].data(), nm, ns, nv);
                    }
                    catch (...)
                    {
#ifdef _OPENMP
#pragma omp critical(libbse_kmpi_allocation)
#endif
                        {
                            if (!allocation_error) allocation_error = std::current_exception();
                        }
                    }
                }
                if (allocation_error) std::rethrow_exception(allocation_error);
            }
            auto real_g_memory = watch_memory("LibRI.kmpi_real_virtual", real_g);
            auto projected_memory = watch_memory("LibRI.kmpi_projected_x", projected_x);
            auto transformed_memory = watch_memory("LibRI.kmpi_transformed_x", transformed_x);
            const auto local_k = lr_.k2_indices.size(), gauge_pairs = local_k / 2;
            const bool tr_enabled = paired && gauge_pairs > 0 && partner_slots.size() == local_k
                && kmpi_cache_fits(cache_budget, local_k, nm, nn, ns, nt, nv, tile, gauge_pairs);
            std::vector<int> tr_source(tr_enabled ? local_k : 0, -1), tr_target(tr_source);
            std::vector<std::vector<Complex>> tr_u(tr_enabled ? local_k : 0);
            long long tr_tested = 0, tr_accepted = 0;
            double tr_maximum = 0., tr_accepted_maximum = 0.;
            if (tr_enabled)
            {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+:tr_tested,tr_accepted) reduction(max:tr_maximum,tr_accepted_maximum)
#endif
                for (std::size_t owned = 0; owned < lr_.k2_indices.size(); ++owned)
                {
                    const int other = partner_slots[owned];
                    if (other <= static_cast<int>(owned)) continue;
                    const int k = lr_.k2_indices[owned], q = lr_.k2_indices[other];
                    ++tr_tested;
                    try
                    {
                        auto fit = fit_virtual_conjugate(fn[k], fn[q], nn * nt, nb, no, nv);
                        if (!fit.solved) continue;
                        tr_maximum = std::max(tr_maximum, fit.residual);
                        if (fit.residual > 1e-13 || fit.unitarity > 1e-10) continue;
                        ++tr_accepted;
                        tr_accepted_maximum = std::max(tr_accepted_maximum, fit.residual);
                        tr_source[other] = owned;
                        tr_target[owned] = other;
                        tr_u[other] = std::move(fit.matrix);
                    }
                    catch (...)
                    {
#ifdef _OPENMP
#pragma omp critical(libbse_kmpi_allocation)
#endif
                        {
                            if (!allocation_error) allocation_error = std::current_exception();
                        }
                    }
                }
                if (allocation_error) std::rethrow_exception(allocation_error);
            }
            auto tr_memory = watch_memory("LibRI.kmpi_tr_u", tr_u);
            log_ << "LRI_KMPI_TR_REUSE enabled=" << tr_enabled << " tested_pairs=" << tr_tested
                 << " accepted_pairs=" << tr_accepted << " fit_max=" << tr_maximum
                 << " accepted_max=" << tr_accepted_maximum << std::endl;
            double communication_seconds = 0;
            std::size_t collective_bytes = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
            {
                int thread = 0;
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif
                using RI::Blas_Interface::gemm;
                const Complex one(1.0), zero{};
                auto &output = outputs[thread];
                std::vector<Complex> phase(nk_), wf(paired ? 0 : nn * ns * no),
                    wg(paired ? 0 : nm * nt * nv), z(paired ? 0 : ns * nv),
                    jx(paired ? 0 : nn * ns * nv), jxt(paired ? 0 : ns * nn * nv),
                    wgt(nt * nm * nv), result(pair_dimension_),
                    d10_projected(paired ? static_cast<std::size_t>(nn) * ns * nt : 0),
                    temp00(nt * nm * no), temp(no * nt), h01(nt * projection_tile * no),
                    h10(ns * projection_tile * nv),
                    d01(nm * nt * ns), d01t(nt * nm * ns), d10t(ns * nn * nt);
                std::vector<Complex> conjugate_u(tr_accepted ? static_cast<std::size_t>(nv) * nv : 0);
                std::vector<double> wr(paired ? 2ull * nm * nn : 0),
                    products(paired ? 4ull * nm * nt * nv : 0);
                for (std::size_t begin = 0; begin < images.size(); begin += tile)
                {
                    const int count = std::min<std::size_t>(tile, images.size() - begin);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (int row = 0; row < count; ++row)
                    {
                        auto *d00 = packet.data() + row * fields;
                        auto *d01_raw = d00 + ns * nt;
                        auto *d10 = d01_raw + nn * nt * ns;
                        auto *d11 = d10 + static_cast<std::size_t>(paired ? nm : nn) * ns * nt;
                        const auto &[cell, image_weight, w] = images[begin + row];
                        std::fill_n(d00, fields, Complex{});
                        for (const int k : lr_.k_indices)
                        {
                            double angle = 0.0;
                            for (int axis = 0; axis < 3; ++axis)
                                angle += lr_.kindex_map[k][axis] * cell[axis];
                            phase[k] = std::polar(1.0, 2.0 * PARAM.constants.pi * angle);
                        }
                        if (paired)
                            for (std::size_t entry = 0; entry < static_cast<std::size_t>(nm) * nn; ++entry)
                            {
                                wr[entry] = w[entry].real();
                                wr[static_cast<std::size_t>(nm) * nn + entry] = w[entry].imag();
                            }
                        for (std::size_t owned = 0; owned < lr_.k2_indices.size(); ++owned)
                        {
                            const int k = lr_.k2_indices[owned];
                            const auto &f = cs[k][1], &g = cs[k][3], &gt = cs[k][4];
                            if (paired)
                            {
                                // Cache by packet row: a capped team can handle several R images.
                                if (tr_enabled && tr_source[owned] >= 0)
                                {
                                    const auto *primary = weighted_virtuals[row].data()
                                        + static_cast<std::size_t>(tr_source[owned]) * nt * nm * nv;
                                    gemm('N', 'N', nt * nm, nv, nv, one, primary, nv,
                                         tr_u[owned].data(), nv, zero, wgt.data(), nv);
                                }
                                else
                                {
                                    paired_virtual_products(nm, nn, nt, nv, wr, real_g[k], products,
                                        wgt.data(), weighted_virtuals[row].data() + owned * nt * nm * nv);
                                    if (tr_enabled && tr_target[owned] >= 0)
                                    {
                                        const int q = tr_target[owned];
                                        for (int j = 0; j < nv * nv; ++j)
                                            conjugate_u[j] = std::conj(tr_u[q][j]);
                                        auto *secondary = weighted_virtuals[row].data()
                                            + static_cast<std::size_t>(q) * nt * nm * nv;
                                        gemm('N', 'N', nt * nm, nv, nv, one, wgt.data(), nv,
                                             conjugate_u.data(), nv, zero, secondary, nv);
                                    }
                                }
                                const auto &px = projected_x[k], &tx = transformed_x[k];
                                gemm('N', 'N', ns, nt, nv, phase[k], px.data(), nv,
                                     on[k] + no * nt, nt, one, d00, nt);
                                gemm('N', 'T', nn * nt, ns, nv, phase[k], g.data(), nv,
                                     px.data(), nv, one, d01_raw, ns);
                                gemm('N', 'N', ns * nm, nt, nv, phase[k], tx.data(), nv,
                                     on[k] + no * nt, nt, one, d10, nt);
                                gemm('N', 'T', ns, nt, nm * nv, phase[k], tx.data(), nm * nv,
                                     wgt.data(), nm * nv, one, d11, nt);
                                continue;
                            }
                            gemm('T', 'N', nn, ns * no, nm, one, w, nn, f.data(), ns * no, zero,
                                 wf.data(), ns * no);
                            gemm('C', 'N', ns, nv, no, one, om[k], ns,
                                 x.data() + k * pair_dimension_, nv, zero, z.data(), nv);
                            gemm('N', 'N', ns, nt, nv, phase[k], z.data(), nv, on[k] + no * nt, nt,
                                 one, d00, nt);
                            gemm('N', 'T', nn * nt, ns, nv, phase[k], g.data(), nv, z.data(), nv,
                                 one, d01_raw, ns);
                            gemm('N', 'N', nn * ns, nv, no, one, wf.data(), no,
                                 x.data() + k * pair_dimension_, nv, zero, jx.data(), nv);
                            gemm('N', 'N', nn * ns, nt, nv, phase[k], jx.data(), nv,
                                 on[k] + no * nt, nt, one, d10, nt);
                            swap_aux_ao(jx.data(), jxt.data(), nn, ns, nv);
                            gemm('N', 'T', ns, nt, nn * nv, phase[k], jxt.data(), nn * nv,
                                 gt.data(), nn * nv, one, d11, nt);
                        }
                    }
#ifdef _OPENMP
#pragma omp master
#endif
                    {
                        const double start = MPI_Wtime();
                        MPI_Allreduce(MPI_IN_PLACE, packet.data(), static_cast<int>(fields * count),
                                      MPI_C_DOUBLE_COMPLEX, MPI_SUM, pair_comm);
                        communication_seconds += MPI_Wtime() - start;
                        collective_bytes += fields * count * sizeof(Complex);
                    }
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(static)
#endif
                    for (int row = 0; row < count; ++row)
                    {
                        auto *d00 = packet.data() + row * fields;
                        auto *d01_raw = d00 + ns * nt;
                        auto *d10 = d01_raw + nn * nt * ns;
                        auto *d11 = d10 + static_cast<std::size_t>(paired ? nm : nn) * ns * nt;
                        const auto &[cell, image_weight, w] = images[begin + row];
                        // A capped team may assign several input images to one thread.
                        // Recompute the matching output phase instead of reusing its last image.
                        for (const int k : lr_.k_indices)
                        {
                            double angle = 0.0;
                            for (int axis = 0; axis < 3; ++axis)
                                angle += lr_.kindex_map[k][axis] * cell[axis];
                            phase[k] = std::polar(1.0, 2.0 * PARAM.constants.pi * angle);
                        }
                        gemm('N', 'N', nm, nt * ns, nn, one, w, nn, d01_raw, nt * ns, zero,
                             d01.data(), nt * ns);
                        swap_aux_ao(d01.data(), d01t.data(), nm, nt, ns);
                        if (paired)
                        {
                            // d01 is free after its transpose; reuse it for raw D10 reordering.
                            swap_aux_ao(d10, d01.data(), ns, nm, nt);
                            gemm('T', 'N', nn, ns * nt, nm, one, w, nn, d01.data(), ns * nt,
                                 zero, d10_projected.data(), ns * nt);
                            swap_aux_ao(d10_projected.data(), d10t.data(), nn, ns, nt);
                        }
                        else
                            swap_aux_ao(d10, d10t.data(), nn, ns, nt);

                        for (std::size_t batch = 0, begin_k = 0;
                             begin_k < lr_.k1_indices.size(); ++batch, begin_k += projection_tile)
                        {
                            const int k_count = std::min(projection_tile, lr_.k1_indices.size() - begin_k);
                            const int nf = k_count * no, ng = k_count * nv;
                            gemm('N', 'N', nt, nf, nm * ns, one, d01t.data(), nm * ns,
                                 output_f[batch].data(), nf, zero, h01.data(), nf);
                            gemm('N', 'N', ns, ng, nn * nt, one, d10t.data(), nn * nt,
                                 output_g[batch].data(), ng, zero, h10.data(), ng);
                            for (int slot = 0; slot < k_count; ++slot)
                            {
                                const std::size_t owned = begin_k + slot;
                                const int k = lr_.k1_indices[owned];
                                const auto &ft = cs[k][2], &g = cs[k][5];
                                std::fill(result.begin(), result.end(), Complex{});
                                const Complex *weighted_g = nullptr;
                                if (paired)
                                    weighted_g = weighted_virtuals[row].data() + owned * nt * nm * nv;
                                else
                                {
                                    gemm('N', 'N', nm, nt * nv, nn, one, w, nn, g.data(), nt * nv, zero,
                                         wg.data(), nt * nv);
                                    swap_aux_ao(wg.data(), wgt.data(), nm, nt, nv);
                                    weighted_g = wgt.data();
                                }
                                gemm('T', 'N', nt, nm * no, ns, one, d00, nt, ft.data(), nm * no, zero,
                                     temp00.data(), nm * no);
                                gemm('T', 'N', no, nv, nt * nm, one, temp00.data(), no, weighted_g, nv,
                                     one, result.data(), nv);
                                gemm('T', 'C', no, nv, nt, one, h01.data() + slot * no, nf,
                                     on[k] + no * nt, nt, one, result.data(), nv);
                                gemm('N', 'N', no, nv, ns, one, om[k], ns,
                                     h10.data() + slot * nv, ng, one, result.data(), nv);
                                gemm('N', 'N', no, nt, ns, one, om[k], ns, d11, nt, zero, temp.data(),
                                     nt);
                                gemm('N', 'C', no, nv, nt, one, temp.data(), nt, on[k] + no * nt, nt,
                                     one, result.data(), nv);
                                const Complex factor = scale * image_weight * std::conj(phase[k]);
                                for (int j = 0; j < pair_dimension_; ++j)
                                    output[owned * pair_dimension_ + j] += factor * result[j];
                            }
                        }
                    }
                }
            }
            for (const auto &output : outputs)
                for (std::size_t owned = 0; owned < lr_.k1_indices.size(); ++owned)
                    for (int j = 0; j < pair_dimension_; ++j)
                        local[lr_.k1_indices[owned] * pair_dimension_ + j] +=
                            output[owned * pair_dimension_ + j];
            log_ << "LRI_KMPI_FIELD local_k=" << lr_.k_indices.size() << " images=" << images.size()
                 << " tile=" << tile << " packet_bytes=" << packet.size() * sizeof(Complex)
                 << " cached=" << paired << " cache_budget_bytes=" << cache_budget
                 << " collective_bytes=" << collective_bytes
                 << " collective_seconds=" << communication_seconds << std::endl;
        }
}

} // namespace

void MolecularLri::apply_tda(const std::vector<Complex> &x,
                             std::vector<Complex> &y,
                             const std::string &spin_type)
{
    apply_tda_impl(x, y, spin_type, options_.bse_haydock_apply == "fft");
}

void MolecularLri::apply_tda_real_space(const std::vector<Complex> &x,
                                        std::vector<Complex> &y,
                                        const std::string &spin_type)
{
    apply_tda_impl(x, y, spin_type, false);
}

void MolecularLri::apply_tda_fft(const std::vector<Complex> &x,
                                 std::vector<Complex> &y,
                                 const std::string &spin_type)
{
    apply_tda_impl(x, y, spin_type, true);
}

void MolecularLri::apply_tda_impl(const std::vector<Complex> &x,
                                       std::vector<Complex> &y,
                                       const std::string &spin_type, bool use_fft)
{
    if (use_fft && !FftBatch::available())
        throw std::runtime_error("FFT LRI requires a build with LIBBSE_ENABLE_FFTW=ON");
    const int dimension = nk_ * pair_dimension_;
    if (&x == &y || x.size() != static_cast<std::size_t>(dimension))
        throw std::invalid_argument("invalid or aliased LRI real-space input vector");
    for (const auto z : x)
        if (!std::isfinite(std::abs(z)))
            throw std::invalid_argument("nonfinite LRI real-space input vector");
    const auto weights = interaction_coefficients(spin_type);
    const int no = options_.nocc, nv = options_.nvirt, nb = no + nv;
    const double normalization = PARAM.constants.ha_to_ry / nk_;
    int rank = 0;
    MPI_Comm_rank(dataset_.comm_h.comm, &rank);
    const bool pair_k = uses_distributed_k();
    if (use_fft) require_cartesian_k_pairs();
    PairCommunicator pair_comm;
    if (pair_k)
    {
        const int pairs = static_cast<int>(dataset_.atoms.size() * dataset_.atoms.size());
        MPI_Comm_split(dataset_.comm_h.comm, rank % pairs, rank / pairs, &pair_comm.value);
    }
    std::vector<Complex> local(dimension, Complex{});
    auto local_memory = watch_memory("LibRI.apply_R_local", local);

    // Distributed-k owners sum the R densities within each atom pair before
    // local output projection; other layouts retain their original action.
    if (!lr_.k1_indices.empty() && !lr_.k2_indices.empty())
    {
        if (weights.hartree != 0.0)
        {
            std::map<int, std::map<int, RI::Tensor<Complex>>> phi;
            auto phi_memory = watch_memory("LibRI.apply_R_phi", phi);
            for (const int atom : lr_.list_IJ)
                for (const int k : lr_.k_indices)
                    phi[atom][k] = RI::Cs_ao_mo_to_Cs_mo(
                        lr_.Csk_ao_mo, lr_.map_psi, k, k, atom,
                        "O", "V", no, nv, false);
            const auto &bare = lr_.lrik.data_pool.at("Vs_").Ds_ab;
            for (const int nu : lr_.list_J)
            {
                const int nn = phi.at(nu).begin()->second.shape[0];
                std::vector<Complex> charge(nn, Complex{});
                for (const int k : lr_.k2_indices)
                {
                    const auto *p = phi.at(nu).at(k).ptr();
                    for (int n = 0; n < nn; ++n)
                        for (int j = 0; j < pair_dimension_; ++j)
                            charge[n] += std::conj(p[n * pair_dimension_ + j])
                                         * x[k * pair_dimension_ + j];
                }
                if (pair_k)
                    MPI_Allreduce(MPI_IN_PLACE, charge.data(), nn, MPI_C_DOUBLE_COMPLEX,
                                  MPI_SUM, pair_comm.value);
                for (const int mu : lr_.list_I)
                {
                    const int nm = phi.at(mu).begin()->second.shape[0];
                    std::vector<Complex> potential(nm, Complex{});
                    for (const auto &key : RI::Divide_Atoms::traversal_atom_period(
                             std::vector<int>{nu}, lr_.lrik.period))
                    {
                        const auto &v = RI::Global_Func::find(bare, mu, key);
                        if (v.empty()) continue;
                        if (v.shape[0] != static_cast<std::size_t>(nm)
                            || v.shape[1] != static_cast<std::size_t>(nn))
                            throw std::runtime_error("inconsistent Hartree auxiliary dimensions");
                        for (int m = 0; m < nm; ++m)
                            for (int n = 0; n < nn; ++n)
                                potential[m] += v.ptr()[m * nn + n] * charge[n];
                    }
                    for (const int k : lr_.k1_indices)
                    {
                        const auto *p = phi.at(mu).at(k).ptr();
                        for (int m = 0; m < nm; ++m)
                            for (int j = 0; j < pair_dimension_; ++j)
                                local[k * pair_dimension_ + j] += weights.hartree * normalization
                                    * p[m * pair_dimension_ + j] * potential[m];
                    }
                }
            }
        }

        if (weights.screened != 0.0)
        {
            if (use_fft)
                add_screened_fft(x, local, weights.screened * normalization);
            else if (pair_k)
                add_screened_kmpi(lr_, options_, nk_, pair_dimension_, x, local,
                                  weights.screened * normalization, pair_comm.value, tr_partner_slots_, log_);
            else
            {
            const auto &screened = lr_.lrik.data_pool.at("Ws_").Ds_ab;
            for (const int mu : lr_.list_I)
                for (const int nu : lr_.list_J)
                {
                    std::vector<const Complex *> om(nk_), on(nk_), fm(nk_), fn(nk_);
                    const auto &cm = lr_.Csk_ao_mo.at(lr_.k_indices.front()).at(mu);
                    const auto &cn = lr_.Csk_ao_mo.at(lr_.k_indices.front()).at(nu);
                    const int nm = cm.shape[0], nn = cn.shape[0];
                    const int ns = cm.shape[1], nt = cn.shape[1];
                    for (const int k : lr_.k_indices)
                    {
                        om[k] = lr_.map_psi.at(k).at(mu).ptr();
                        on[k] = lr_.map_psi.at(k).at(nu).ptr();
                        fm[k] = lr_.Csk_ao_mo.at(k).at(mu).ptr();
                        fn[k] = lr_.Csk_ao_mo.at(k).at(nu).ptr();
                    }
                    // Band slices are independent of R; pack them once per atom pair.
                    std::vector<std::array<std::vector<Complex>, 6>> cs(nk_);
                    for (const int k : lr_.k2_indices)
                    {
                        cs[k][1] = pack_cs(fm[k], nm, ns, nb, 0, no, true);
                        cs[k][3] = pack_cs(fn[k], nn, nt, nb, no, nv, false);
                        cs[k][4].resize(nt * nn * nv);
                        swap_aux_ao(cs[k][3].data(), cs[k][4].data(), nn, nt, nv);
                    }
                    for (const int k : lr_.k1_indices)
                    {
                        cs[k][0] = pack_cs(fm[k], nm, ns, nb, 0, no, false);
                        cs[k][2].resize(ns * nm * no);
                        swap_aux_ao(cs[k][0].data(), cs[k][2].data(), nm, ns, no);
                        cs[k][5] = pack_cs(fn[k], nn, nt, nb, no, nv, true);
                    }
                    auto cs_memory = watch_memory("LibRI.apply_R_packed_Cs", cs);
                    struct Image
                    {
                        std::array<int, 3> cell;
                        double weight;
                        const Complex *w;
                    };
                    std::vector<Image> images;
                    for (const auto &key : RI::Divide_Atoms::traversal_atom_period(
                             std::vector<int>{nu}, lr_.lrik.period))
                    {
                        const auto &w = RI::Global_Func::find(screened, mu, key);
                        if (w.empty()) continue;
                        if (w.shape[0] != static_cast<std::size_t>(nm)
                            || w.shape[1] != static_cast<std::size_t>(nn))
                            throw std::runtime_error("inconsistent screened auxiliary dimensions");
                        for (const auto &[cell, weight] : w_boundary_images(
                                 key.second, lr_.lrik.period, options_.bse_w_boundary_average))
                            images.push_back({cell, weight, w.ptr()});
                    }
                    if (images.empty()) continue;
                    int threads = 1;
#ifdef _OPENMP
                    threads = std::min<std::size_t>(omp_get_max_threads(), images.size());
#endif
                    // Each thread owns its densities and output; observe memory only outside OpenMP.
                    std::vector<std::array<std::vector<Complex>, 7>> buffers(threads);
                    std::exception_ptr allocation_error;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
                    for (int worker = 0; worker < threads; ++worker)
                    {
                        try
                        {
                            auto &buffer = buffers[worker];
                            buffer[0].resize(dimension);
                            buffer[1].resize(ns * nt);
                            buffer[2].resize(nm * nt * ns);
                            buffer[3].resize(nn * ns * nt);
                            buffer[4].resize(ns * nt);
                            buffer[5].resize(nt * nm * ns);
                            buffer[6].resize(ns * nn * nt);
                        }
                        catch (...)
                        {
#ifdef _OPENMP
#pragma omp critical (libbse_R_buffer_allocation)
#endif
                            {
                                if (!allocation_error) allocation_error = std::current_exception();
                            }
                        }
                    }
                    if (allocation_error) std::rethrow_exception(allocation_error);
                    auto buffers_memory = watch_memory("LibRI.apply_R_threads", buffers);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
                    {
                        int thread = 0;
#ifdef _OPENMP
                        thread = omp_get_thread_num();
#endif
                        // Reuse scratch across R images and k points on each thread.
                        std::vector<Complex> phase(nk_), wf(nn * ns * no),
                            wg(nm * nt * nv), z(ns * nv), jx(nn * ns * nv),
                            jxt(ns * nn * nv), wgt(nt * nm * nv), result(pair_dimension_),
                            temp00(nt * nm * no), temp10(no * nn * nt),
                            temp(no * nt), h01(nt * no);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (std::size_t image = 0; image < images.size(); ++image)
                    {
                        auto &output = buffers[thread][0];
                        auto &d00 = buffers[thread][1], &d01 = buffers[thread][2],
                             &d10 = buffers[thread][3], &d11 = buffers[thread][4];
                        auto &d01t = buffers[thread][5], &d10t = buffers[thread][6];
                        // d10t is not needed until all densities have been accumulated.
                        auto &d01_raw = d10t;
                        for (const int field : {1, 3, 4, 6})
                            std::fill(buffers[thread][field].begin(), buffers[thread][field].end(), Complex{});
                        const auto &[cell, image_weight, w] = images[image];
                        for (const int k : lr_.k_indices)
                        {
                            double angle = 0.0;
                            for (int axis = 0; axis < 3; ++axis)
                                angle += lr_.kindex_map[k][axis] * cell[axis];
                            phase[k] = std::polar(1.0, 2.0 * PARAM.constants.pi * angle);
                        }
                        using RI::Blas_Interface::gemm;
                        const Complex one(1.0), zero{};
                        // d01 permits moving W outside the k sum; the other
                        // terms retain the band-factor contractions.
                        for (const int k : lr_.k2_indices)
                        {
                            const auto &f = cs[k][1], &g = cs[k][3], &gt = cs[k][4];
                            gemm('T','N', nn,ns*no,nm, one,w,nn,
                                 f.data(),ns*no, zero,wf.data(),ns*no);
                            gemm('C','N', ns,nv,no, one,om[k],ns,
                                 x.data()+k*pair_dimension_,nv, zero,z.data(),nv);
                            gemm('N','N', ns,nt,nv, phase[k],z.data(),nv,
                                 on[k]+no*nt,nt, one,d00.data(),nt);
                            gemm('N','T', nn*nt,ns,nv, phase[k],g.data(),nv,
                                 z.data(),nv, one,d01_raw.data(),ns);
                            gemm('N','N', nn*ns,nv,no, one,wf.data(),no,
                                 x.data()+k*pair_dimension_,nv, zero,jx.data(),nv);
                            gemm('N','N', nn*ns,nt,nv, phase[k],jx.data(),nv,
                                 on[k]+no*nt,nt, one,d10.data(),nt);
                            swap_aux_ao(jx.data(), jxt.data(), nn, ns, nv);
                            gemm('N','T', ns,nt,nn*nv, phase[k],jxt.data(),nn*nv,
                                 gt.data(),nn*nv, one,d11.data(),nt);
                        }
                        gemm('N','N', nm,nt*ns,nn, one,w,nn,
                             d01_raw.data(),nt*ns, zero,d01.data(),nt*ns);
                        swap_aux_ao(d01.data(), d01t.data(), nm, nt, ns);
                        swap_aux_ao(d10.data(), d10t.data(), nn, ns, nt);
                        for (const int k : lr_.k1_indices)
                        {
                            const auto &f = cs[k][0], &ft = cs[k][2], &g = cs[k][5];
                            std::fill(result.begin(), result.end(), Complex{});
                            gemm('N','N', nm,nt*nv,nn, one,w,nn,
                                 g.data(),nt*nv, zero,wg.data(),nt*nv);
                            swap_aux_ao(wg.data(), wgt.data(), nm, nt, nv);
                            gemm('T','N', nt,nm*no,ns, one,d00.data(),nt,
                                 ft.data(),nm*no, zero,temp00.data(),nm*no);
                            gemm('T','N', no,nv,nt*nm, one,temp00.data(),no,
                                 wgt.data(),nv, one,result.data(),nv);
                            gemm('N','N', nt,no,nm*ns, one,d01t.data(),nm*ns,
                                 f.data(),no, zero,h01.data(),no);
                            gemm('T','C', no,nv,nt, one,h01.data(),no,
                                 on[k]+no*nt,nt, one,result.data(),nv);
                            gemm('N','N', no,nn*nt,ns, one,om[k],ns,
                                 d10t.data(),nn*nt, zero,temp10.data(),nn*nt);
                            gemm('N','N', no,nv,nn*nt, one,temp10.data(),nn*nt,
                                 g.data(),nv, one,result.data(),nv);
                            gemm('N','N', no,nt,ns, one,om[k],ns,
                                 d11.data(),nt, zero,temp.data(),nt);
                            gemm('N','C', no,nv,nt, one,temp.data(),nt,
                                 on[k]+no*nt,nt, one,result.data(),nv);
                            const Complex factor = weights.screened * normalization
                                * image_weight * std::conj(phase[k]);
                            for (int j = 0; j < pair_dimension_; ++j)
                                output[k * pair_dimension_ + j] += factor * result[j];
                        }
                    }
                    }
                    for (const auto &buffer : buffers)
                        for (int pair = 0; pair < dimension; ++pair)
                            local[pair] += buffer[0][pair];
                }
            }
        }
    }
    y.resize(dimension);
    MPI_Allreduce(local.data(), y.data(), dimension, MPI_C_DOUBLE_COMPLEX,
                  MPI_SUM, dataset_.comm_h.comm);
    for (int k = 0; k < nk_; ++k)
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
            {
                const int pair = k * pair_dimension_ + i * nv + a;
                const auto offset = static_cast<std::size_t>(k) * qp_.nbands;
                y[pair] += (qp_.energies_ry[offset + no + a] - qp_.energies_ry[offset + i]) * x[pair];
            }
}

} // namespace libbse
