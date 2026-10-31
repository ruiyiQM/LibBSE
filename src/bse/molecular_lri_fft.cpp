#include "molecular_lri.h"
#include "w_boundary_phase.h"
#include "utils/memory_views.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace libbse
{

void MolecularLri::add_screened_fft(const std::vector<Complex> &x,
                                    std::vector<Complex> &local, double factor)
{
    if (!fft_grid_) fft_grid_ = std::make_unique<UniformFftGrid>(lr_.kindex_map);
    const auto &grid = *fft_grid_;
    const auto &screened = lr_.lrik.data_pool.at("Ws_").Ds_ab;
    const int no = options_.nocc, nv = options_.nvirt, nb = no + nv;
    using RI::Blas_Interface::gemm;
    const Complex one(1.0), zero{};

    for (const int mu : lr_.list_I)
        for (const int nu : lr_.list_J)
        {
            const auto &cm = lr_.Csk_ao_mo.at(lr_.k_indices.front()).at(mu);
            const auto &cn = lr_.Csk_ao_mo.at(lr_.k_indices.front()).at(nu);
            const int nm = cm.shape[0], nn = cn.shape[0];
            const int ns = cm.shape[1], nt = cn.shape[1];
            if (ns <= 0 || nt <= 0 || ns > std::numeric_limits<int>::max() / nt)
                throw std::overflow_error("FFT AO field count overflow");
            const int fields = ns * nt;
            auto &plan = fft_plans_[fields];
            if (!plan) plan = std::make_unique<FftBatch>(grid.shape, fields);

            struct Image
            {
                int site;
                double weight;
                const Complex *values;
            };
            std::vector<Image> images;
            for (const auto &key : RI::Divide_Atoms::traversal_atom_period(
                     std::vector<int>{nu}, lr_.lrik.period))
            {
                const auto &w = RI::Global_Func::find(screened, mu, key);
                if (w.empty()) continue;
                if (w.shape[0] != static_cast<std::size_t>(nm)
                    || w.shape[1] != static_cast<std::size_t>(nn))
                    throw std::runtime_error("inconsistent FFT auxiliary dimensions");
                for (const auto &[cell, weight] : w_boundary_images(
                         key.second, lr_.lrik.period, options_.bse_w_boundary_average))
                    images.push_back({grid.index(cell), weight, w.ptr()});
            }
            if (images.empty()) continue;

            std::vector<const Complex *> om(nk_), on(nk_), fm(nk_), fn(nk_);
            for (const int k : lr_.k_indices)
            {
                om[k] = lr_.map_psi.at(k).at(mu).ptr();
                on[k] = lr_.map_psi.at(k).at(nu).ptr();
                fm[k] = lr_.Csk_ao_mo.at(k).at(mu).ptr();
                fn[k] = lr_.Csk_ao_mo.at(k).at(nu).ptr();
            }
            const std::size_t density_size = static_cast<std::size_t>(nk_) * fields;
            std::vector<Complex> z(static_cast<std::size_t>(nk_) * ns * nv), zm(z.size());
            std::vector<Complex> d00(density_size), d10(density_size),
                                 c00(density_size), c01(density_size),
                                 c10(density_size), c11(density_size), kernel(nk_);
            auto z_memory = watch_memory("LibRI.fft_z", z);
            auto zm_memory = watch_memory("LibRI.fft_zm", zm);
            auto d00_memory = watch_memory("LibRI.fft_d00", d00);
            auto d10_memory = watch_memory("LibRI.fft_d10", d10);
            auto c00_memory = watch_memory("LibRI.fft_c00", c00);
            auto c01_memory = watch_memory("LibRI.fft_c01", c01);
            auto c10_memory = watch_memory("LibRI.fft_c10", c10);
            auto c11_memory = watch_memory("LibRI.fft_c11", c11);
            auto kernel_memory = watch_memory("LibRI.fft_kernel", kernel);

            // Missing local k2 entries stay zero. The common k-mesh shift
            // cancels in k2-k1, so only canonical integer grid indices enter.
            for (const int k : lr_.k2_indices)
            {
                auto *zk = z.data() + static_cast<std::size_t>(k) * ns * nv;
                auto *dk = d00.data() + static_cast<std::size_t>(grid.k_to_grid[k]) * fields;
                gemm('C','N',ns,nv,no, one,om[k],ns,
                     x.data()+k*pair_dimension_,nv, zero,zk,nv);
                gemm('N','N',ns,nt,nv, one,zk,nv,on[k]+no*nt,nt, zero,dk,nt);
            }
            plan->backward(d00);

            std::vector<Complex> occupied(ns * no), virtual_bra(nt * nv),
                                 t0(no * nt), t1(no * nt), result(pair_dimension_);
            for (int m = 0; m < nm; ++m)
            {
                std::fill(d10.begin(), d10.end(), zero);
                for (const int k : lr_.k2_indices)
                {
                    for (int s = 0; s < ns; ++s)
                        for (int i = 0; i < no; ++i)
                            occupied[s * no + i] = std::conj(fm[k][(m * ns + s) * nb + i]);
                    auto *zk = zm.data() + static_cast<std::size_t>(k) * ns * nv;
                    auto *dk = d10.data() + static_cast<std::size_t>(grid.k_to_grid[k]) * fields;
                    gemm('N','N',ns,nv,no, one,occupied.data(),no,
                         x.data()+k*pair_dimension_,nv, zero,zk,nv);
                    gemm('N','N',ns,nt,nv, one,zk,nv,on[k]+no*nt,nt, zero,dk,nt);
                }
                plan->backward(d10);
                // Stream auxiliary pairs; the number of FFT fields is only
                // ns*nt, not nm*nn*ns*nt. No auxiliary-rank approximation.
                for (int n = 0; n < nn; ++n)
                {
                    std::fill(kernel.begin(), kernel.end(), zero);
                    for (const auto &image : images)
                        kernel[image.site] += image.weight * image.values[m * nn + n];
                    if (std::all_of(kernel.begin(), kernel.end(),
                                    [](Complex value) { return value == Complex{}; }))
                        continue;
                    c00 = d00;
                    c10 = d10;
                    std::fill(c01.begin(), c01.end(), zero);
                    std::fill(c11.begin(), c11.end(), zero);
                    for (const int k : lr_.k2_indices)
                    {
                        const auto offset = static_cast<std::size_t>(grid.k_to_grid[k]) * fields;
                        const auto zoffset = static_cast<std::size_t>(k) * ns * nv;
                        const auto *g = fn[k] + n * nt * nb + no;
                        gemm('N','T',ns,nt,nv, one,z.data()+zoffset,nv,
                             g,nb, zero,c01.data()+offset,nt);
                        gemm('N','T',ns,nt,nv, one,zm.data()+zoffset,nv,
                             g,nb, zero,c11.data()+offset,nt);
                    }
                    plan->backward(c01);
                    plan->backward(c11);
                    for (int r = 0; r < nk_; ++r)
                        for (int field = 0; field < fields; ++field)
                        {
                            const auto offset = static_cast<std::size_t>(r) * fields + field;
                            c00[offset] *= kernel[r];
                            c01[offset] *= kernel[r];
                            c10[offset] *= kernel[r];
                            c11[offset] *= kernel[r];
                        }
                    plan->forward(c00);
                    plan->forward(c01);
                    plan->forward(c10);
                    plan->forward(c11);
                    for (const int k : lr_.k1_indices)
                    {
                        const auto offset = static_cast<std::size_t>(grid.k_to_grid[k]) * fields;
                        const auto *f = fm[k] + m * ns * nb;
                        for (int t = 0; t < nt; ++t)
                            for (int a = 0; a < nv; ++a)
                                virtual_bra[t * nv + a] = std::conj(fn[k][(n * nt + t) * nb + no + a]);
                        gemm('T','N',no,nt,ns, one,f,nb,c00.data()+offset,nt, zero,t0.data(),nt);
                        gemm('N','N',no,nt,ns, one,om[k],ns,c10.data()+offset,nt, one,t0.data(),nt);
                        gemm('T','N',no,nt,ns, one,f,nb,c01.data()+offset,nt, zero,t1.data(),nt);
                        gemm('N','N',no,nt,ns, one,om[k],ns,c11.data()+offset,nt, one,t1.data(),nt);
                        gemm('N','N',no,nv,nt, one,t0.data(),nt,virtual_bra.data(),nv, zero,result.data(),nv);
                        gemm('N','C',no,nv,nt, one,t1.data(),nt,on[k]+no*nt,nt, one,result.data(),nv);
                        for (int pair = 0; pair < pair_dimension_; ++pair)
                            local[k * pair_dimension_ + pair] += factor * result[pair];
                    }
                }
            }
        }
}

} // namespace libbse
