#include "molecular_lri.h"
#include "w_boundary_phase.h"
#include "utils/memory_views.h"

#include <cmath>
#include <stdexcept>

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
    std::vector<Complex> local(dimension, Complex{});
    auto local_memory = watch_memory("LibRI.apply_R_local", local);

    // Keep the original Cartesian atom/k partitions; each rank performs a
    // partial k2->R transform and R->k1 projection before the final reduction.
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
                    for (const auto &key : RI::Divide_Atoms::traversal_atom_period(
                             std::vector<int>{nu}, lr_.lrik.period))
                    {
                        const auto &w = RI::Global_Func::find(screened, mu, key);
                        if (w.empty()) continue;
                        if (w.shape[0] != static_cast<std::size_t>(nm)
                            || w.shape[1] != static_cast<std::size_t>(nn))
                            throw std::runtime_error("inconsistent screened auxiliary dimensions");
                        for (const auto &[cell, image_weight] : w_boundary_images(
                                 key.second, lr_.lrik.period, options_.bse_w_boundary_average))
                        {
                            std::vector<Complex> phase(nk_);
                            for (const int k : lr_.k_indices)
                            {
                                double angle = 0.0;
                                for (int axis = 0; axis < 3; ++axis)
                                    angle += lr_.kindex_map[k][axis] * cell[axis];
                                phase[k] = std::polar(1.0, 2.0 * PARAM.constants.pi * angle);
                            }
                            using RI::Blas_Interface::gemm;
                            const Complex one(1.0), zero{};
                            // Four generalized AO densities. Contract W before
                            // the k sums to avoid an aux^2 * AO^2 inner loop.
                            std::vector<Complex> d00(ns * nt), d01(nm * nt * ns),
                                                 d10(nn * ns * nt), d11(ns * nt);
                            auto d00_memory = watch_memory("LibRI.apply_R_d00", d00);
                            auto d01_memory = watch_memory("LibRI.apply_R_d01", d01);
                            auto d10_memory = watch_memory("LibRI.apply_R_d10", d10);
                            auto d11_memory = watch_memory("LibRI.apply_R_d11", d11);
                            for (const int k : lr_.k2_indices)
                            {
                                auto f = pack_cs(fm[k], nm, ns, nb, 0, no, true);
                                auto g = pack_cs(fn[k], nn, nt, nb, no, nv, false);
                                std::vector<Complex> wf(nn * ns * no), wg(nm * nt * nv),
                                                     z(ns * nv), jx(ns * nv);
                                gemm('T','N', nn,ns*no,nm, one,w.ptr(),nn,
                                     f.data(),ns*no, zero,wf.data(),ns*no);
                                gemm('N','N', nm,nt*nv,nn, one,w.ptr(),nn,
                                     g.data(),nt*nv, zero,wg.data(),nt*nv);
                                gemm('C','N', ns,nv,no, one,om[k],ns,
                                     x.data()+k*pair_dimension_,nv, zero,z.data(),nv);
                                gemm('N','N', ns,nt,nv, phase[k],z.data(),nv,
                                     on[k]+no*nt,nt, one,d00.data(),nt);
                                gemm('N','T', nm*nt,ns,nv, phase[k],wg.data(),nv,
                                     z.data(),nv, one,d01.data(),ns);
                                for (int n = 0; n < nn; ++n)
                                {
                                    gemm('N','N', ns,nv,no, one,wf.data()+n*ns*no,no,
                                         x.data()+k*pair_dimension_,nv, zero,jx.data(),nv);
                                    gemm('N','N', ns,nt,nv, phase[k],jx.data(),nv,
                                         on[k]+no*nt,nt, one,d10.data()+n*ns*nt,nt);
                                    gemm('N','T', ns,nt,nv, phase[k],jx.data(),nv,
                                         g.data()+n*nt*nv,nv, one,d11.data(),nt);
                                }
                            }
                            for (const int k : lr_.k1_indices)
                            {
                                auto g = pack_cs(fn[k], nn, nt, nb, no, nv, true);
                                std::vector<Complex> wg(nm * nt * nv), result(pair_dimension_),
                                                     temp(no * nt), h01(nt * no);
                                gemm('N','N', nm,nt*nv,nn, one,w.ptr(),nn,
                                     g.data(),nt*nv, zero,wg.data(),nt*nv);
                                for (int m = 0; m < nm; ++m)
                                {
                                    gemm('T','N', no,nt,ns, one,fm[k]+m*ns*nb,nb,
                                         d00.data(),nt, zero,temp.data(),nt);
                                    gemm('N','N', no,nv,nt, one,temp.data(),nt,
                                         wg.data()+m*nt*nv,nv, one,result.data(),nv);
                                    gemm('N','N', nt,no,ns, one,d01.data()+m*nt*ns,ns,
                                         fm[k]+m*ns*nb,nb, one,h01.data(),no);
                                }
                                gemm('T','C', no,nv,nt, one,h01.data(),no,
                                     on[k]+no*nt,nt, one,result.data(),nv);
                                for (int n = 0; n < nn; ++n)
                                {
                                    gemm('N','N', no,nt,ns, one,om[k],ns,
                                         d10.data()+n*ns*nt,nt, zero,temp.data(),nt);
                                    gemm('N','N', no,nv,nt, one,temp.data(),nt,
                                         g.data()+n*nt*nv,nv, one,result.data(),nv);
                                }
                                gemm('N','N', no,nt,ns, one,om[k],ns,
                                     d11.data(),nt, zero,temp.data(),nt);
                                gemm('N','C', no,nv,nt, one,temp.data(),nt,
                                     on[k]+no*nt,nt, one,result.data(),nv);
                                const Complex factor = weights.screened * normalization
                                    * image_weight * std::conj(phase[k]);
                                for (int j = 0; j < pair_dimension_; ++j)
                                    local[k * pair_dimension_ + j] += factor * result[j];
                            }
                        }
                    }
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
