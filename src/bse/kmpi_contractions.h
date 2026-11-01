// Author: Ruiyi Zhou.
#pragma once

#include "bse_types.h"
#include <RI/global/Blas_Interface.h>

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace libbse
{

inline bool kmpi_cache_fits(std::size_t bytes, std::size_t nk, std::size_t nm,
                           std::size_t nn, std::size_t ns, std::size_t nt,
                           std::size_t nv, std::size_t tile,
                           std::size_t gauge_pairs = 0)
{
    if (bytes == 0) return false;
    std::size_t remaining = bytes / sizeof(Complex);
    const auto reserve = [&](std::initializer_list<std::size_t> factors) {
        std::size_t count = 1;
        for (const auto factor : factors)
        {
            if (factor != 0 && count > remaining / factor) return false;
            count *= factor;
        }
        remaining -= count;
        return true;
    };
    // Split real/imaginary G occupies the same payload as complex G.
    return reserve({tile, nk, nt, nm, nv}) && reserve({nk, nn, nt, nv})
        && reserve({nk, ns, nv, nm}) && reserve({nk, ns, nv})
        && reserve({gauge_pairs, nv, nv});
}

// W=A+iB, G=C+iD: one real GEMM supplies both WG and W*conj(G).
inline void paired_virtual_products(int nm, int nn, int nt, int nv,
    const std::vector<double> &stacked_w, const std::vector<double> &paired_g,
    std::vector<double> &products, Complex *input, Complex *output)
{
    const int columns = nt * nv;
    RI::Blas_Interface::gemm('N', 'N', 2 * nm, 2 * columns, nn, 1.,
        stacked_w.data(), nn, paired_g.data(), 2 * columns, 0., products.data(), 2 * columns);
    for (int m = 0; m < nm; ++m)
        for (int t = 0; t < nt; ++t)
            for (int a = 0; a < nv; ++a)
            {
                const auto col = t * nv + a;
                const auto i = static_cast<std::size_t>(m) * 2 * columns + col;
                const auto j = static_cast<std::size_t>(m + nm) * 2 * columns + col;
                const auto target = (static_cast<std::size_t>(t) * nm + m) * nv + a;
                const double p = products[i], u = products[i + columns],
                             v = products[j], q = products[j + columns];
                input[target] = Complex(p - q, u + v);
                output[target] = Complex(p + q, v - u);
            }
}

} // namespace libbse
