// Author: Ruiyi Zhou.
#include "bse/haydock.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using Complex = std::complex<double>;

int main()
{
    try
    {
        constexpr int n = 4;
        const std::array<double, n> energies{0.2, 0.4, 0.4, 1.1};
        std::array<std::array<Complex, n>, n> u{}, h{};
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u[i][j] = std::polar(0.5, 2.0 * std::acos(-1.0) * i * j / n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int s = 0; s < n; ++s)
                    h[i][j] += u[i][s] * energies[s] * std::conj(u[j][s]);
        const libbse::TdaApply apply = [&](const auto &x, auto &y) {
            y.assign(n, Complex{});
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) y[i] += h[i][j] * x[j];
        };
        const std::vector<Complex> seed{{1.1, 0.3}, {-0.7, 0.4}, {0.2, 0.8}, {0.6, -0.9}};
        const auto chain = libbse::haydock_recursion(apply, seed, n);
        if (!chain.terminated || chain.alpha.size() != 3)
            throw std::runtime_error("degenerate invariant subspace was not detected");
        for (int grid = 0; grid < 80; ++grid)
        {
            const Complex z(0.02 * grid, 0.015);
            Complex exact{};
            for (int s = 0; s < n; ++s)
            {
                Complex overlap{};
                for (int i = 0; i < n; ++i) overlap += std::conj(u[i][s]) * seed[i];
                exact += std::norm(overlap) / (z - energies[s]);
            }
            if (std::abs(libbse::haydock_resolvent(chain, z) - exact) > 2.e-11)
                throw std::runtime_error("complex Hermitian resolvent differs from spectral decomposition");
        }
        const auto zero = libbse::haydock_recursion(apply, std::vector<Complex>(n), n);
        if (!zero.terminated || libbse::haydock_resolvent(zero, {0.5, 0.1}) != Complex{})
            throw std::runtime_error("zero optical seed was not preserved");
        const auto short_chain = libbse::haydock_recursion(apply, seed, 1);
        if (short_chain.terminated || short_chain.alpha.size() != 1)
            throw std::runtime_error("iteration limit was ignored");
        if (libbse::haydock_gagq_chain(short_chain).alpha != short_chain.alpha
            || !libbse::haydock_gagq_chain(zero).alpha.empty())
            throw std::runtime_error("GAGQ changed an empty or one-step chain");
        const auto exhausted_quadrature = libbse::haydock_gagq_chain(chain);
        for (int grid = 0; grid < 80; ++grid)
        {
            const Complex z(0.02 * grid, 0.015);
            if (std::abs(libbse::haydock_resolvent(exhausted_quadrature, z)
                         - libbse::haydock_resolvent(chain, z)) > 2.e-11)
                throw std::runtime_error("GAGQ changed an exhausted invariant subspace");
        }
        const std::vector<double> diagonal{4., 5., 6., 7., 8., 9.};
        const std::vector<double> offdiagonal{0.4, 0.5, 0.6, 0.7, 0.8, 0.};
        const auto tridiagonal_apply = [&](const auto &x, auto &y) {
            y.resize(x.size());
            for (std::size_t j = 0; j < x.size(); ++j)
            {
                y[j] = diagonal[j] * x[j];
                if (j) y[j] += offdiagonal[j - 1] * x[j - 1];
                if (j + 1 < x.size()) y[j] += offdiagonal[j] * x[j + 1];
            }
        };
        std::vector<Complex> e1(diagonal.size());
        e1[0] = 1.;
        int calls = 0;
        const auto three_steps = libbse::haydock_recursion(
            [&](const auto &x, auto &y) { ++calls; tridiagonal_apply(x, y); }, e1, 3);
        const auto gagq = libbse::haydock_gagq_chain(three_steps);
        if (calls != 3 || gagq.alpha.size() != 5 || gagq.beta.size() != 5
            || std::abs(gagq.beta[2] - 0.6) > 1.e-14 || gagq.beta.back() != 0.)
            throw std::runtime_error("GAGQ lost the final residual or required extra Hx");
        std::vector<Complex> exact_moment = e1, quadrature_moment(gagq.alpha.size());
        quadrature_moment[0] = 1.;
        for (int degree = 0; degree <= 6; ++degree)
        {
            if (std::abs(exact_moment[0] - quadrature_moment[0])
                > 1.e-12 * std::max(1., std::abs(exact_moment[0])))
                throw std::runtime_error("GAGQ failed degree-2k moment matching");
            std::vector<Complex> next;
            tridiagonal_apply(exact_moment, next);
            exact_moment = next;
            next.assign(quadrature_moment.size(), Complex{});
            for (std::size_t j = 0; j < next.size(); ++j)
            {
                next[j] = gagq.alpha[j] * quadrature_moment[j];
                if (j) next[j] += gagq.beta[j - 1] * quadrature_moment[j - 1];
                if (j + 1 < next.size()) next[j] += gagq.beta[j] * quadrature_moment[j + 1];
            }
            quadrature_moment = next;
        }
        bool rejected_nonpositive = false;
        try {
            libbse::haydock_gagq_chain({1., {1., 1.}, {0.8, 1.}, false});
        } catch (const std::domain_error &) { rejected_nonpositive = true; }
        if (!rejected_nonpositive)
            throw std::runtime_error("GAGQ accepted a nonpositive quadrature node");
        bool rejected_invalid = false;
        try {
            libbse::haydock_gagq_chain({1., {1., 1.}, {0.8}, false});
        } catch (const std::invalid_argument &) { rejected_invalid = true; }
        if (!rejected_invalid) throw std::runtime_error("GAGQ accepted a malformed chain");
        bool rejected = false;
        try {
            libbse::haydock_recursion([](const auto &x, auto &y) {
                y = x;
                for (auto &v : y) v *= Complex(1, 1);
            }, seed, n);
        } catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("non-Hermitian operator was accepted");
        rejected = false;
        try {
            libbse::haydock_recursion(apply, {Complex(std::numeric_limits<double>::quiet_NaN())}, 1);
        } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("nonfinite optical seed was accepted");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
