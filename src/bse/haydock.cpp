#include "haydock.h"
#include "utils/memory.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace libbse
{
namespace
{
using Complex = std::complex<double>;

Complex dot(const std::vector<Complex> &a, const std::vector<Complex> &b)
{
    Complex value{};
    for (std::size_t i = 0; i < a.size(); ++i) value += std::conj(a[i]) * b[i];
    return value;
}

double norm_squared(const std::vector<Complex> &x)
{
    double value = 0.0;
    for (const auto z : x) value += std::norm(z);
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite Haydock vector");
    return value;
}
} // namespace

HaydockChain haydock_recursion(const TdaApply &apply,
                              const std::vector<Complex> &seed, int max_steps)
{
    if (seed.empty() || max_steps <= 0)
        throw std::invalid_argument("Haydock requires a nonempty seed and positive step count");
    HaydockChain chain;
    chain.seed_norm_squared = norm_squared(seed);
    if (chain.seed_norm_squared == 0.0)
    {
        chain.terminated = true;
        return chain;
    }
    std::vector<std::vector<Complex>> basis;
    auto basis_memory = watch_memory("haydock.krylov_basis", basis);
    std::vector<Complex> q = seed;
    auto q_memory = watch_memory("haydock.vector", q);
    for (auto &z : q) z /= std::sqrt(chain.seed_norm_squared);
    const int steps = std::min<std::size_t>(max_steps, seed.size());
    for (int step = 0; step < steps; ++step)
    {
        basis.push_back(q);
        std::vector<Complex> hq;
        auto hq_memory = watch_memory("haydock.operator_result", hq);
        apply(q, hq);
        if (hq.size() != seed.size())
            throw std::runtime_error("Haydock operator returned the wrong dimension");
        const double scale = std::max(1.0, std::sqrt(norm_squared(hq)));
        const Complex alpha = dot(q, hq);
        if (std::abs(alpha.imag()) > 1.e-10 * scale)
            throw std::runtime_error("Haydock requires a Hermitian TDA operator");
        chain.alpha.push_back(alpha.real());
        for (std::size_t i = 0; i < q.size(); ++i)
        {
            hq[i] -= alpha.real() * q[i];
            if (step) hq[i] -= chain.beta[step - 1] * basis[step - 1][i];
        }
        for (int pass = 0; pass < 2; ++pass)
            for (const auto &previous : basis)
            {
                const Complex overlap = dot(previous, hq);
                for (std::size_t i = 0; i < q.size(); ++i) hq[i] -= overlap * previous[i];
            }
        const double beta = std::sqrt(norm_squared(hq));
        chain.beta.push_back(beta);
        if (beta <= 1.e-13 * scale)
        {
            chain.terminated = true;
            break;
        }
        for (std::size_t i = 0; i < q.size(); ++i) q[i] = hq[i] / beta;
    }
    return chain;
}

Complex haydock_resolvent(const HaydockChain &chain, Complex energy)
{
    if (energy.imag() <= 0.0 || !std::isfinite(std::abs(energy))
        || chain.alpha.size() != chain.beta.size())
        throw std::invalid_argument("invalid Haydock resolvent arguments");
    Complex tail{};
    for (std::size_t j = chain.alpha.size(); j-- > 0;)
        tail = 1.0 / (energy - chain.alpha[j] - chain.beta[j] * chain.beta[j] * tail);
    return chain.seed_norm_squared * tail;
}

} // namespace libbse
