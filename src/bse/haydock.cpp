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

Complex dot(const std::vector<Complex> &a, const std::vector<Complex> &b, MPI_Comm comm)
{
    Complex value{};
    for (std::size_t i = 0; i < a.size(); ++i) value += std::conj(a[i]) * b[i];
    if (comm != MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
    return value;
}

double norm_squared(const std::vector<Complex> &x, MPI_Comm comm)
{
    double value = 0.0;
    for (const auto z : x) value += std::norm(z);
    if (comm != MPI_COMM_NULL)
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite Haydock vector");
    return value;
}
} // namespace

HaydockChain haydock_recursion(const TdaApply &apply,
                              const std::vector<Complex> &seed, int max_steps)
{
    return haydock_recursion(apply, seed, max_steps, MPI_COMM_NULL);
}

HaydockChain haydock_recursion(const TdaApply &apply,
                              const std::vector<Complex> &seed, int max_steps,
                              MPI_Comm comm)
{
    unsigned long long dimension = seed.size();
    int minimum_steps = max_steps, maximum_steps = max_steps;
    if (comm != MPI_COMM_NULL)
    {
        MPI_Allreduce(MPI_IN_PLACE, &dimension, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_steps, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_steps, 1, MPI_INT, MPI_MAX, comm);
    }
    if (dimension == 0 || minimum_steps <= 0 || minimum_steps != maximum_steps)
        throw std::invalid_argument("Haydock requires a nonempty global seed and consistent positive step count");
    HaydockChain chain;
    chain.seed_norm_squared = norm_squared(seed, comm);
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
    const int steps = std::min<unsigned long long>(max_steps, dimension);
    for (int step = 0; step < steps; ++step)
    {
        basis.push_back(q);
        std::vector<Complex> hq;
        auto hq_memory = watch_memory("haydock.operator_result", hq);
        apply(q, hq);
        int valid = hq.size() == seed.size();
        if (comm != MPI_COMM_NULL)
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
        if (!valid)
            throw std::runtime_error("Haydock operator returned the wrong dimension");
        const double scale = std::max(1.0, std::sqrt(norm_squared(hq, comm)));
        const Complex alpha = dot(q, hq, comm);
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
                const Complex overlap = dot(previous, hq, comm);
                for (std::size_t i = 0; i < q.size(); ++i) hq[i] -= overlap * previous[i];
            }
        const double beta = std::sqrt(norm_squared(hq, comm));
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

HaydockChain haydock_gagq_chain(const HaydockChain &chain)
{
    if (chain.alpha.size() != chain.beta.size()
        || !std::isfinite(chain.seed_norm_squared) || chain.seed_norm_squared < 0.0)
        throw std::invalid_argument("invalid GAGQ chain");
    for (std::size_t j = 0; j < chain.alpha.size(); ++j)
        if (!std::isfinite(chain.alpha[j]) || !std::isfinite(chain.beta[j])
            || chain.beta[j] < 0.0)
            throw std::invalid_argument("invalid GAGQ coefficient");
    HaydockChain quadrature = chain;
    const std::size_t k = chain.alpha.size();
    if (k < 2) return quadrature;
    // Shao et al., SIAM J. Matrix Anal. Appl. 39 (2018), Eq. (14).
    for (std::size_t j = k - 1; j-- > 0;)
        quadrature.alpha.push_back(chain.alpha[j]);
    for (std::size_t j = k - 2; j-- > 0;)
        quadrature.beta.push_back(chain.beta[j]);
    quadrature.beta.push_back(0.0);
    // An LDL^T factorization detects nonpositive quadrature nodes in O(k).
    double pivot = quadrature.alpha[0];
    for (std::size_t j = 0; j < quadrature.alpha.size(); ++j)
    {
        if (j) pivot = quadrature.alpha[j]
                       - quadrature.beta[j - 1] * quadrature.beta[j - 1] / pivot;
        if (!std::isfinite(pivot) || pivot <= 0.0)
            throw std::domain_error("GAGQ extension has nonpositive excitation nodes");
    }
    return quadrature;
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
