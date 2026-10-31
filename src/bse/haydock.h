#pragma once

#include <complex>
#include <functional>
#include <vector>

namespace libbse
{

using TdaApply = std::function<void(const std::vector<std::complex<double>> &,
                                    std::vector<std::complex<double>> &)>;

struct HaydockChain
{
    double seed_norm_squared = 0.0;
    std::vector<double> alpha;
    // beta[j] connects j to j+1; the final entry is the residual norm.
    std::vector<double> beta;
    bool terminated = false;
};

// Replicated vectors; apply may perform collective distributed contractions.
// Retain the Krylov vectors for two-pass reorthogonalization, O(steps*D) memory.
HaydockChain haydock_recursion(const TdaApply &apply,
                              const std::vector<std::complex<double>> &seed,
                              int max_steps);

std::complex<double> haydock_resolvent(const HaydockChain &chain,
                                      std::complex<double> energy);

} // namespace libbse
