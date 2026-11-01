#pragma once

#include <mpi.h>
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

// Collective recurrence with disjoint local vector slices, including empty ones.
// apply consumes and returns local slices. Norms and overlaps use comm.
// History storage is O(steps*local_dimension); no global vectors are retained.
HaydockChain haydock_recursion(const TdaApply &apply,
                              const std::vector<std::complex<double>> &local_seed,
                              int max_steps, MPI_Comm comm);

// Generalized averaged Gauss quadrature, using the final residual beta without
// additional operator applications. Throws domain_error for nonpositive nodes.
HaydockChain haydock_gagq_chain(const HaydockChain &chain);

std::complex<double> haydock_resolvent(const HaydockChain &chain,
                                      std::complex<double> energy);

} // namespace libbse
