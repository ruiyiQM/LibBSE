#pragma once

#include "bse_types.h"

#include <memory>

namespace libbse
{

struct UniformFftGrid
{
    explicit UniformFftGrid(const std::vector<KPoint> &points);
    int index(const Cell &cell) const;
    Cell shape{};
    std::vector<int> k_to_grid;
};

// In-place transforms of interleaved [grid site][field] arrays. Both signs
// are unnormalized, so backward -> multiply W(R) -> forward is the k sum.
class FftBatch
{
public:
    FftBatch(const Cell &shape, int fields);
    ~FftBatch();
    static bool available();
    void backward(std::vector<Complex> &values) const;
    void forward(std::vector<Complex> &values) const;

private:
    void transform(std::vector<Complex> &values, bool backward) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t size_ = 0;
};

} // namespace libbse
