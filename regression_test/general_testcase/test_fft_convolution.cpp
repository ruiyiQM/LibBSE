// Author: Ruiyi Zhou.
#include "bse/fft_convolution.h"
#include "bse/w_boundary_phase.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace libbse;

namespace
{
template<class Error, class Call> void rejected(Call call)
{
    try { call(); }
    catch (const Error &) { return; }
    throw std::runtime_error("invalid FFT input was accepted");
}

void check(const Cell &shape)
{
    std::vector<KPoint> points;
    for (int x = 0; x < shape[0]; ++x)
        for (int y = 0; y < shape[1]; ++y)
            for (int z = 0; z < shape[2]; ++z)
                points.push_back({.13 + double(x)/shape[0],
                                  -.27 + double(y)/shape[1],
                                  1.19 + double(z)/shape[2]});
    std::reverse(points.begin(), points.end());
    UniformFftGrid grid(points);
    if (grid.shape != shape) throw std::runtime_error("wrong FFT mesh shape");
    if (grid.index({-1, -1, -1}) != int(points.size())-1)
        throw std::runtime_error("negative R wrapping is incorrect");
    const int nk = points.size(), fields = 6;
    std::vector<Complex> input(nk * fields), values(nk * fields), kernel(nk);
    for (int k = 0; k < nk; ++k)
        for (int f = 0; f < fields; ++f)
        {
            input[k*fields+f] = {std::sin(.31*(k+1)*(f+1)), std::cos(.17*(k+2)*(f+1))};
            values[grid.k_to_grid[k]*fields+f] = input[k*fields+f];
        }
    const Complex weight(.4, -.2);
    const auto images = w_boundary_images({-1,-1,-1}, {2,2,2}, true);
    for (const auto &[cell, factor] : images)
        kernel[grid.index(cell)] += factor * weight;
    kernel[0] += Complex(.3,.1);
    FftBatch plan(shape, fields);
    auto replay = values;
    plan.backward(replay);
    plan.forward(replay);
    for (std::size_t i = 0; i < values.size(); ++i)
        if (std::abs(replay[i] - double(nk)*values[i]) > 1e-12 * nk)
            throw std::runtime_error("FFT batch layout or normalization is wrong");
    plan.backward(values);
    for (int r = 0; r < nk; ++r)
        for (int f = 0; f < fields; ++f)
            values[r*fields+f] *= kernel[r];
    plan.forward(values);
    for (int k1 = 0; k1 < nk; ++k1)
        for (int f = 0; f < fields; ++f)
        {
            Complex expected{};
            for (int k2 = 0; k2 < nk; ++k2)
            {
                Complex w(.3,.1);
                for (const auto &[cell, factor] : images)
                {
                    double angle = 0;
                    for (int a = 0; a < 3; ++a)
                        angle += (points[k2][a]-points[k1][a])*cell[a];
                    w += weight * factor * std::polar(1.0, 2*std::acos(-1.0)*angle);
                }
                expected += w * input[k2*fields+f];
            }
            if (std::abs(values[grid.k_to_grid[k1]*fields+f] - expected)
                    > 2e-12 * std::max(1.0, std::abs(expected)))
                throw std::runtime_error("FFT differs from direct complex convolution");
        }
    std::vector<Complex> bad(1);
    rejected<std::invalid_argument>([&] { plan.forward(bad); });
}
}

int main()
{
    try
    {
        rejected<std::invalid_argument>([] { UniformFftGrid grid({}); });
        rejected<std::invalid_argument>([] { UniformFftGrid grid({KPoint{0,0,0}, KPoint{0,0,0}}); });
        rejected<std::invalid_argument>([] { UniformFftGrid grid({KPoint{0,0,0}, KPoint{0,0,.3}}); });
        rejected<std::invalid_argument>([] { UniformFftGrid grid({KPoint{0,0,std::numeric_limits<double>::quiet_NaN()}}); });
        rejected<std::invalid_argument>([] { FftBatch plan({1,0,2}, 3); });
        rejected<std::invalid_argument>([] { FftBatch plan({1,1,2}, 0); });
        if (!FftBatch::available())
            rejected<std::runtime_error>([] { FftBatch plan({1,1,2}, 3); });
        else
            for (const Cell shape : {Cell{1,1,1}, Cell{1,1,4}, Cell{2,2,3}, Cell{3,4,5}})
                check(shape);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
