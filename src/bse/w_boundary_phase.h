#pragma once

#include "parameter/parameter.h"

#include <array>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

namespace libbse
{

// The real-space image representation of the same endpoint correction below.
inline std::vector<std::pair<std::array<int, 3>, double>> w_boundary_images(
    const std::array<int, 3> &cell, const std::array<int, 3> &period, bool average)
{
    std::vector<std::pair<std::array<int, 3>, double>> images{{cell, 1.0}};
    if (average)
        for (int axis = 0; axis < 3; ++axis)
            if (period[axis] % 2 == 0 && cell[axis] == -period[axis] / 2)
            {
                const auto count = images.size();
                for (std::size_t j = 0; j < count; ++j)
                {
                    images[j].second *= 0.5;
                    auto opposite = images[j];
                    opposite.first[axis] *= -1;
                    images.push_back(opposite);
                }
            }
    return images;
}

// Return the averaged boundary phase minus the original single-image phase.
// q is fractional, cell is integer, and period is the coarse BvK mesh size.
inline std::complex<double> w_boundary_phase_delta(
    const std::array<double, 3> &w_qfrac,
    const std::array<int, 3> &w_cell,
    const std::array<int, 3> &w_period)
{
    constexpr double w_grid_tolerance = 1.e-12;
    bool w_on_coarse_grid = true;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double w_scaled_q = w_qfrac[axis] * w_period[axis];
        w_on_coarse_grid = w_on_coarse_grid
                          && std::abs(w_scaled_q - std::round(w_scaled_q))
                                 < w_grid_tolerance;
    }
    if (w_on_coarse_grid) return 0.0; // Preserve coarse samples exactly.

    const double w_two_pi = 2.0 * Constants::pi;
    std::complex<double> w_original_phase = 1.0;
    std::complex<double> w_averaged_phase = 1.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double w_angle = w_two_pi * w_qfrac[axis] * w_cell[axis];
        const std::complex<double> w_axis_phase = std::polar(1.0, w_angle);
        w_original_phase *= w_axis_phase;

        // Average the +/-N/2 images with cos(angle); products cover edges/corners.
        const bool w_is_boundary = w_period[axis] % 2 == 0
                                   && w_cell[axis] == -w_period[axis] / 2;
        if (w_is_boundary)
            w_averaged_phase *= std::complex<double>(std::cos(w_angle), 0.0);
        else
            w_averaged_phase *= w_axis_phase;
    }
    return w_averaged_phase - w_original_phase;
}

} // namespace libbse
