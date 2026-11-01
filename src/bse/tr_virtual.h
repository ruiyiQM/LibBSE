// Author: Ruiyi Zhou.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace libbse
{

inline std::vector<int> tr_partners(const std::vector<std::array<double, 3>> &points)
{
    using Key = std::array<long long, 3>;
    const auto key = [](const std::array<double, 3> &point) {
        Key result{};
        for (int axis = 0; axis < 3; ++axis)
            result[axis] = std::llround((point[axis] - std::floor(point[axis])) * 1e10)
                % 10000000000ll;
        return result;
    };
    std::vector<int> result(points.size(), -1);
    std::map<Key, int> lookup;
    for (std::size_t k = 0; k < points.size(); ++k)
    {
        for (const double value : points[k])
            if (!std::isfinite(value)) return result;
        // Ambiguous coordinates cannot identify a unique partner.
        if (!lookup.emplace(key(points[k]), k).second) return result;
    }
    for (std::size_t k = 0; k < points.size(); ++k)
    {
        auto negative = points[k];
        for (auto &value : negative) value = -value;
        const auto found = lookup.find(key(negative));
        if (found != lookup.end()) result[k] = found->second;
    }
    for (std::size_t k = 0; k < result.size(); ++k)
        if (result[k] >= 0 && result[result[k]] != static_cast<int>(k)) result[k] = -1;
    return result;
}

inline std::vector<int> tr_owned_k(const std::vector<int> &partner, int owner, int owners)
{
    if (owners < 1 || owner < 0 || owner >= owners
        || static_cast<std::size_t>(owners) > partner.size())
        throw std::invalid_argument("invalid k owner partition");
    for (std::size_t k = 0; k < partner.size(); ++k)
        if (partner[k] < -1 || (partner[k] >= 0
            && (static_cast<std::size_t>(partner[k]) >= partner.size()
                || partner[partner[k]] != static_cast<int>(k))))
            throw std::invalid_argument("invalid inversion partner map");
    std::vector<std::vector<int>> pairs, singles;
    std::vector<bool> used(partner.size());
    for (int k = 0; k < static_cast<int>(partner.size()); ++k)
        if (!used[k])
        {
            const int q = partner[k];
            used[k] = true;
            if (q >= 0 && q != k && !used[q])
            {
                used[q] = true;
                pairs.push_back({k, q});
            }
            else
                singles.push_back({k});
        }
    std::vector<int> result;
    // Keep every owner nonempty when there are too few inversion orbits.
    if (pairs.size() + singles.size() < static_cast<std::size_t>(owners))
    {
        for (int k = owner; k < static_cast<int>(partner.size()); k += owners)
            result.push_back(k);
        return result;
    }
    std::vector<int> load(owners);
    for (const auto *groups : {&pairs, &singles})
        for (const auto &group : *groups)
        {
            const int target = std::min_element(load.begin(), load.end()) - load.begin();
            load[target] += group.size();
            if (target == owner) result.insert(result.end(), group.begin(), group.end());
        }
    std::sort(result.begin(), result.end());
    return result;
}

inline std::vector<int> tr_partner_slots(const std::vector<int> &partner,
                                         const std::vector<int> &owned)
{
    std::vector<int> slot(partner.size(), -1), result(owned.size(), -1);
    for (std::size_t i = 0; i < owned.size(); ++i)
    {
        if (slot.at(owned[i]) >= 0) throw std::invalid_argument("duplicate local k index");
        slot.at(owned[i]) = i;
    }
    for (std::size_t i = 0; i < owned.size(); ++i)
    {
        const int q = partner.at(owned[i]);
        if (q >= 0) result[i] = slot.at(q);
    }
    return result;
}

struct VirtualConjugateFit
{
    bool solved = false;
    std::vector<std::complex<double>> matrix;
    double residual = std::numeric_limits<double>::infinity();
    double unitarity = std::numeric_limits<double>::infinity();
};

// Two-pass thin QR avoids squaring the virtual-vertex condition number.
inline VirtualConjugateFit fit_virtual_conjugate(const std::complex<double> *first,
    const std::complex<double> *second, int rows, int stride, int start, int count)
{
    using C = std::complex<double>;
    VirtualConjugateFit result;
    if (!first || !second || rows < 1 || count < 1 || start < 0
        || start > stride || count > stride - start)
        return result;
    const std::size_t ld = stride, n = count;
    std::vector<C> q(static_cast<std::size_t>(rows) * count),
        r(static_cast<std::size_t>(count) * count), u(r.size()), v(rows);
    double scale = 0., norm_b = 0.;
    for (int j = 0; j < count; ++j)
    {
        double norm = 0.;
        for (int k = 0; k < rows; ++k)
        {
            norm += std::norm(first[k * ld + start + j]);
            norm_b += std::norm(second[k * ld + start + j]);
        }
        scale = std::max(scale, std::sqrt(norm));
    }
    if (!(scale > 0.) || !(norm_b > 0.) || !std::isfinite(scale + norm_b)) return result;
    for (int j = 0; j < count; ++j)
    {
        for (int k = 0; k < rows; ++k) v[k] = std::conj(first[k * ld + start + j]);
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < j; ++i)
            {
                C dot{};
                for (int k = 0; k < rows; ++k) dot += std::conj(q[k * n + i]) * v[k];
                r[i * n + j] += dot;
                for (int k = 0; k < rows; ++k) v[k] -= q[k * n + i] * dot;
            }
        double norm = 0.;
        for (const C value : v) norm += std::norm(value);
        norm = std::sqrt(norm);
        if (!(norm > 1e-12 * scale) || !std::isfinite(norm)) return result;
        r[j * n + j] = norm;
        for (int k = 0; k < rows; ++k) q[k * n + j] = v[k] / norm;
    }
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < count; ++j)
            for (int k = 0; k < rows; ++k)
                u[i * n + j] += std::conj(q[k * n + i]) * second[k * ld + start + j];
    for (int i = count - 1; i >= 0; --i)
        for (int j = 0; j < count; ++j)
        {
            for (int k = i + 1; k < count; ++k) u[i * n + j] -= r[i * n + k] * u[k * n + j];
            u[i * n + j] /= r[i * n + i];
        }
    double error = 0., unitarity = 0.;
    for (int k = 0; k < rows; ++k)
        for (int j = 0; j < count; ++j)
        {
            C value{};
            for (int i = 0; i < count; ++i)
                value += std::conj(first[k * ld + start + i]) * u[i * n + j];
            error += std::norm(value - second[k * ld + start + j]);
        }
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < count; ++j)
        {
            C value{};
            for (int k = 0; k < count; ++k) value += std::conj(u[k * n + i]) * u[k * n + j];
            unitarity = std::max(unitarity, std::abs(value - C(i == j ? 1. : 0.)));
        }
    result.residual = std::sqrt(error / norm_b);
    result.unitarity = unitarity;
    result.solved = std::isfinite(result.residual + unitarity);
    result.matrix = std::move(u);
    return result;
}

} // namespace libbse
