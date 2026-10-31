#include "fft_convolution.h"
#include "uniform_qmesh.h"

#include <limits>
#include <mutex>
#include <stdexcept>

#ifdef LIBBSE_HAVE_FFTW
#include <fftw3.h>
#endif

namespace libbse
{
namespace
{
#ifdef LIBBSE_HAVE_FFTW
std::mutex planner_mutex;
#endif
}

UniformFftGrid::UniformFftGrid(const std::vector<KPoint> &points)
{
    // Coordinates are fractional; reduce first to avoid overflowing the
    // integer grid lookup on reciprocal-equivalent input coordinates.
    auto folded = points;
    for (auto &point : folded)
        for (auto &coordinate : point)
        {
            if (!std::isfinite(coordinate))
                throw std::invalid_argument("nonfinite FFT k point");
            coordinate -= std::floor(coordinate);
        }
    shape = infer_uniform_qmesh(folded);
    if (shape[0] == 0)
        throw std::invalid_argument("FFT LRI requires a complete uniform fine k mesh");
    if (points.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("FFT grid exceeds integer indexing");
    std::vector<bool> seen(points.size(), false);
    for (const auto &point : folded)
    {
        Cell cell{};
        for (int axis = 0; axis < 3; ++axis)
            cell[axis] = static_cast<int>(std::llround(
                (point[axis] - folded.front()[axis]) * shape[axis]));
        const int site = index(cell);
        if (seen[site]) throw std::invalid_argument("duplicate FFT grid site");
        seen[site] = true;
        k_to_grid.push_back(site);
    }
}

int UniformFftGrid::index(const Cell &cell) const
{
    int site = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        int coordinate = cell[axis] % shape[axis];
        if (coordinate < 0) coordinate += shape[axis];
        site = site * shape[axis] + coordinate;
    }
    return site;
}

struct FftBatch::Impl
{
#ifdef LIBBSE_HAVE_FFTW
    fftw_plan backward = nullptr, forward = nullptr;
    ~Impl()
    {
        std::lock_guard<std::mutex> lock(planner_mutex);
        if (backward) fftw_destroy_plan(backward);
        if (forward) fftw_destroy_plan(forward);
    }
#endif
};

bool FftBatch::available()
{
#ifdef LIBBSE_HAVE_FFTW
    return true;
#else
    return false;
#endif
}

FftBatch::FftBatch(const Cell &shape, int fields) : impl_(std::make_unique<Impl>())
{
    if (fields <= 0) throw std::invalid_argument("invalid FFT field count");
    std::size_t points = 1;
    for (const int n : shape)
    {
        if (n <= 0 || points > static_cast<std::size_t>(std::numeric_limits<int>::max()) / n)
            throw std::invalid_argument("invalid or overflowing FFT grid shape");
        points *= n;
    }
    if (points > std::vector<Complex>().max_size() / fields)
        throw std::overflow_error("FFT buffer size overflow");
    size_ = points * fields;
#ifdef LIBBSE_HAVE_FFTW
    std::vector<Complex> buffer(size_);
    auto *data = reinterpret_cast<fftw_complex *>(buffer.data());
    std::lock_guard<std::mutex> lock(planner_mutex);
    impl_->backward = fftw_plan_many_dft(3, shape.data(), fields,
        data, nullptr, fields, 1, data, nullptr, fields, 1,
        FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED);
    impl_->forward = fftw_plan_many_dft(3, shape.data(), fields,
        data, nullptr, fields, 1, data, nullptr, fields, 1,
        FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED);
    if (!impl_->backward || !impl_->forward)
        throw std::runtime_error("failed to create FFTW plans");
#else
    throw std::runtime_error("FFT LRI requires a build with LIBBSE_ENABLE_FFTW=ON");
#endif
}

FftBatch::~FftBatch() = default;

void FftBatch::transform(std::vector<Complex> &values, bool backward) const
{
    if (values.size() != size_) throw std::invalid_argument("FFT buffer shape mismatch");
#ifdef LIBBSE_HAVE_FFTW
    auto *data = reinterpret_cast<fftw_complex *>(values.data());
    fftw_execute_dft(backward ? impl_->backward : impl_->forward, data, data);
#else
    (void)backward;
    throw std::runtime_error("FFTW is unavailable");
#endif
}

void FftBatch::backward(std::vector<Complex> &values) const { transform(values, true); }
void FftBatch::forward(std::vector<Complex> &values) const { transform(values, false); }

} // namespace libbse
