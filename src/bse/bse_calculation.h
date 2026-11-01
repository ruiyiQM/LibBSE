#pragma once

#include "bse_types.h"

#include <memory>

namespace librpa_int
{
class Dataset;
}

namespace libbse
{

// Haydock releases the dataset's raw wavefunctions after optical preparation.
void run_bse(const InputParameters &options,
             const std::shared_ptr<librpa_int::Dataset> &dataset);

} // namespace libbse
