#pragma once

/// Shared between Metal shaders and C++ host code.
/// Metal defines __METAL_VERSION__; C++ does not.
#ifdef __METAL_VERSION__
using uint32 = uint;
#else
#include <cstdint>
using uint32 = uint32_t;
#endif

struct GemmParams {
    uint32 M;       // rows of A / rows of C
    uint32 N;       // rows of B / cols of C  (B is transposed)
    uint32 K;       // cols of A / cols of B
    float alpha;
    float beta;
};
