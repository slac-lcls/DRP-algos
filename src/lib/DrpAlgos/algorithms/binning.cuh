#ifndef DRP_ALGOS_ALGORITHMS_BINNING_CUH
#define DRP_ALGOS_ALGORITHMS_BINNING_CUH

namespace DrpAlgos {
  namespace algorithms {
    /**
     * Sum-bin a 3-D (a, y, x) float image and scale the result.
     *
     * Grid-stride over the n output elements. Shapes are in elements.
     */
    __device__ __forceinline__ void binning(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            unsigned y_len_in,
                                            unsigned x_len_in,
                                            unsigned y_len_out,
                                            unsigned x_len_out,
                                            unsigned n,
                                            unsigned bin_asic,
                                            unsigned bin_y,
                                            unsigned bin_x,
                                            float scale) {
      for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const unsigned x { i % x_len_out };
        const unsigned y { (i / x_len_out) % y_len_out };
        const unsigned a { i / (x_len_out * y_len_out) };

        float sum { 0.0f };
        for (unsigned da = 0; da < bin_asic; ++da) {
          for (unsigned dy = 0; dy < bin_y; ++dy) {
            for (unsigned dx = 0; dx < bin_x; ++dx) {
              sum +=
                in[((a * bin_asic + da) * y_len_in + (y * bin_y + dy)) * x_len_in + (x * bin_x + dx)];
            }
          }
        }
        out[i] = sum * scale;
      }
    }
  } // namespace algorithms
} // namespace DrpAlgos

#endif // DRP_ALGOS_ALGORITHMS_BINNING_CUH
