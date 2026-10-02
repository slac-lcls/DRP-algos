#ifndef DRP_ALGOS_ALGORITHMS_ROIBIN_CUH
#define DRP_ALGOS_ALGORITHMS_ROIBIN_CUH

namespace DrpAlgos::algorithms {
  /**
   * Select an ROI from a 3D image, optionally threshold, and apply pixel-binning.
   *
   * The routine currently expects a 3D input (asic, y, x), and all shapes and offsets
   * are in units of "elements". A separate scaling factor is provided as input. In
   * general, you will likely scale by the binning factor, but this is not required.
   *
   * @note The function runs with a grid-stride over the `num_elem` output elements for
   *       binning.
   * @note This function requires that the input and output be of the same datatype.
   *       I.e., it will not handle conversion between float/double/integer etc.
   *
   * @tparam DType The type of the input and output data.
   * @param[in] in The input image to ROI, threshold and bin.
   * @param[in] out Where to write the output image.
   * @param[in] y_len_in The input y-dimension shape.
   * @param[in] x_len_in The input x-dimension shape.
   * @param[in] a_start The starting offset in the `asic` (first) dimension.
   * @param[in] y_start The starting offset in the `y` (second) dimension.
   * @param[in] x_start The starting offset in the `x` (third) dimension.
   * @param[in] y_len_out The output y-dimension shape.
   * @param[in] x_len_out The output x-dimension shape.
   * @param[in] n The number of output elements to process for this thread.
   * @param[in] bin_asic The binning factor along the `asic` (first) dimension.
   * @param[in] bin_y The binning factor along the `y` (second) dimension.
   * @param[in] bin_x The binning factor along the `x` (third) dimension.
   * @param[in] apply_threshold Whether thresholding will be used.
   * @param[in] threshold The threshold to use, if `apply_threshold = true`.
   * @param[in] scale The scaling factor. In general, this will be the binning
   *            factor (or rather, its inverse), however, this is not required.
   */
  template <typename DType = float>
  __device__ __forceinline__ void roibin(const DType* __restrict__ in,
                                         DType* __restrict__ out,
                                         unsigned y_len_in,
                                         unsigned x_len_in,
                                         unsigned a_start,
                                         unsigned y_start,
                                         unsigned x_start,
                                         unsigned y_len_out,
                                         unsigned x_len_out,
                                         unsigned num_elem,
                                         unsigned bin_asic,
                                         unsigned bin_y,
                                         unsigned bin_x,
                                         bool apply_threshold,
                                         float threshold,
                                         float scale) {
    const unsigned idx { blockIdx.x * blockDim.x + threadIdx.x };
    const unsigned stride { blockDim.x * gridDim.x };

    for (unsigned i = idx; i < num_elem; i += stride) {
      const unsigned x_off { i % x_len_out };
      const unsigned y_off { (i / x_len_out) % y_len_out };
      const unsigned a_off { i / (x_len_out * y_len_out) };

      DType sum {};
      for (unsigned da = 0; da < bin_asic; ++da) {
        for (unsigned dy = 0; dy < bin_y; ++dy) {
          for (unsigned dx = 0; dx < bin_x; ++dx) {
            const unsigned offset {
              ((a_start + a_off * bin_asic + da) * y_len_in + (y_start + y_off * bin_y + dy)) * x_len_in
              + (x_start + x_off * bin_x + dx)
            };

            const DType val { in[offset] };
            sum += (apply_threshold && val < threshold) ? DType {} : val;
          }
        }
      }

      out[i] = sum * scale;
    }
  }

  /**
   * Sum a sequence of contiguous frames of num_elem elements each
   *
   * This "time" binning routine sums contiguous frames. It assumes that the inputs
   * are arranged in a fixed, deterministic, order in a single buffer without gaps.
   *
   * @note The function runs with a grid-stride over the `num_elem` output.
   * @note This function requires that the input and output be of the same datatype.
   *       I.e., it will not handle conversion between float/double/integer etc.
   *
   * @tparam DType The type of the input and output data.
   * @param[in] frames The sequence of frames to sum.
   * @param[out] out Where to write the reduced/summed frames.
   * @param[in] num_elem The number of elements in each frame.
   * @param[in] num_frames The number of frames to sum.
   */
  template <typename DType = float>
  __device__ __forceinline__ void sum_frames(const DType* __restrict__ frames,
                                             DType* __restrict__ out,
                                             unsigned num_elem,
                                             unsigned num_frames) {
    const unsigned idx { blockIdx.x * blockDim.x + threadIdx.x };
    const unsigned stride { blockDim.x * gridDim.x };

    for (unsigned i = idx; i < num_elem; i += stride) {
      DType sum {};
      for (unsigned f_idx = 0; f_idx < num_frames; ++f_idx) {
        unsigned long long f_offset {
          static_cast<unsigned long long>(f_idx) * num_elem + i
        };
        sum += frames[f_offset];
      }

      out[i] = sum;
    }
  }
} // namespace DrpAlgos::algorithms

#endif // DRP_ALGOS_ALGORITHMS_ROIBIN_CUH
