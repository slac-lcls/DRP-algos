#ifndef DRP_ALGOS_ALGORITHMS_ROIBIN_CUH
#define DRP_ALGOS_ALGORITHMS_ROIBIN_CUH

namespace DrpAlgos::algorithms {
  struct AsicPlacement {
    unsigned asic;  ///< ASIC index in an ASIC-major input buffer.
    bool flip_rows; ///< Panel rows run opposite to the ASIC rows
    bool flip_cols; ///< Panel columns run opposite to the ASIC columns
  };

  __device__ __forceinline__ unsigned panel_to_buffer(unsigned row,
                                                      unsigned col,
                                                      unsigned asic_rows,
                                                      unsigned asic_cols,
                                                      unsigned grid_cols,
                                                      const AsicPlacement* __restrict__ layout) {
    const unsigned div_r { row / asic_rows };
    const unsigned mod_r { row % asic_rows };
    const unsigned div_c { col / asic_cols };
    const unsigned mod_c { col % asic_cols };

    const AsicPlacement placement { layout[div_r * grid_cols + div_c] };
    const unsigned y { placement.flip_rows ? asic_rows - 1 - mod_r : mod_r };
    const unsigned x { placement.flip_cols ? asic_cols - 1 - mod_c : mod_c };

    return (placement.asic * asic_rows + y) * asic_cols + x;
  }

  /**
   * Select an ROI from a 3D image, optionally threshold, and apply pixel-binning.
   *
   * Shapes and offsets are in panel coordinates (plane, row, col), in units of
   * "elements". Each plane is one panel stored ASIC-major in the input buffer; pixels
   * are located with `panel_to_buffer`. A separate scaling factor is provided as
   * input. In general, you will likely scale by the binning factor, but this is not
   * required.
   *
   * @note The function runs with a grid-stride over the `num_elem` output elements for
   *       binning.
   * @note This function requires that the input and output be of the same datatype.
   *       I.e., it will not handle conversion between float/double/integer etc.
   *
   * @tparam DType The type of the input and output data.
   * @param[in] in The input image to ROI, threshold and bin.
   * @param[in] out Where to write the output image.
   * @param[in] y_len_in The panel rows.
   * @param[in] x_len_in The panel columns.
   * @param[in] asic_rows Rows in one ASIC.
   * @param[in] asic_cols Columns in one ASIC.
   * @param[in] grid_cols ASICs per row of the panel's ASIC grid.
   * @param[in] layout Placement of each ASIC grid position, row-major.
   * @param[in] a_start The starting offset in the plane (first) dimension.
   * @param[in] y_start The starting panel row.
   * @param[in] x_start The starting panel column.
   * @param[in] y_len_out The output y-dimension shape.
   * @param[in] x_len_out The output x-dimension shape.
   * @param[in] n The number of output elements to process for this thread.
   * @param[in] bin_asic The binning factor along the plane (first) dimension.
   * @param[in] bin_y The binning factor along panel rows.
   * @param[in] bin_x The binning factor along panel columns.
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
                                         unsigned asic_rows,
                                         unsigned asic_cols,
                                         unsigned grid_cols,
                                         const AsicPlacement* __restrict__ layout,
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
    const unsigned plane_len { y_len_in * x_len_in };

    for (unsigned i = idx; i < num_elem; i += stride) {
      const unsigned x_off { i % x_len_out };
      const unsigned y_off { (i / x_len_out) % y_len_out };
      const unsigned a_off { i / (x_len_out * y_len_out) };

      DType sum {};
      for (unsigned da = 0; da < bin_asic; ++da) {
        for (unsigned dy = 0; dy < bin_y; ++dy) {
          for (unsigned dx = 0; dx < bin_x; ++dx) {
            const unsigned offset {
              (a_start + a_off * bin_asic + da) * plane_len +
              panel_to_buffer(y_start + y_off * bin_y + dy,
                              x_start + x_off * bin_x + dx,
                              asic_rows,
                              asic_cols,
                              grid_cols,
                              layout)
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

  /**
   * Sum a sequence of frames of num_elem elements each, potentially with gaps.
   *
   * This "time" binning routine sums frames from a contiguous block. Unlike with
   * sum_frames, it can handle if some of the block contains gaps (due to missing
   * data).
   *
   * @note The function runs with a grid-stride over the `num_elem` output.
   * @note This function requires that the input and output be of the same datatype.
   *       I.e., it will not handle conversion between float/double/integer etc.
   *
   * @note If the presence indicator is a nullptr, or there are more frames than
   *       the bit-mask accounts for, all frames will be summed.
   *
   * @tparam DType The type of the input and output data.
   * @param[in] frames The sequence of frames to sum.
   * @param[out] out Where to write the reduced/summed frames.
   * @param[in] num_elem The number of elements in each frame.
   * @param[in] num_frames The number of frames to sum.
   * @param[in] num_presence The number of 8-byte frame presence indicator bit-masks.
   * @param[in] presence The bit-masks for indicating presence/absence of frames.
   */
  template <typename DType = float>
  __device__ __forceinline__ void sum_present_frames(const DType* __restrict__ frames,
                                                     DType* __restrict__ out,
                                                     unsigned num_elem,
                                                     unsigned num_frames,
                                                     unsigned long long num_presence,
                                                     const unsigned long long* presence) {
    const unsigned idx { blockIdx.x * blockDim.x + threadIdx.x };
    const unsigned stride { blockDim.x * gridDim.x };

    for (unsigned i = idx; i < num_elem; i += stride) {
      DType sum {};
      for (unsigned f_idx = 0; f_idx < num_frames; ++f_idx) {
        unsigned long long p_idx { f_idx / 64 }; // 8 byte bit-masks
        if ((presence == nullptr)   ||
            (p_idx >= num_presence) ||
            ((presence[p_idx] >> (f_idx % 64)) & 1ULL)) {
          unsigned long long f_offset {
            static_cast<unsigned long long>(f_idx) * num_elem + i
          };
          sum += frames[f_offset];
        }
      }

      out[i] = sum;
    }
  }
} // namespace DrpAlgos::algorithms

#endif // DRP_ALGOS_ALGORITHMS_ROIBIN_CUH
