#ifndef DRP_ALGOS_ABI_KERNEL_CTX_V1_HH
#define DRP_ALGOS_ABI_KERNEL_CTX_V1_HH

/**
 * @file KernelCtx.hh
 *
 * The KernelCtx defines the single object that will be passed into DRP algorithm
 * descriptors and associated helper utilities to pull information from it.
 *
 * Because the header can be used in both the ahead-of-time (AOT) and just-in-time (JIT)
 * compilation pathways, the header MUST be JIT-compilable.
 *
 * Currently, to be JIT-compatible a header must not include any files, including the
 * standard library/CCCL headers.
 */

// V1 ABI
// Compared to V0 - Adds the return code (`retCode`) and sequence counter (`seq`)

namespace DrpAlgos::abi::v1 {
  using CalibFP = float;
  using BigUint = unsigned long long; // Avoid issues with size_t etc across NVRTC

  /**
   * Unified context with state to run an algorithm within a DRP reducer.
   *
   * The KernelCtx holds information to correctly select input and output areas from
   * DRP ring buffers. It also holds the gating information needed to determine whether
   * the algorithm can proceed. For AOT compiled algorithms, the parameters are accessible
   * via the context.
   *
   * @note DrpKernelCtx gets filled on the lcls2 DRP side. Within `DRP-algos`, it is an
   *       opaque handle. It's turned into `DrpEvent` at the start, and providing a
   *       `DrpResult` at the end allows the integration into lcls2 to refill whats
   *       needed on the core DRP infrastructure side.
   */
  struct DrpKernelCtx {
    static constexpr unsigned abi { 1 };

    const unsigned* gate; ///< If nullptr, always run. Otherwise run only when *gate == 1
    const unsigned* slot; ///< Ring slot index (read on device, NOT at graph capture)
    const CalibFP* in;    ///< Base of calibrated input ring
    BigUint inStride;     ///< Elements (floats) per input slot
    unsigned char* out;   ///< Base of output ring
    BigUint outStride;    ///< Bytes per output slot
    unsigned* retCode;    ///< Return/result code for the entire reduction
    const BigUint* seq;   ///< Event sequence number (e.g. L1Accept counter)
    const void* params;   ///< Device copy of the algorithm's parameter struct (AOT only)
  };

  /**
   * For use within DRP-algos, all information necessary for IO.
   */
  struct DrpEvent {
    bool is_active;     ///< Whether the algorithm can proceed. If false, return without IO
    BigUint seq;        ///< Event sequence number (e.g. L1Accept counter)
    const CalibFP* in;  ///< Pointer to the event's calibrated input
    unsigned char* out; ///< Pointer to the slot to write output data

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    template <typename T>
    __device__ T* out_as() const { return reinterpret_cast<T*>(out); }
#endif // __CUDACC__ || __CUDACC_RTC__
  };

  struct DrpResult {
    BigUint size { 0 };    ///< Total bytes of reduced data written - used if variableSize() == true
    unsigned status { 0 }; ///< Processing result. 0 = Ok
  };


#if defined(__CUDACC__) || defined(__CUDACC_RTC__)

  __device__ inline DrpEvent drp_event(const DrpKernelCtx& ctx) {
    const bool is_active { (ctx.gate == nullptr) || (*ctx.gate == 1) };
    if (!is_active) {
      return {
        false,
        0,
        nullptr,
        nullptr
      };
    }

    const unsigned slot { *ctx.slot };
    return {
      true,
      ctx.seq ? *ctx.seq : 0ULL,
      ctx.in + slot * ctx.inStride,
      ctx.out + slot * ctx.outStride
    };
  }

  /**
   * Retrieve device side algorithms.
   */
  template <typename P>
  __device__ inline const P& drp_params(const DrpKernelCtx& ctx) {
    return *reinterpret_cast<const P*>(ctx.params);
  }

  /**
   * Commit the result of each event's reduction processing.
   *
   * @note This should be called once, EXACTLY once, per event, from one thread.
   */
  __device__ inline void drp_commit(const DrpKernelCtx& ctx,
                                    const DrpResult& result) {
    if (ctx.gate && *ctx.gate != 1) {
      return; // Inactive/blocked event
    }

    reinterpret_cast<BigUint*>(ctx.out + (*ctx.slot) * ctx.outStride)[-1] = result.size;

    if (ctx.retCode) {
      *ctx.retCode = result.status;
    }
  }
#endif // __CUDACC__ || __CUDACC_RTC__
} // namespace DrpAlgos::abi::v1

#endif // DRP_ALGOS_ABI_KERNEL_CTX_V1_HH
