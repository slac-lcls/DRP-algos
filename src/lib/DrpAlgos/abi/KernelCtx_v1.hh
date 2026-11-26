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

  constexpr unsigned kGateIdle   { 0 }; ///< No event
  constexpr unsigned kGateActive { 1 }; ///< Event with data
  constexpr unsigned kGateDone   { 2 }; ///< Set after reduction on finalize
  constexpr unsigned kGateVetoed { 3 }; ///< Event exists, but has no data (vetoed)

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
    static constexpr unsigned kABI { 1 };

    unsigned abi;         ///< Filled on the host to check ABI mismatch on device-side
    const unsigned* gate; ///< If nullptr, always run. Otherwise run only when *gate == 1
    const unsigned* slot; ///< Ring slot index (read on device, NOT at graph capture)
    const CalibFP* in;    ///< Base of calibrated input ring
    BigUint inStride;     ///< Elements (floats) per input slot
    unsigned char* out;   ///< Base of output ring
    BigUint outStride;    ///< Bytes per output slot
    unsigned* retCode;    ///< Return/result code for the entire reduction
    const BigUint* seq;   ///< Event sequence number (e.g. L1Accept counter)
    BigUint rawBytes;     ///< Number of raw (prescale) bytes ahead of reduced data
    const void* params;   ///< Device copy of the algorithm's parameter struct (AOT only)
  };

  /**
   * For use within DRP-algos, all information necessary for IO.
   */
  struct DrpEvent {
    bool is_active;     ///< Whether the algorithm can proceed. If false, return without IO
    bool is_vetoed;     ///< Whether the event was vetoed (for book-keeping on event couonters)
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
    if (ctx.abi != DrpKernelCtx::kABI) {
      // Wrong ABI
      return {
        false,
        false,
        0,
        nullptr,
        nullptr
      };
    }

    const unsigned gate { ctx.gate ? *ctx.gate : kGateActive };
    const BigUint seq { ctx.seq ? *ctx.seq : 0ULL };
    const bool is_active { gate == kGateActive };
    const bool is_vetoed { gate == kGateVetoed };

    const unsigned slot { *ctx.slot };
    return {
      is_active,
      is_vetoed,
      seq,
      is_active ? ctx.in + slot * ctx.inStride : nullptr,
      is_active ? ctx.out + slot * ctx.outStride + ctx.rawBytes : nullptr
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
    if (ctx.gate && *ctx.gate != kGateActive) {
      return; // Inactive/blocked event
    }

    // NOTE: The data pointer moves if rawBytes in non-zero. That's done in drp_event
    //       Here, the size is always returned before the raw block.
    reinterpret_cast<BigUint*>(ctx.out + (*ctx.slot) * ctx.outStride)[-1] = result.size;

    if (ctx.retCode) {
      *ctx.retCode = result.status;
    }
  }
#endif // __CUDACC__ || __CUDACC_RTC__

  // The namespace itself should always be visible, even if its empty
  namespace details {
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    /**
     * Complete the state machine transitions after reduction completes.
     *
     * If active, `fixedSize` is written into the size location. If `vetoed`,
     * 0 is written (no payload). In both cases, if the seqStride is non-zero,
     * the sequence counter is advanced for the use of any reduction algorithms
     * that will need to count events. The done state is then written to move
     * the system along.
     *
     * @note For the idle case, nothing is done here.
     *
     * @warning This is an adapter utility only! I.e., for AlgorithmHost, or the
     *          actual DRP integration. Actual algorithms should only use `drp_commit`.
     */
    __device__ inline void drp_finalize(const DrpKernelCtx& ctx,
                                        BigUint fixedSize,
                                        BigUint seqStride) {
      const auto evt { drp_event(ctx) };
      if (!evt.is_active && !evt.is_vetoed) {
        return;
      }

      const unsigned slot { *ctx.slot };
      BigUint* const size {
        reinterpret_cast<BigUint*>(ctx.out + slot * ctx.outStride) - 1
      };

      if (evt.is_vetoed) {
        *size = 0;
      } else if (fixedSize) {
        *size = fixedSize;
      }

      if (ctx.seq && seqStride) {
        *const_cast<BigUint*>(ctx.seq) += seqStride;
      }

      if (ctx.gate) {
        *const_cast<unsigned*>(ctx.gate) = kGateDone;
      }
    }
#endif // __CUDACC__ || __CUDACC_RTC__
  } // namespace details
} // namespace DrpAlgos::abi::v1

#endif // DRP_ALGOS_ABI_KERNEL_CTX_V1_HH
