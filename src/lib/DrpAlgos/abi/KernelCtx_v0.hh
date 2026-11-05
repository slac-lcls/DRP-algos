#ifndef DRP_ALGOS_ABI_KERNEL_CTX_V0_HH
#define DRP_ALGOS_ABI_KERNEL_CTX_V0_HH

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

namespace DrpAlgos::abi::v0 {
  /**
   * Unified context with state to run an algorithm within a DRP reducer.
   *
   * The KernelCtx holds information to correctly select input and output areas from
   * DRP ring buffers. It also holds the gating information needed to determine whether
   * the algorithm can proceed. For AOT compiled algorithms, the parameters are accessible
   * via the context.
   */
  struct DrpKernelCtx {
    static constexpr unsigned abi { 0 };

    const unsigned* gate;         ///< If nullptr, always run. Otherwise run only when *gate == 1
    const unsigned* slot;         ///< Ring slot index (read on device, NOT at graph capture)
    const float* in;              ///< Base of calibrated input ring
    unsigned long long inStride;  ///< Elements (floats) per input slot
    unsigned char* out;           ///< Base of output ring
    unsigned long long outStride; ///< Bytes per output slot
    const void* params;           ///< Device copy of the algorithm's parameter struct (AOT only)
  };

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)

  __device__ inline bool drp_active(const DrpKernelCtx& ctx) {
    return (ctx.gate == nullptr) || (*ctx.gate == 1);
  }

  __device__ inline const float* drp_in(const DrpKernelCtx& ctx) {
    return ctx.in + (*ctx.slot) * ctx.inStride;
  }

  template <typename T>
  __device__ inline T* drp_out(const DrpKernelCtx& ctx) {
    return reinterpret_cast<T*>(ctx.out + (*ctx.slot) * ctx.outStride);
  }

  template <typename P>
  __device__ inline const P& drp_params(const DrpKernelCtx& ctx) {
    return *reinterpret_cast<const P*>(ctx.params);
  }

  __device__ inline void drp_set_size(const DrpKernelCtx& ctx,
                                      unsigned long long nBytes) {
    reinterpret_cast<unsigned long long*>(ctx.out + (*ctx.slot) * ctx.outStride)[-1] = nBytes;
  }

#endif // __CUDACC__ || __CUDACC_RTC__
} // namespace DrpAlgos::abi::v0
#endif // DRP_ALGOS_ABI_KERNEL_CTX_V0_HH
