#ifndef DRP_ALGOS_DRP_LAUNCH_HH
#define DRP_ALGOS_DRP_LAUNCH_HH

// Provide conversion between all ABI variants
#include "DrpAlgos/abi/KernelCtx_v0.hh"
#include "DrpAlgos/abi/KernelCtx_v1.hh"
#include "DrpAlgos/drp/KernelDescriptor.hh"
#include "DrpAlgos/export_macro.hh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace DrpAlgos {
  using LatestCtx = abi::v1::DrpKernelCtx;

  // For adapter helper functions like drp_finalize
  namespace details = abi::v1::details;

  /**
   * Conversion from V1 ABI to V0 as needed.
   */
  inline abi::v0::DrpKernelCtx to_abi_v0(const LatestCtx& ctx) {
    return DrpAlgos::abi::v0::DrpKernelCtx {
      .abi = abi::v0::DrpKernelCtx::kABI,
      .gate = ctx.gate,
      .slot = ctx.slot,
      .in = ctx.in,
      .inStride = ctx.inStride,
      .out = ctx.out,
      .outStride = ctx.outStride,
      .params = ctx.params
    };
  }

  inline void launchAnyAbi(IKernelDescriptor& desc,
                           const LatestCtx& host_ctx,
                           cudaStream_t stream) {
    if (host_ctx.abi != LatestCtx::kABI) {
      throw std::invalid_argument("launchAnyAbi: host ctx.abi must be LatestCtx::kABI");
    }

    switch (desc.ctxAbi()) {
    case 1: {
      desc.launch(&host_ctx, stream);
      break;
    }
    case 0: {
      auto ctx { to_abi_v0(host_ctx) };
      desc.launch(&ctx, stream);
      break;
    }
    default: {
      throw std::runtime_error("Unsupported KernelCtx ABI version!");
    }
    }
  }

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
   */
  DRP_ALGOS_API void recordFinalize(const LatestCtx& ctx,
                                    unsigned long long fixedSize,
                                    unsigned long long seqStride,
                                    cudaStream_t stream);

  /**
   * Run `launchAnyAbi` and then `recordFinalize`.
   */
  DRP_ALGOS_API void recordAlgorithm(IKernelDescriptor& desc,
                                     const LatestCtx& ctx,
                                     unsigned long long seqStride,
                                     cudaStream_t stream);
}
#endif // DRP_ALGOS_DRP_LAUNCH_HH
