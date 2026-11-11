#ifndef DRP_ALGOS_DRP_LAUNCH_HH
#define DRP_ALGOS_DRP_LAUNCH_HH

// Provide conversion between all ABI variants
#include "DrpAlgos/abi/KernelCtx_v0.hh"
#include "DrpAlgos/abi/KernelCtx_v1.hh"
#include "DrpAlgos/drp/KernelDescriptor.hh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace DrpAlgos {
  using LatestCtx = abi::v1::DrpKernelCtx;

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
}
#endif // DRP_ALGOS_DRP_LAUNCH_HH
