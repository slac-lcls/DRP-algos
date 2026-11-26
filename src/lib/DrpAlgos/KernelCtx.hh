#ifndef DRP_ALGOS_KERNEL_CTX_HH
#define DRP_ALGOS_KERNEL_CTX_HH

/**
 * @file KernelCtx.hh
 *
 * The KernelCtx defines the single object that will be passed into DRP algorithm
 * descriptors. The containing header also provides associated helper utilities
 * to pull information from it.
 *
 * The KernelCtx is ABI VERSIONED. This is set independently for each build target
 * at BUILD time - it is not controlled in the source.
 *
 * @note Because the header can be used in both the ahead-of-time (AOT) and
 * just-in-time (JIT) compilation pathways, the header MUST be JIT-compilable.
 *
 * Currently, to be JIT-compatible a header must not include any files, including the
 * standard library/CCCL headers.
 */

#define DRP_KERNEL_CTX_ABI_LATEST 1                   // Current version of ABI

#ifndef DRP_KERNEL_CTX_ABI
#define DRP_KERNEL_CTX_ABI DRP_KERNEL_CTX_ABI_LATEST
#endif                                                // Set DRP_KERNEL_CTX_ABI

#if DRP_KERNEL_CTX_ABI == 0                           // ABI V0

#include "DrpAlgos/abi/KernelCtx_v0.hh"

namespace DrpAlgosCtx {
  namespace CtxAbi = DrpAlgos::abi::v0;
} // namespace DrpAlgosCtx

#elif DRP_KERNEL_CTX_ABI == 1                         // ABI V1

#include "DrpAlgos/abi/KernelCtx_v1.hh"

namespace DrpAlgosCtx {
  namespace CtxAbi = DrpAlgos::abi::v1;
} // namespace DrpAlgosCtx

#else

#error "Unknown DRP_KERNEL_CTX_ABI set!"

#endif                                                // DRP_KERNEL_CTX_ABI Version Check

// The V0 ABI is included to test the ABI migration feature works.
// - It also the worst possible case because the facilities to do it gracefully
//   weren't yet implemented.
using DrpKernelCtx = DrpAlgosCtx::CtxAbi::DrpKernelCtx;

#if DRP_KERNEL_CTX_ABI >= 1
using DrpEvent = DrpAlgosCtx::CtxAbi::DrpEvent;
using DrpResult = DrpAlgosCtx::CtxAbi::DrpResult;
#endif

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
// Export helper routines
#if DRP_KERNEL_CTX_ABI == 0
using DrpAlgosCtx::CtxAbi::drp_active;
using DrpAlgosCtx::CtxAbi::drp_in;
using DrpAlgosCtx::CtxAbi::drp_out;
using DrpAlgosCtx::CtxAbi::drp_params;
using DrpAlgosCtx::CtxAbi::drp_set_size;
#else
using DrpAlgosCtx::CtxAbi::drp_event;
using DrpAlgosCtx::CtxAbi::drp_params;
using DrpAlgosCtx::CtxAbi::drp_commit;
#endif

#endif // __CUDACC__ || __CUDACC_RTC__

#endif // DRP_ALGOS_KERNEL_CTX_HH
