#include "DrpAlgos/drp/Launch.hh"

#include "DrpAlgos/drp/KernelDescriptor.hh"

#include <stdexcept>
#include <string>

namespace {
  void check(const char* what) {
    const cudaError_t err { cudaGetLastError() };

    if (err != cudaSuccess) {
      throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
  }

  __global__ void drpFinalize(DrpAlgos::LatestCtx ctx,
                              unsigned long long const fixedSize,
                              unsigned long long const seqStride) {
    DrpAlgos::details::drp_finalize(ctx, fixedSize, seqStride);
  }
} // anonymous namespace

namespace DrpAlgos {
  void recordFinalize(const LatestCtx& ctx,
                      unsigned long long fixedSize,
                      unsigned long long seqStride,
                      cudaStream_t stream) {
    drpFinalize<<<1, 1, 0, stream>>>(ctx, fixedSize, seqStride);
    check("recordFinalize");
  }

  void recordAlgorithm(IKernelDescriptor& desc,
                       const LatestCtx& ctx,
                       unsigned long long seqStride,
                       cudaStream_t stream) {
    launchAnyAbi(desc, ctx, stream);
    recordFinalize(ctx,
                   desc.variableSize() ? 0ULL : desc.payloadSize(),
                   seqStride,
                   stream);
  }
} // namespace DrpAlgos