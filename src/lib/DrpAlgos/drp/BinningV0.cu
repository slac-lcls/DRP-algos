#include "DrpAlgos/algorithms/binning.cuh"
#include "DrpAlgos/drp/KernelRegistry.hh"
#include "DrpAlgos/drp/TypedKernelDescriptor.hh"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {
  struct BinningParams {
    std::uint32_t in_shape[3]; ///< (a, y, x) of the calibrated input
    std::uint32_t bin[3];      ///< Bin factor per dimension
    float         scale;       ///< Multiplier applied to each summed bin
  };

  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BinningParams, in_shape, bin, scale)

  __global__ void binningKernel(DrpKernelCtx ctx) {
    if (!drp_active(ctx)) {
      return;
    }

    const BinningParams& params { drp_params<BinningParams>(ctx) };
    const unsigned oa { params.in_shape[0] / params.bin[0] };
    const unsigned oy { params.in_shape[1] / params.bin[1] };
    const unsigned ox { params.in_shape[2] / params.bin[2] };

    DrpAlgos::algorithms::binning(drp_in(ctx),
                                  drp_out<float>(ctx),
                                  params.in_shape[1],
                                  params.in_shape[2],
                                  oy,
                                  ox,
                                  oa * oy * ox,
                                  params.bin[0],
                                  params.bin[1],
                                  params.bin[2],
                                  params.scale);
  }

  class BinningV0_1_0 final : public DrpAlgos::TypedKernelDescriptor<BinningParams> {
  public:
    std::size_t payloadSize() const override { return outElems() * sizeof(float); }

    void launch(DrpKernelCtx ctx, cudaStream_t stream) override {
      constexpr unsigned threads { 256 };
      const unsigned blocks { static_cast<unsigned>((outElems() + threads - 1) / threads) };
      binningKernel<<<blocks, threads, 0, stream>>>(withParams(ctx));
    }

  protected:
    void onConfigure() override {
      for (int d = 0; d < 3; ++d) {
        if (m_params.bin[d] == 0 || (m_params.in_shape[d] % m_params.bin[d]) != 0) {
          throw std::invalid_argument("Binning v1: in_shape must be divisible by bin in every dimension");
        }
      }
    }

  private:
    std::size_t outElems() const {
      return std::size_t { m_params.in_shape[0] / m_params.bin[0] }
           * (m_params.in_shape[1] / m_params.bin[1])
           * (m_params.in_shape[2] / m_params.bin[2]);
    }
  };

  REGISTER_KERNEL("Binning", 0, 1, 0, BinningV0_1_0)
} // anonymous namespace
