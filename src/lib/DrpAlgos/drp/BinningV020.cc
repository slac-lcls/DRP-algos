#include "DrpAlgos/drp/JITKernelDescriptor.hh"
#include "DrpAlgos/drp/KernelRegistry.hh"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace DrpAlgos {
  extern const char* const kBinningKernelHeaderSource;
}

namespace {

  // Parameters arrive as compile-time constants in namespace jit_params:
  //   in_shape(i), in_shape_size, bin(i), bin_size, scale
  constexpr const char* kBinningSource = R"cuda(
#include "DrpAlgos/KernelCtx.hh"
#include "DrpAlgos/algorithms/binning.cuh"

using namespace jit_params;

static_assert(in_shape_size == 3 && bin_size == 3, "Binning expects 3-D in_shape and bin");
static_assert(in_shape(0) % bin(0) == 0 && in_shape(1) % bin(1) == 0 && in_shape(2) % bin(2) == 0,
              "in_shape must be divisible by bin");

constexpr unsigned OA { static_cast<unsigned>(in_shape(0) / bin(0)) };
constexpr unsigned OY { static_cast<unsigned>(in_shape(1) / bin(1)) };
constexpr unsigned OX { static_cast<unsigned>(in_shape(2) / bin(2)) };
static_assert(1ULL * OA * OY * OX < 4294967296ULL, "output too large for 32-bit indexing");

extern "C" __global__ void drp_kernel(DrpKernelCtx ctx) {
  if (!drp_active(ctx)) {
    return;
  }

  DrpAlgos::algorithms::binning(drp_in(ctx),
                                drp_out<float>(ctx),
                                static_cast<unsigned>(in_shape(1)),
                                static_cast<unsigned>(in_shape(2)),
                                OY,
                                OX,
                                OA * OY * OX,
                                static_cast<unsigned>(bin(0)),
                                static_cast<unsigned>(bin(1)),
                                static_cast<unsigned>(bin(2)),
                                static_cast<float>(scale));
}
)cuda";

  class BinningJIT final : public DrpAlgos::JITKernelDescriptor {
  public:
    std::size_t payloadSize() const override { return m_payloadSize; }

    std::uint32_t ctxAbi() const override { return DrpKernelCtx::kABI; }

  protected:
    DrpAlgos::JITSource source() const override {
      return {
        kBinningSource,
        "drp_kernel",
        { { "DrpAlgos/algorithms/binning.cuh", DrpAlgos::kBinningKernelHeaderSource } }
      };
    }

    void onConfigure(const nlohmann::json& params) override {
      const auto shape = params.at("in_shape").get<std::vector<unsigned long long>>();
      const auto bin   = params.at("bin").get<std::vector<unsigned long long>>();
      if (shape.size() != 3 || bin.size() != 3) {
        throw std::invalid_argument("Binning 0.2.0: in_shape and bin must have 3 entries");
      }
      std::size_t n { 1 };
      for (std::size_t d = 0; d < 3; ++d) {
        if (bin[d] == 0 || shape[d] % bin[d] != 0) {
          throw std::invalid_argument("Binning 0.2.0: in_shape must be divisible by bin in every dimension");
        }
        n *= shape[d] / bin[d];
      }
      m_payloadSize = n * sizeof(float);
    }

  private:
    std::size_t m_payloadSize { 0 };
  };

  REGISTER_KERNEL("Binning", 0, 2, 0, BinningJIT)

} // anonymous namespace
