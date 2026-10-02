#ifndef DRP_ALGOS_DRP_TYPED_KERNEL_DESCRIPTOR_HH
#define DRP_ALGOS_DRP_TYPED_KERNEL_DESCRIPTOR_HH

#include "DrpAlgos/drp/KernelCtx.hh"
#include "DrpAlgos/drp/KernelDescriptor.hh"
#include "DrpAlgos/export_macro.hh" // DRP_ALGOS_API for non-hidden symbols

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <type_traits>

namespace DrpAlgos {
  /**
   * A DRP algorithm descriptor templated on the parameter struct it requires.
   *
   * The parameter type used in instantiating the descriptor will be constructed
   * using run-time values retrieved from the configDB (this is done automatically).
   *
   * @note Because the parameters will be moved onto device, the type ParamT must be
   *       trivially copyable.
   *
   * @tparam ParamT The type of the parameter struct used to configure the algorithm.
   */
  template <typename ParamT>
  class DRP_ALGOS_API TypedKernelDescriptor : public IKernelDescriptor {
    static_assert(std::is_trivially_copyable_v<ParamT>,
                  "ParamT is memcpy'd to the GPU: must be trivially copyable!");

  public:
    TypedKernelDescriptor() = default;
    TypedKernelDescriptor(const TypedKernelDescriptor&) = delete;
    TypedKernelDescriptor& operator=(const TypedKernelDescriptor&) = delete;

    ~TypedKernelDescriptor() override {
      if (m_devParams) {
        cudaFree(m_devParams);
      }
    }

    /**
     * Run algorithm configuration with provided parameter sets.
     *
     * This base class unpacks the parameters into the `m_params` member.
     * Sub-classes should use the `onConfigure` routine if needing to do
     * additional validation or checks.
     *
     * @param[in] params The parameter set to use for the algorithm.
     */
    void configure(const nlohmann::json& params) final {
      // The parameter struct should have been registered with nlohmann JSON
      // where the algorithm was registered with the registry.
      m_params = params.get<ParamT>();

      // Specific derived classes can do checks and validations here or throw
      onConfigure();

      // Move parameter set into device memory
      if (!m_devParams) {
        check(cudaMalloc(&m_devParams, sizeof(ParamT)), "cudaMalloc(params)");
      }

      check(cudaMemcpy(m_devParams, &m_params, sizeof(ParamT), cudaMemcpyHostToDevice),
            "cudaMemcpy(params)");
    }

    const ParamT& params() const { return m_params; }

    ParamT& params() { return m_params; }

  protected:
    virtual void onConfigure() {}

    DrpKernelCtx withParams(DrpKernelCtx ctx) const {
      ctx.params = m_devParams;

      return ctx;
    }

    ParamT m_params {}; ///< Host-side parameters constructed by nlohmann/json

    static void check(cudaError_t err, const char* what) {
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
      }
    }

  private:
    void* m_devParams { nullptr }; ///< Device-side parameters copy
  };
} // namespace DrpAlgos

#endif // DRP_ALGOS_DRP_TYPED_KERNEL_DESCRIPTOR_HH
