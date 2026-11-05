#ifndef DRP_ALGOS_DRP_JIT_KERNEL_DESCRIPTOR_HH
#define DRP_ALGOS_DRP_JIT_KERNEL_DESCRIPTOR_HH

#include "DrpAlgos/KernelCtx.hh"
#include "DrpAlgos/drp/KernelDescriptor.hh"
#include "DrpAlgos/export_macro.hh" // DRP_ALGOS_API for non-hidden symbols

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

namespace DrpAlgos {
  /**
   * A header made available to a JIT-compiled algorithm's `include`.
   *
   * Headers should (in general) only include other DrpAlgos headers.
   */
  struct JITHeader {
    const char* name;   ///< Include name, e.g. "DrpAlgos/algorithms/binning.cuh"
    const char* source; ///< Header contents (usually embedded at build time)
  };

  /**
   * C++ source code for a JIT-compiled DRP algorithm.
   *
   * Currently, only inclusion of DrpAlgos headers is allowed.
   */
  struct JITSource {
    std::string code;                   ///< The actual source code.
    std::string entry { "drp_kernel" }; ///< The name of the (extern C) kernel entry-point
    std::vector<JITHeader> headers {};  ///< Extra included headers.
  };

  /**
   * Opaque handle for a compiled and loaded kernel.
   */
  struct JITKernel;

  /**
   * Base descriptor class for a DRP algorithm which will be JIT compiled.
   */
  class DRP_ALGOS_API JITKernelDescriptor : public IKernelDescriptor {
  public:
    void configure(const char* params, std::size_t paramsLen) final;

    void launch(const void* ctx, cudaStream_t stream) final;

    /**
     * Construct the prelude for baking parameters into the kernel.
     *
     * When using JIT compilation, parameters are embedded as compile-time
     * constexpr values into a `jit_params` namespace for kernel usage. The
     * actual kernel source code can refer to them directly then, without
     * having to access them as runtime values through the KernelCtx.
     * - Literal scalars are directly retrieved by the name of the parameter.
     * - Literal array types are retrieved via constexpr function call.
     *   E.g. `my_array(1)` retrieves the second value of the parameter `my_array`.
     *
     * @param[in] params The parameter values to bake (e.g. from configDB).
     * @returns The prelud with constexpr parameters embedded into the `jit_params`
     *          namespace.
     * @throws invalid_argument if `params` is not a JSON object (dictionary), or
     *         if the parameter name cannot be used as a C++ identifier/contains
     *         control characters.
     */
    static std::string makePrelude(const nlohmann::json& params);

    std::uint32_t ctxAbi() const override { return DRP_KERNEL_CTX_ABI; } // From header, via build args

  protected:
    /**
     * The kernel source code.
     *
     * The source code may come from the configDB, or be compiled into the library
     * where the slim descriptor lives.
     *
     * @returns The source code (+ the entry point name).
     */
    virtual JITSource source() const = 0;

    /**
     * Validate provided parameters and compute information needed for payloadSize.
     *
     * NOTE: This function can throw on error during validation.
     *
     * @param[in] params The configDB parameters for the algorithm.
     */
    virtual void onConfigure(const nlohmann::json& params) {}

    /**
     * The thread-block size to use for kernel launches.
     *
     * @returns The thread-block size to use for kernel launches.
     */
    virtual unsigned blockSize() const { return 256; }

    /**
     * The grid size for kernel launches.
     *
     * Kernels currently should use a grid-stride loop for traversal.
     *
     * @returns The (1D) grid size for kernel launches.
     */
    virtual unsigned gridSize() const { return m_defaultGrid; }

  private:
    std::shared_ptr<const JITKernel> m_kernel;
    unsigned m_defaultGrid { 0 };
  };

} // namespace DrpAlgos

#endif // DRP_ALGOS_DRP_JIT_KERNEL_DESCRIPTOR_HH
