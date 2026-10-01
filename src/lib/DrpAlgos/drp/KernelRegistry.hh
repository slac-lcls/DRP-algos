#ifndef DRP_ALGOS_DRP_KERNEL_REGISTRY_HH
#define DRP_ALGOS_DRP_KERNEL_REGISTRY_HH

#include "DrpAlgos/drp/KernelDescriptor.hh"
#include "DrpAlgos/drp/Version.hh"
#include "DrpAlgos/export_macro.hh" // DRP_ALGOS_API for non-hidden symbols

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace DrpAlgos {
  using KernelFactoryFn = std::function<std::shared_ptr<IKernelDescriptor>()>;

  /**
   * Main registry for looking up DRP algorithms.
   *
   * A singleton instance lives in the core library - algorithms are registered,
   * and may live in multiple libraries which are independently loaded.
   * The algorithms are then looked-up using an algorithm name, version, and a
   * library name.
   */
  class DRP_ALGOS_API KernelRegistry {
  public:
    /**
     * Return a singleton instance of the registry.
     *
     * This function must be defined in a separate translation unit and included
     * in the core shared library to ensure all algorithms use the same registry.
     */
    static KernelRegistry& instance();

    /**
     * Register a specific named and versioned algorithm with the Registry.
     *
     * @param[in] name The name of the algorithm for lookup.
     * @param[in] Version The version of the algorithm for lookup.
     * @param[in] The factory function which returns the algorithm descriptor.
     */
    void registerKernel(const std::string& name,
                        const Version& version,
                        KernelFactoryFn factory);

    /**
     * Create an algorithm descriptor for the requested algorithm.
     *
     * The algorithm must've been registered with the Registry.
     *
     * @param[in] name The algorithm name.
     * @param[in] version The algorithm version.
     * @param[in] soname The library to find the algorithm in.
     * @returns The descriptor to run the algorithm.
     * @throws runtime_error for incorrect `soname`, an error was encountered loading
     *         the library, the algorithm version was not in the library, or the
     *         algorithm factory had an error during construction.
     */
    std::shared_ptr<IKernelDescriptor> create(const std::string& name,
                                              const Version& version,
                                              const std::string& soname);

    /**
     * Retrieve the list of all registered algorithms.
     *
     * @returns A vector of (name, Version) pairs of all registered algorithms.
     */
    std::vector<std::pair<std::string, Version>> registered() const;

    // No copies
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;

  private:
    KernelRegistry() = default;
    ~KernelRegistry() = default; // Libraries are intentionally never dlclose'd

    using Key = std::pair<std::string, Version>;

    std::map<Key, KernelFactoryFn> m_factories;     ///< Factories to create descriptors of algorithms
    std::map<std::string, void*> m_libs;            ///< Map of loaded shared libraries (soname: handle)
    std::map<std::string, std::string> m_libErrors; ///< Errors when loading a shared library
    std::vector<std::string> m_pendingErrors;       ///< Hold on to the errors in the interim as cannot throw during load
  };

  /**
   * Simple helper struct for the registration macro REGISTER_KERNEL.
   */
  struct KernelRegistrar {
    KernelRegistrar(const char* name, Version version, KernelFactoryFn factory) {
      KernelRegistry::instance().registerKernel(name, version, std::move(factory));
    }
  };
} // namespace DrpAlgos

#define DRP_CONCAT_IMPL(x, y) x##y
#define DRP_CONCAT(x, y) DRP_CONCAT_IMPL(x, y)

/**
 * @def REGISTER_KERNEL(NameStr, Major, Minor, Micro, DescriptorClass)
 * @brief A helper macro to include a DRP algorithm version in the registry.
 *
 * The macro must be included in the algorithm's library, at the namespace scope.
 *
 * @example REGISTER_KERNEL("Binning", 1, 0, 0, BinningV1) // Register Binning 1.0.0
 */
#define REGISTER_KERNEL(NameStr, Major, Minor, Micro, DescriptorClass)                            \
  static const ::DrpAlgos::KernelRegistrar                                                        \
  DRP_CONCAT(g_drp_registrar_, __LINE__)(NameStr,                                                 \
                                         ::DrpAlgos::Version { Major, Minor, Micro },             \
                                         []() -> std::shared_ptr<::DrpAlgos::IKernelDescriptor> { \
                                           return std::make_shared<DescriptorClass>();            \
                                         });
#endif // DRP_ALGOS_DRP_KERNEL_REGISTRY_HH
