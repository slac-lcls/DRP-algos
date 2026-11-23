#ifndef DRP_ALGOS_DRP_REQUIREMENTS_HH
#define DRP_ALGOS_DRP_REQUIREMENTS_HH

#include <cstdint>

namespace DrpAlgos {
  /**
   * Requirements the descriptors has of the host that will launch it.
   *
   * Some algorithms may have hard requirements of information provided by the
   * DRP/AlgorithmHost. E.g., certain information on event distribution. The
   * requirements can be queried via the same `query` API used by Metrics:
   * - `query(Requirements_v1::id)`
   *
   * The response is valid from `configure` until the next reconfigure. A `nullptr`
   * response indicates no requirements.
   *
   * The struct contains the following fields:
   * - `required`: the luancher must ensure every bit is set, or otherwise, refuse
   *   to run the algorithm if it does not support any of the bits (including those
   *   it may not recognize).
   * - `hints`: the launcher can use these for additional guidance, but may choose
   *   to ignore them.
   *
   * @note Over time, if new needs arise, they are added as new bits. Any new values
   *       should be added at the end of the struct. `size` indicates to the launcher
   *       how much was filled in by the Descriptor (e.g., if not all values were used).
   */
  struct Requirements_v1 {
    static constexpr std::uint32_t id { 0x52455131 }; ///< Identifier for requirements - `REQ1` as bytes

    // `required` bits
    static constexpr std::uint32_t kVetoedEvents { 1u << 0 }; ///< Dispatch vetoed L1Accepts (gate 3)

    std::uint32_t size     { sizeof(Requirements_v1) };
    std::uint32_t required { 0 };
    std::uint32_t hints    { 0 };
  };
} // namespace DrpAlgos

#endif // DRP_ALGOS_DRP_REQUIREMENTS_HH
