#ifndef DRP_ALGOS_HOST_ALGORITHM_HOST_HH
#define DRP_ALGOS_HOST_ALGORITHM_HOST_HH

#include "DrpAlgos/drp/KernelDescriptor.hh"
#include "DrpAlgos/drp/Version.hh"
#include "DrpAlgos/export_macro.hh" // DRP_ALGOS_API for non-hidden symbols

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace DrpAlgos {
  /**
   * Ring configuration for the standalone host.
   */
  struct HostConfig {
    unsigned slots { 4 };             ///< Ring depth (as with DRP worker buffers)
    unsigned long long inElems { 0 }; ///< Floats per input slot (one calibrated frame)
  };

  /**
   * Result of running one event through the algorithm.
   */
  struct EventResult {
    std::vector<std::uint8_t> payload; ///< Output bytes, as the DRP would record them
    float deviceMs { 0.0f };           ///< Graph execution time for this event
  };

  /**
   * Timing record for graph per-event executions.
   */
  struct BenchResult {
    std::size_t iterations { 0 }; ///< Number of executions/"events"
    float meanUs { 0.0f };        ///< Mean graph execution time in microseconds
    float minUs { 0.0f };         ///< Minimum graph execution time in microseconds
    float maxUs { 0.0f };         ///< Maximum graph execution time in microseconds
  };

  /**
   * Standalone harness to host and run a single algorithm.
   *
   * The AlgorithmHost harness is setup to follow the execution model used by the
   * DAQ's DRP. As with execution run through the DRP, an algorithm run via
   * AlgorithmHost will be created via the KernelRegistry, and configured using
   * parameters coming from the same schema. The input/output organization is setup
   * to mimic the DRP's ring buffers. It contains:
   * - Input and output rings with size headers preceeding each output slot.
   * - A "gating handshake" that mimics the DRP handshake.
   * - Execution via CUDA graph, recorded once, and replayed on each "event".
   */
  class DRP_ALGOS_API AlgorithmHost {
  public:
    AlgorithmHost(const std::string& name,
                  const Version& version,
                  const std::string& soname,
                  const nlohmann::json& params,
                  HostConfig cfg);
    ~AlgorithmHost();

    // No copy
    AlgorithmHost(const AlgorithmHost&) = delete;
    AlgorithmHost& operator=(const AlgorithmHost&) = delete;

    std::size_t payloadSize() const { return m_payloadSize; }

    bool variableSize() const { return m_desc->variableSize(); }

    /**
     * Process a single event analogously to the DRP.
     *
     * For each event, the frame will be copied into an available ring slot,
     * the reduction graph will be run, and then bytes are returned.
     *
     * @param[in] frame Host pointer to `inElems` floats.
     * @returns An EventResult from the processing.
     * @throws runtime_error on CUDA errors, or if the handshake did not complete.
     */
    EventResult process(const float* frame);

    /**
     * Replay the graph repeatedly on one slot, re-arming the gate each time.
     *
     * @param[in] iterations Number of timed replays (after one warm-up).
     * @returns The full benchmarking result.
     */
    BenchResult benchmark(std::size_t iterations);

  private:
    void release() noexcept;

    HostConfig m_cfg;
    std::shared_ptr<IKernelDescriptor> m_desc;
    std::size_t m_payloadSize { 0 };
    unsigned long long m_outStride { 0 };
    unsigned m_next { 0 };

    cudaStream_t m_stream { nullptr };
    cudaGraphExec_t m_graphExec { nullptr };
    cudaEvent_t m_start { nullptr };
    cudaEvent_t m_stop { nullptr };

    float* m_in { nullptr };              ///< Input ring (slots * inElems floats)
    unsigned char* m_outBase { nullptr }; ///< Output ring incl. leading header space
    unsigned* m_gate { nullptr };         ///< Handshake word (device)
    unsigned* m_slot { nullptr };         ///< Current slot index (device)
    unsigned* m_one { nullptr };          ///< Pinned host word holding 1, for re-arming
  };
} // namespace DrpAlgos

#endif // DRP_ALGOS_HOST_ALGORITHM_HOST_HH
