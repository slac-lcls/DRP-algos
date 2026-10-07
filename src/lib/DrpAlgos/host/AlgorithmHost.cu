#include "DrpAlgos/host/AlgorithmHost.hh"

#include "DrpAlgos/drp/KernelCtx.hh"
#include "DrpAlgos/drp/KernelDescriptor.hh"
#include "DrpAlgos/drp/KernelRegistry.hh"

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  constexpr unsigned long long kHeader { sizeof(unsigned long long) }; ///< Size header per slot
  constexpr unsigned long long kAlign { 256 };                         ///< Slot alignment

  void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
  }

  unsigned long long roundUp(unsigned long long v, unsigned long long a) {
    return (v + a - 1) / a * a;
  }

  __global__ void drpFinalize(unsigned* const gate,
                              const unsigned* const slot,
                              unsigned char* const out,
                              unsigned long long const outStride,
                              unsigned long long const fixedSize) {
    if (gate && *gate != 1) {
      return;
    }

    if (fixedSize) {
      reinterpret_cast<unsigned long long*>(out + (*slot) * outStride)[-1] = fixedSize;
    }

    if (gate) {
      *gate = 2; // Hand off to the DRP's reducer loop
    }
  }

  void recordWithHandshake(DrpAlgos::IKernelDescriptor& desc,
                           DrpKernelCtx ctx,
                           cudaStream_t stream) {
    desc.launch(ctx, stream);

    drpFinalize<<<1, 1, 0, stream>>>(const_cast<unsigned*>(ctx.gate),
                                     ctx.slot,
                                     ctx.out,
                                     ctx.outStride,
                                     desc.variableSize() ? 0ULL : desc.payloadSize());

    const cudaError_t err { cudaGetLastError() };
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("DRP finalize launch failed: ") +
                               cudaGetErrorString(err));
    }
  }

  __global__ void armWorker(unsigned* const gate,
                            unsigned* const slotWord,
                            unsigned const slot,
                            unsigned char* const outBase,
                            unsigned long long const outStride) {
    *slotWord = slot;
    *reinterpret_cast<unsigned long long*>(outBase + slot * outStride) = 0; // Clear size header
    *gate = 1;
  }

  /**
   * Delay a stream for `ns` nanonseconds.
   *
   * Can use to simulate worker completing events out of order by perturbing the
   * duration.
   *
   * @param[in] ns The nanoseconds to delay the stream execution by.
   */
  __global__ void delayStream(unsigned long long const ns) {
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    unsigned long long t { t0 };

    while (t - t0 < ns) {
      __nanosleep(1000);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    }
  }

} // anonymous namespace

namespace DrpAlgos {
  AlgorithmHost::AlgorithmHost(const std::string& name,
                               const Version& version,
                               const std::string& soname,
                               const nlohmann::json& params,
                               HostConfig cfg)
    : m_cfg { cfg }
    , m_rng { cfg.seed }
  {
    if (cfg.slots == 0 || cfg.inElems == 0 || cfg.workers == 0) {
      throw std::invalid_argument("AlgorithmHost: slots, inElems, and workers must be > 0");
    }
    if (cfg.slots < cfg.workers) {
      throw std::invalid_argument("AlgorithmHost: slots must be >= workers");
    }

    m_workers.resize(cfg.workers);
    m_desc = KernelRegistry::instance().create(name, version, soname);
    m_desc->setWorkers(cfg.workers);
    m_desc->configure(params);
    m_payloadSize = m_desc->payloadSize();

    // Each slot's size header sits in the tail padding of the previous slot,
    // so the stride must leave room for it
    m_outStride = roundUp(m_payloadSize + kHeader, kAlign);

    try {
      const std::size_t inBytes {
        static_cast<std::size_t>(cfg.slots) * cfg.inElems * sizeof(float)
      };
      check(cudaMalloc(&m_in, inBytes), "cudaMalloc(in)");
      check(cudaMemset(m_in, 0, inBytes), "cudaMemset(in)");
      check(cudaMalloc(&m_outBase, kHeader + static_cast<std::size_t>(cfg.slots) * m_outStride),
            "cudaMalloc(out)");
      check(cudaMalloc(&m_slotGate, cfg.slots * sizeof(unsigned)),
            "cudaMalloc(slotGate)");

      m_slotStart.resize(cfg.slots, nullptr);
      m_slotStop.resize(cfg.slots, nullptr);
      m_slotDone.resize(cfg.slots, nullptr);
      for (unsigned slot = 0; slot < cfg.slots; ++slot) {
        check(cudaEventCreate(&m_slotStart[slot]), "cudaEventCreate(slotStart)");
        check(cudaEventCreate(&m_slotStop[slot]), "cudaEventCreate(slotStop)");
        check(cudaEventCreateWithFlags(&m_slotDone[slot], cudaEventDisableTiming),
              "cudaEventCreate (slotDone)");
      }

      for (Worker& worker : m_workers) {
        check(cudaStreamCreateWithFlags(&worker.stream, cudaStreamNonBlocking),
              "cudaStreamCreate(worker)");
        check(cudaMalloc(&worker.gate, sizeof(unsigned)), "cudaMalloc(worker gate)");
        check(cudaMalloc(&worker.slot, sizeof(unsigned)), "cudaMalloc(worker slot)");

        check(cudaMemset(worker.gate, 0, sizeof(unsigned)), "cudaMemset(worker gate)");
        check(cudaMemset(worker.slot, 0, sizeof(unsigned)), "cudaMemset(worker slot)");

        const DrpKernelCtx ctx {
          worker.gate,
          worker.slot,
          m_in,
          cfg.inElems,
          m_outBase + kHeader,
          m_outStride,
          nullptr
        };

        cudaGraph_t graph { nullptr };
        check(cudaStreamBeginCapture(worker.stream, cudaStreamCaptureModeThreadLocal),
              "BeginCapture (worker)");
        try {
          recordWithHandshake(*m_desc, ctx, worker.stream);
        } catch (...) {
          cudaStreamEndCapture(worker.stream, &graph);
          if (graph) {
            cudaGraphDestroy(graph);
          }
          throw;
        }

        check(cudaStreamEndCapture(worker.stream, &graph), "EndCapture (worker)");
        const cudaError_t err { cudaGraphInstantiate(&worker.graphExec, graph, 0) };
        cudaGraphDestroy(graph);
        check(err, "cudaGraphInstantiate (worker)");
      }
    } catch (...) {
      release();
      throw;
    }
  }

  AlgorithmHost::~AlgorithmHost() { release(); }

  void AlgorithmHost::release() noexcept {
    for (Worker& worker : m_workers) {
      if (worker.stream) {
        cudaStreamSynchronize(worker.stream);
      }
    }

    for (Worker& worker : m_workers) {
      if (worker.graphExec) {
        cudaGraphExecDestroy(worker.graphExec);
      }

      if (worker.gate) {
        cudaFree(worker.gate);
      }

      if (worker.slot) {
        cudaFree(worker.slot);
      }

      if (worker.stream) {
        cudaStreamDestroy(worker.stream);
      }

      worker.graphExec = nullptr;
      worker.gate = nullptr;
      worker.slot = nullptr;
      worker.stream = nullptr;
    }

    if (m_in) {
      cudaFree(m_in);
    }

    if (m_outBase) {
      cudaFree(m_outBase);
    }

    if (m_slotGate) {
      cudaFree(m_slotGate);
    }

    if (m_frame) {
      cudaFree(m_frame);
    }

    m_in = nullptr;
    m_outBase = nullptr;
    m_slotGate = nullptr;
    m_frame = nullptr;

    for (std::vector<cudaEvent_t>* evts : { &m_slotStart, &m_slotStop, &m_slotDone }) {
      for (cudaEvent_t& evt : *evts) {
        if (evt) {
          cudaEventDestroy(evt);
        }
      }
      evts->clear();
    }
  }

  void AlgorithmHost::submit(std::size_t event, const float* devFrame) {
    const unsigned slot { static_cast<unsigned>(event % m_cfg.slots) };
    Worker& worker { m_workers[event % m_workers.size()] };

    if (devFrame) {
      float* const in { m_in + static_cast<std::size_t>(slot) * m_cfg.inElems };
      check(cudaMemcpyAsync(in,
                            devFrame,
                            m_cfg.inElems * sizeof(float),
                            cudaMemcpyHostToDevice,
                            worker.stream),
            "copy frame (worker)");
    }

    armWorker<<<1, 1, 0, worker.stream>>>(worker.gate,
                                          worker.slot,
                                          slot,
                                          m_outBase,
                                          m_outStride);

    // Add perturbation (optionally) to simulate out-of-order completeion
    if (m_cfg.maxDelayUs > 0.0f) {
      std::uniform_real_distribution<float> dist { 0.0f, m_cfg.maxDelayUs };
      // Stream delay is in nanoseconds, so convert
      delayStream<<<1, 1, 0, worker.stream>>>(static_cast<unsigned long long>(dist(m_rng) * 1000.0f));
    }

    check(cudaEventRecord(m_slotStart[slot], worker.stream), "eventRecord (worker)");
    check(cudaGraphLaunch(worker.graphExec, worker.stream), "cudaGraphLaunch (worker)");
    check(cudaEventRecord(m_slotStop[slot], worker.stream), "eventRecord (worker)");

    check(cudaMemcpyAsync(&m_slotGate[slot],
                          worker.gate,
                          sizeof(unsigned),
                          cudaMemcpyDeviceToDevice,
                          worker.stream),
          "cudaMemcpy (worker gate to AlgHost)");

    check(cudaEventRecord(m_slotDone[slot], worker.stream), "eventRecord");
    check(cudaGetLastError(), "submit");
  }

  EventResult AlgorithmHost::collect(std::size_t event) {
    const unsigned slot { static_cast<unsigned>(event % m_cfg.slots) };
    check(cudaEventSynchronize(m_slotDone[slot]), "eventSync");

    Worker& worker { m_workers[event % m_workers.size()] };

    EventResult result;
    check(cudaEventElapsedTime(&result.deviceMs, m_slotStart[slot], m_slotStop[slot]),
          "eventElapsedTime (worker)");

    unsigned gateAfter { 0 };
    unsigned long long size { 0 };
    unsigned char* const hdr { m_outBase + static_cast<std::size_t>(slot) * m_outStride };
    check(cudaMemcpy(&gateAfter, &m_slotGate[slot], sizeof(gateAfter), cudaMemcpyDeviceToHost),
          "read gate (worker)");
    check(cudaMemcpy(&size, hdr, sizeof(size), cudaMemcpyDeviceToHost),
          "read size (worker)");

    if (gateAfter != 2) {
      throw std::runtime_error("Handshake not completed (gate = " + std::to_string(gateAfter) + ")");
    }
    if (size > m_outStride - kHeader) {
      throw std::runtime_error("Size header (" + std::to_string(size) + " B) exceeds slot (" +
                               std::to_string(m_outStride - kHeader) + " B)");
    }

    check(cudaGetLastError(), "submit");

    result.payload.resize(size);
    check(cudaMemcpy(result.payload.data(), hdr + kHeader, size, cudaMemcpyDeviceToHost),
          "read payload (worker)");

    return result;
  }

  EventResult AlgorithmHost::process(const float* frame) {
    static bool initialized { false };
    if (!initialized) {
      if (m_frame) {
        cudaFree(m_frame);
        m_frame = nullptr;
      }

      check(cudaMalloc(&m_frame, m_cfg.inElems * sizeof(float)),
            "cudaMalloc (create devFrame)");
      initialized = true;
    }
    check(cudaMemcpyAsync(m_frame,
                          frame,
                          m_cfg.inElems * sizeof(float),
                          cudaMemcpyHostToDevice),
          "cudaMemcpy (move frame to device)");

    std::size_t event { static_cast<std::size_t>(m_next++) };
    submit(event, m_frame);

    return collect(event);
  }

  BenchResult AlgorithmHost::benchmark(std::size_t iterations) {
    BenchResult bench;
    bench.iterations = iterations;
    if (iterations == 0) {
      return bench;
    }

    for (std::size_t warmup = 0; warmup < m_workers.size(); ++warmup) {
      submit(warmup, m_frame);
      (void)collect(warmup);
    }

    bench.minUs = 1e30f;
    double totalUs { 0.0 };
    const auto t0 { std::chrono::steady_clock::now() };
    for (std::size_t event = 0; event < iterations; ++event) {
      submit(event, m_frame);
      auto res = collect(event);

      const float iterUs { res.deviceMs * 1000.0f };
      totalUs += iterUs;
      bench.minUs = std::min(bench.minUs, iterUs);
      bench.maxUs = std::max(bench.maxUs, iterUs);
    }
    const auto t1 { std::chrono::steady_clock::now() };

    bench.meanUs = static_cast<float>(totalUs / static_cast<double>(iterations));

    const double wallUs { std::chrono::duration<double, std::micro>(t1 - t0).count() };
    bench.rateKHz = static_cast<float>(1000.0 * static_cast<double>(iterations) / wallUs);

    return bench;
  }
} // namespace DrpAlgos
