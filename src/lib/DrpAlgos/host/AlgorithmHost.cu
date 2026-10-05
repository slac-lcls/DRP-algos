#include "DrpAlgos/host/AlgorithmHost.hh"

#include "DrpAlgos/drp/KernelCtx.hh"
#include "DrpAlgos/drp/KernelDescriptor.hh"
#include "DrpAlgos/drp/KernelRegistry.hh"

#include <algorithm>
#include <stdexcept>
#include <string>

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
} // anonymous namespace

namespace DrpAlgos {
  AlgorithmHost::AlgorithmHost(const std::string& name,
                               const Version& version,
                               const std::string& soname,
                               const nlohmann::json& params,
                               HostConfig cfg)
    : m_cfg { cfg }
  {
    if (cfg.slots == 0 || cfg.inElems == 0 || cfg.workers == 0) {
      throw std::invalid_argument("AlgorithmHost: slots, inElems, and workers must be > 0");
    }
    if (cfg.slots < cfg.workers) {
      throw std::invalid_argument("AlgorithmHost: slots must be >= workers");
    }

    //m_desc = KernelRegistry::instance().create(name, version, soname);
    //m_desc->configure(params);
    m_workers.resize(cfg.workers);
    for (Worker& worker : m_workers) {
      worker.desc = KernelRegistry::instance().create(name, version, soname);
      worker.desc->configure(params);
    }
    m_payloadSize = m_workers.front().desc->payloadSize();

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

      //check(cudaMalloc(&m_slotGate, cfg.slots * sizeof(unsigned)), "cudaMalloc(slotGate)");

      m_slotStart.resize(cfg.slots, nullptr);
      m_slotStop.resize(cfg.slots, nullptr);
      for (unsigned slot = 0; slot < cfg.slots; ++slot) {
        check(cudaEventCreate(&m_slotStart[slot]), "cudaEventCreate(slotStart)");
        check(cudaEventCreate(&m_slotStop[slot]), "cudaEventCreate(slotStop)");
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
          recordWithHandshake(*worker.desc, ctx, worker.stream);
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

      //check(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking), "cudaStreamCreate");
      //check(cudaEventCreate(&m_start), "cudaEventCreate");
      //check(cudaEventCreate(&m_stop), "cudaEventCreate");
      //check(cudaMalloc(&m_in, std::size_t { cfg.slots } * cfg.inElems * sizeof(float)),
      //      "cudaMalloc(in)");
      //check(cudaMalloc(&m_outBase, kHeader + std::size_t { cfg.slots } * m_outStride),
      //      "cudaMalloc(out)");
      //check(cudaMalloc(&m_gate, sizeof(unsigned)), "cudaMalloc(gate)");
      //check(cudaMalloc(&m_slot, sizeof(unsigned)), "cudaMalloc(slot)");
      //check(cudaMallocHost(&m_one, sizeof(unsigned)), "cudaMallocHost(one)");

      //*m_one = 1;
      //check(cudaMemset(m_gate, 0, sizeof(unsigned)), "memset(gate)");
      //check(cudaMemset(m_slot, 0, sizeof(unsigned)), "memset(slot)");

      //const DrpKernelCtx ctx {
      //  m_gate,
      //  m_slot,
      //  m_in,
      //  cfg.inElems,
      //  m_outBase + kHeader,
      //  m_outStride,
      //  nullptr
      //};

      // Record once, exactly as the DRP's recordGraph does
      //cudaGraph_t graph { nullptr };
      //check(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal),
      //      "BeginCapture");
      //try {
      //  recordWithHandshake(*m_desc, ctx, m_stream);
      //} catch (...) {
      //  cudaStreamEndCapture(m_stream, &graph);
      //  if (graph) {
      //    cudaGraphDestroy(graph);
      //  }
      //  throw;
      //}
      //check(cudaStreamEndCapture(m_stream, &graph), "EndCapture");
      //const cudaError_t err { cudaGraphInstantiate(&m_graphExec, graph, 0) };
      //cudaGraphDestroy(graph);
      //check(err, "cudaGraphInstantiate");
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

    m_in = nullptr;
    m_outBase = nullptr;

    for (std::vector<cudaEvent_t>* evts : { &m_slotStart, &m_slotStop }) {
      for (cudaEvent_t& evt : *evts) {
        if (evt) {
          cudaEventDestroy(evt);
        }
      }
      evts->clear();
    }
    /*
    if (m_stream) {
      cudaStreamSynchronize(m_stream);
    }

    if (m_graphExec) {
      cudaGraphExecDestroy(m_graphExec);
    }
    if (m_start) {
      cudaEventDestroy(m_start);
    }
    if (m_stop) {
      cudaEventDestroy(m_stop);
    }
    if (m_in) {
      cudaFree(m_in);
    }
    if (m_outBase) {
      cudaFree(m_outBase);
    }
    if (m_gate) {
      cudaFree(m_gate);
    }
    if (m_slot) {
      cudaFree(m_slot);
    }
    if (m_one) {
      cudaFreeHost(m_one);
    }
    if (m_stream) {
      cudaStreamDestroy(m_stream);
    }

    m_graphExec = nullptr;
    m_start = m_stop = nullptr;
    m_in = nullptr;
    m_outBase = nullptr;
    m_gate = m_slot = m_one = nullptr;
    m_stream = nullptr;
    */
  }

  EventResult AlgorithmHost::process(const float* frame) {
    const unsigned slot { m_next++ % m_cfg.slots };
    Worker& worker { m_workers[static_cast<std::size_t>(m_next) % m_workers.size()] };

    float* const in { m_in + static_cast<std::size_t>(slot) * m_cfg.inElems };
    check(cudaMemcpyAsync(in,
                          frame,
                          m_cfg.inElems * sizeof(float),
                          cudaMemcpyHostToDevice,
                          worker.stream),
          "copy frame");

    armWorker<<<1, 1, 0, worker.stream>>>(worker.gate,
                                          worker.slot,
                                          slot,
                                          m_outBase,
                                          m_outStride);

    //if (m_cfg.maxDelayUs > 0.0f) {
    //  std::uniform_real_distribution<float> dist { 0.0f, m_cfg.maxDelayUs };
    //  delayStream<<<1, 1, 0, worker.stream>>>(static_cast<unsigned long long>(dist(m_rng) * 1000.0f));
    //}

    check(cudaEventRecord(m_slotStart[slot], worker.stream), "eventRecord");
    check(cudaGraphLaunch(worker.graphExec, worker.stream), "cudaGraphLaunch");
    check(cudaEventRecord(m_slotStop[slot], worker.stream), "eventRecord");
    check(cudaStreamSynchronize(worker.stream), "sync");
    //captureGate<<<1, 1, 0, worker.stream>>>(worker.gate, m_slotGate, slot);
    //check(cudaEventRecord(m_slotDone[slot], worker.stream), "eventRecord");
    check(cudaGetLastError(), "submit");

    EventResult result;
    check(cudaEventElapsedTime(&result.deviceMs, m_slotStart[slot], m_slotStop[slot]),
          "eventElapsedTime (worker)");

    return result;
    /*
    const unsigned slot { m_next++ % m_cfg.slots };
    const unsigned long long noSize { 0 };
    float* const in { m_in + std::size_t { slot } * m_cfg.inElems };
    // header precedes slot data
    unsigned char* const hdr { m_outBase + std::size_t { slot } * m_outStride };

    check(cudaMemcpyAsync(in,
                          frame,
                          m_cfg.inElems * sizeof(float),
                          cudaMemcpyHostToDevice,
                          m_stream),
          "copy frame");
    check(cudaMemcpyAsync(m_slot, &slot, sizeof(slot), cudaMemcpyHostToDevice, m_stream),
          "set slot");
    check(cudaMemcpyAsync(hdr,
                          &noSize,
                          sizeof(noSize),
                          cudaMemcpyHostToDevice,
                          m_stream),
          "clear size");
    check(cudaMemcpyAsync(m_gate,
                          m_one,
                          sizeof(unsigned),
                          cudaMemcpyHostToDevice,
                          m_stream),
          "arm gate");

    check(cudaEventRecord(m_start, m_stream), "eventRecord");
    check(cudaGraphLaunch(m_graphExec, m_stream), "cudaGraphLaunch");
    check(cudaEventRecord(m_stop, m_stream), "eventRecord");
    check(cudaStreamSynchronize(m_stream), "sync");

    unsigned gateAfter { 0 };
    unsigned long long size { 0 };
    check(cudaMemcpy(&gateAfter, m_gate, sizeof(gateAfter), cudaMemcpyDeviceToHost),
          "read gate");
    check(cudaMemcpy(&size, hdr, sizeof(size), cudaMemcpyDeviceToHost), "read size");

    if (gateAfter != 2) {
      throw std::runtime_error("Handshake not completed (gate = " +
                               std::to_string(gateAfter) + ")");
    }
    if (size > m_outStride - kHeader) {
      throw std::runtime_error("Size header (" + std::to_string(size) + " B) exceeds slot ("
                               + std::to_string(m_outStride - kHeader) + " B)");
    }

    EventResult result;
    result.payload.resize(size);
    check(cudaMemcpy(result.payload.data(), hdr + kHeader, size, cudaMemcpyDeviceToHost),
          "read payload");
    check(cudaEventElapsedTime(&result.deviceMs, m_start, m_stop), "eventElapsedTime");

    return result;
    */
  }

  BenchResult AlgorithmHost::benchmark(std::size_t iterations) {
    BenchResult bench;
    bench.iterations = iterations;
    if (iterations == 0) {
      return bench;
    }

    for (Worker& worker : m_workers) {
      check(cudaGraphLaunch(worker.graphExec, worker.stream),
            "cudaGraphLaunch (worker)");
      check(cudaStreamSynchronize(worker.stream), "sync (worker)");
    }

    return bench;
    /*
    check(cudaMemsetAsync(m_slot, 0, sizeof(unsigned), m_stream), "set slot");

    // Warm-up
    check(cudaMemcpyAsync(m_gate,
                          m_one,
                          sizeof(unsigned),
                          cudaMemcpyHostToDevice,
                          m_stream),
          "arm gate");
    check(cudaGraphLaunch(m_graphExec, m_stream), "cudaGraphLaunch");
    check(cudaStreamSynchronize(m_stream), "sync");

    float total { 0.0f };
    bench.minUs = 1e30f;
    for (std::size_t i = 0; i < iterations; ++i) {
      check(cudaMemcpyAsync(m_gate,
                            m_one,
                            sizeof(unsigned),
                            cudaMemcpyHostToDevice,
                            m_stream),
            "arm gate");
      check(cudaEventRecord(m_start, m_stream), "eventRecord");
      check(cudaGraphLaunch(m_graphExec, m_stream), "cudaGraphLaunch");
      check(cudaEventRecord(m_stop, m_stream), "eventRecord");
      check(cudaEventSynchronize(m_stop), "eventSync");

      float ms { 0.0f };
      check(cudaEventElapsedTime(&ms, m_start, m_stop), "eventElapsedTime");
      const float us { ms * 1000.0f };
      total += us;
      bench.minUs = std::min(bench.minUs, us);
      bench.maxUs = std::max(bench.maxUs, us);
    }
    bench.meanUs = total / static_cast<float>(iterations);

    return bench;
    */
  }
} // namespace DrpAlgos
