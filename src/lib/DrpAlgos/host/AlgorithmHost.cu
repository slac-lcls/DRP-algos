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
} // anonymous namespace

namespace DrpAlgos {
  AlgorithmHost::AlgorithmHost(const std::string& name,
                               const Version& version,
                               const std::string& soname,
                               const nlohmann::json& params,
                               HostConfig cfg)
    : m_cfg { cfg }
  {
    if (cfg.slots == 0 || cfg.inElems == 0) {
      throw std::invalid_argument("AlgorithmHost: slots and inElems must be > 0");
    }

    m_desc = KernelRegistry::instance().create(name, version, soname);
    m_desc->configure(params);
    m_payloadSize = m_desc->payloadSize();

    // Each slot's size header sits in the tail padding of the previous slot,
    // so the stride must leave room for it
    m_outStride = roundUp(m_payloadSize + kHeader, kAlign);

    try {
      check(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking), "cudaStreamCreate");
      check(cudaEventCreate(&m_start), "cudaEventCreate");
      check(cudaEventCreate(&m_stop), "cudaEventCreate");
      check(cudaMalloc(&m_in, std::size_t { cfg.slots } * cfg.inElems * sizeof(float)),
            "cudaMalloc(in)");
      check(cudaMalloc(&m_outBase, kHeader + std::size_t { cfg.slots } * m_outStride),
            "cudaMalloc(out)");
      check(cudaMalloc(&m_gate, sizeof(unsigned)), "cudaMalloc(gate)");
      check(cudaMalloc(&m_slot, sizeof(unsigned)), "cudaMalloc(slot)");
      check(cudaMallocHost(&m_one, sizeof(unsigned)), "cudaMallocHost(one)");

      *m_one = 1;
      check(cudaMemset(m_gate, 0, sizeof(unsigned)), "memset(gate)");
      check(cudaMemset(m_slot, 0, sizeof(unsigned)), "memset(slot)");

      const DrpKernelCtx ctx {
        m_gate,
        m_slot,
        m_in,
        cfg.inElems,
        m_outBase + kHeader,
        m_outStride,
        nullptr
      };

      // Record once, exactly as the DRP's recordGraph does
      cudaGraph_t graph { nullptr };
      check(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal),
            "BeginCapture");
      try {
        recordWithHandshake(*m_desc, ctx, m_stream);
      } catch (...) {
        cudaStreamEndCapture(m_stream, &graph);
        if (graph) {
          cudaGraphDestroy(graph);
        }
        throw;
      }
      check(cudaStreamEndCapture(m_stream, &graph), "EndCapture");
      const cudaError_t err { cudaGraphInstantiate(&m_graphExec, graph, 0) };
      cudaGraphDestroy(graph);
      check(err, "cudaGraphInstantiate");
    } catch (...) {
      release();
      throw;
    }
  }

  AlgorithmHost::~AlgorithmHost() { release(); }

  void AlgorithmHost::release() noexcept {
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
  }

  EventResult AlgorithmHost::process(const float* frame) {
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
  }

  BenchResult AlgorithmHost::benchmark(std::size_t iterations) {
    BenchResult bench;
    bench.iterations = iterations;
    if (iterations == 0) {
      return bench;
    }

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
  }
} // namespace DrpAlgos
