#include "DrpAlgos/algorithms/roibin.cuh"
#include "DrpAlgos/drp/KernelRegistry.hh"
#include "DrpAlgos/drp/Metrics.hh"
#include "DrpAlgos/drp/TypedKernelDescriptor.hh"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <cuda/atomic>
#include <cuda/std/array>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
  using DrpAlgos::algorithms::AsicPlacement;

  constexpr unsigned kMaxAsics { 16 }; ///< Maximum ASICs in one panel's ASIC grid
  constexpr unsigned kMaxROIs { 5 };   ///< Maximum allowed ROIs for a single panel/module

  /**
   * Parameters for ROIBin algorithm.
   *
   * Shapes and offsets are in panel coordinates (plane, row, col). The calibrated
   * input is ASIC-major; `asic_grid` and `asic_layout` describe how the panel is
   * tiled by ASICs (see `panel_to_buffer`). An ROI with any zero extent is empty:
   * only the header is output, so time bins stay in step with other panels.
   */
  struct ROIBinParams {
    std::uint32_t in_shape[3];             ///< (planes, rows, cols) of the input panel
    std::uint32_t num_rois;                ///< Total number of ROIs
    std::uint32_t roi_start[kMaxROIs][3];  ///< ROI origin per dimension
    std::uint32_t roi_shape[kMaxROIs][3];  ///< ROI extent per dimension (any 0 = empty)
    std::uint32_t bin[kMaxROIs][3];        ///< Spatial bin factor per dimension
    std::uint32_t frames;                  ///< Consecutive reduced events summed per output (1 = off)
    std::uint32_t use_threshold[kMaxROIs]; ///< Non-zero: pixels below `threshold` are zeroed before binning
    float threshold[kMaxROIs];             ///< Per-pixel threshold
    float scale[kMaxROIs];                 ///< Multiplier applied to each spatial bin
    std::uint32_t asic_grid[2];            ///< (rows, cols) of ASICs tiling the panel
    AsicPlacement asic_layout[kMaxAsics];  ///< Placement per grid position, row-major
  };

  /**
   * Parse ROIBin parameters.
   *
   * `asic_layout` is a flat list of (asic, flip_rows, flip_cols) triples, one per
   * ASIC grid position (row-major), so its length is 3 * asic_grid[0] * asic_grid[1].
   */
  void from_json(const nlohmann::json& j, ROIBinParams& p) {
    j.at("in_shape").get_to(p.in_shape);
    j.at("num_rois").get_to(p.num_rois);
    // Filled below based on number of ROIs
    // - roi_start
    // - roi_shape
    // - bin
    j.at("frames").get_to(p.frames);
    // Filled below based on number of ROIs
    // - use_threshold
    // - threshold
    // - scale
    j.at("asic_grid").get_to(p.asic_grid);

    if (p.num_rois > kMaxROIs) {
      throw std::invalid_argument("[ROIBin 0.2.0] A maximum of 5 ROIs per-panel is allowed!");
    }

    // Fill ROI-num-dependent arrays manually, so not all have to be passed en empty
    for (std::size_t i = 0; i < p.num_rois; ++i) {
      const auto& roi_start { j.at("roi_start").at(i) };
      const auto& roi_shape { j.at("roi_shape").at(i) };
      const auto& bin { j.at("bin").at(i) };
      for (std::size_t k = 0; k < 3; ++k) {
        p.roi_start[i][k] = roi_start.at(k).get<std::uint32_t>();
        p.roi_shape[i][k] = roi_shape.at(k).get<std::uint32_t>();
        p.bin[i][k] = bin.at(k).get<std::uint32_t>();
      }

      p.use_threshold[i] = j.at("use_threshold").at(i).get<std::uint32_t>();
      p.threshold[i] = j.at("threshold").at(i).get<float>();
      p.scale[i] = j.at("scale").at(i).get<float>();
    }

    const std::uint64_t n_asics { std::uint64_t { p.asic_grid[0] } * p.asic_grid[1] };
    if (n_asics == 0 || n_asics > kMaxAsics) {
      throw std::invalid_argument("[ROIBin 0.2.0] asic_grid must hold 1 to 16 ASICs");
    }

    const nlohmann::json& layout { j.at("asic_layout") };
    if (!layout.is_array() || layout.size() != 3 * n_asics) {
      throw std::invalid_argument("[ROIBin 0.2.0] asic_layout must hold one "
                                  "(asic, flip_rows, flip_cols) triple per ASIC");
    }
    for (std::size_t i = 0; i < n_asics; ++i) {
      p.asic_layout[i] = AsicPlacement {
        layout.at(3 * i).get<unsigned>(),
        layout.at(3 * i + 1).get<unsigned>() != 0,
        layout.at(3 * i + 2).get<unsigned>() != 0
      };
    }
  }

  /**
   * A small header added to the reduced output for to facilitate bin tracking.
   *
   * @note The header is 16-byte aligned for global frame alignment.
   */
  struct alignas(16) ROIBinHeader {
    std::uint64_t bin;         ///< Time-bin index since Configure (kNoBin when frames == 1)
    std::uint32_t frames;      ///< Frames summed into this output
    std::uint32_t flags;       ///< kFlag* bits
    std::uint64_t presence[2]; ///< Bit mask of event inclusion
  };
  static_assert(sizeof(ROIBinHeader) == 32);

  constexpr std::uint64_t kNoBin { ~0ULL };         ///< Sentinel for time binning disabled (frames == 1)
  constexpr std::uint32_t kFlagOrderSuspect { 1u }; ///< Staged slot indices not monotonic
  constexpr unsigned kMaxWorkers { 64 };            ///< Maximum number of concurrent reducers supported
  constexpr unsigned kBinSlots { 4 };               ///< Bins in flight; correctness holds for any value >= 1

  void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("ROIBin 0.2.0: ") + what + ": " + cudaGetErrorString(err));
    }
  }

  /// ---- RAII Helpers ---- ///

  // For unique_ptr deallocation
  struct CudaFree {
    void operator()(void* ptr) const noexcept { cudaFree(ptr); }
  };

  template <typename T>
  using DeviceUnique = std::unique_ptr<T, CudaFree>;

  // Allocate a device-side unique_ptr with error checking
  template <typename T> DeviceUnique<T>
  deviceAlloc(std::size_t count, const char* what) {
    using Elem = std::remove_extent_t<T>;
    Elem* ptr { nullptr };
    check(cudaMalloc(&ptr, count * sizeof(Elem)), what);

    return DeviceUnique<T> { ptr };
  }

  /// ---- Time binning state coordination ---- ///

  /**
   * Counters shared by all Reducer workers. Lives in device memory.
   */
  struct FrameBinCounters {
    cuda::std::array<unsigned long long, kBinSlots> owner;   ///< Bin allowed to use each slot
    cuda::std::array<unsigned, kBinSlots> filled;            ///< Frames staged into each slot
    cuda::std::array<unsigned long long[2], kBinSlots> mask;

    unsigned long long binsDropped;   ///< Bins lost when a vetoed event was on the emit event
    unsigned long long framesDropped; ///< Total events lost when a bin is dropped
    unsigned long long binsPartial;   ///< Total bins emitted without all frames (due to vetoed events)
  };

  /// Device view of the time-binning state shared by all reducer workers.
  /**
   * Cross-instance tracker of the current time-binning state.
   *
   * The FrameBinState is shared between workers running concurrently.
   */
  struct FrameBinState {
    FrameBinCounters* counters; ///< Shared counters (device)
    unsigned* index;            ///< [kBinSlots * frames] DRP slot index of each staged frame
    float* staging;             ///< [kBinSlots * frames * outElems]
  };

  /**
   * Constants set for each independent Reducer worker.
   *
   * @note These are fixed at graph capture time.
   */
  struct WorkerBinConsts {
    unsigned worker;   ///< Worker ID - user for buffer access, as dispatch is round-robin.
    unsigned nWorkers; ///< Worker's copy of the total worker count.
    unsigned frames;   ///< Frames to bin in time
    unsigned outElems; ///< Size of the binned output (including pixel binning, ROI etc.)
  };

  /**
   * Map an event's sequence number to the staging slot of its time bin.
   *
   * `frames` events at a time form a full time-bin. Bins are cycled through
   * a series of `kBinSlots` staging slots, where a slot is reused only after
   * the current bin has been completed (this is tracked in the FrameBinCounters
   * owner state).
   *
   * @param[in] seq The global sequence number of the event
   * @param[in] worker_consts The at-launch configured worker constants.
   * @returns The staging slot index in [0, kBinSlots).
   */
  __device__ inline unsigned binSlot(unsigned long long seq,
                                     const WorkerBinConsts& worker_consts) {
    return static_cast<unsigned>((seq / worker_consts.frames) % kBinSlots);
  }

  /**
   * Determine output buffer.
   *
   * This is an adjustment from the standard function provided by KernelCtx.hh
   * to simplify handling of the additional header that is included.
   *
   * @param[in] evt The input DRP event.
   * @returns The location to write reduced output to.
   */
  __device__ inline float* outData(const DrpEvent& evt) {
    return reinterpret_cast<float*>(evt.out + sizeof(ROIBinHeader));
  }

  /**
   * Run ROIing and pixel-binning without binning on time-axis.
   *
   * @note This function is the internal implementation used by either of two paths:
   *       - The direct path without time binning
   *       - The path containing time binning, in addition to ROI and pixel binning.
   *
   * @param[in] params The configured ROI and binning parameters.
   * @param[in] in The input calibrated data.
   * @param[out] out Where to write output reduced data.
   */
  __device__ inline void roibinFrame(const ROIBinParams& params,
                                     const float* in,
                                     float* out) {
    unsigned offset { 0 }; // Offset to output in float elements (not bytes)
    for (unsigned n = 0; n < params.num_rois; ++n) {
      const unsigned oa { params.roi_shape[n][0] / params.bin[n][0] };
      const unsigned oy { params.roi_shape[n][1] / params.bin[n][1] };
      const unsigned ox { params.roi_shape[n][2] / params.bin[n][2] };
      const unsigned num_elem { oa * oy * ox };

      DrpAlgos::algorithms::roibin(in,
                                   out + offset,
                                   params.in_shape[1],
                                   params.in_shape[2],
                                   params.in_shape[1] / params.asic_grid[0],
                                   params.in_shape[2] / params.asic_grid[1],
                                   params.asic_grid[1],
                                   params.asic_layout,
                                   params.roi_start[n][0],
                                   params.roi_start[n][1],
                                   params.roi_start[n][2],
                                   oy,
                                   ox,
                                   num_elem,
                                   params.bin[n][0],
                                   params.bin[n][1],
                                   params.bin[n][2],
                                   params.use_threshold[n] != 0,
                                   params.threshold[n],
                                   params.scale[n]);

      offset += num_elem;
    }
  }

  /**
   * DRP ROI and pixel reduction without time binning.
   *
   * For the case where the timing axis bin factor is 1 (frames == 1), this kernel
   * performs only the ROIing and pixel binning (+ threshold), avoiding the additional
   * overhead and complexity of shared state.
   *
   * @note This function calls the internal ROIbin implementation and also handles the
   *       DRP transitions and header inclusion.
   *
   * @param[in] ctx The DRP algorithm kernel context.
   */
  __global__ void roibinDirect(DrpKernelCtx ctx) {
    const auto evt { drp_event(ctx) };
    if (!evt.is_active) {
      return;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      ROIBinHeader* hdr { evt.out_as<ROIBinHeader>() };
      hdr->bin = kNoBin;
      hdr->frames = 1;
      hdr->flags = 0;
      hdr->presence[0] = 1;
      hdr->presence[1] = 0;
    }

    roibinFrame(drp_params<ROIBinParams>(ctx), evt.in, outData(evt));
  }

  /**
   * Wait until the event's bin is ready for staging.
   *
   * @note Should run on 1 thread.
   *
   * @param[in] ctx The DRP algorithm kernel context.
   * @param[in] state The global algorithm state view.
   * @param[in] worker_consts The at-launch configured worker constants.
   */
  __global__ void frameBinBegin(DrpKernelCtx ctx,
                                FrameBinState state,
                                WorkerBinConsts worker_consts) {
    const auto evt { drp_event(ctx) };

    const unsigned long long seq { evt.seq };
    const unsigned slot { binSlot(seq, worker_consts) };


    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> owner {
      state.counters->owner[slot]
    };
    while (owner.load(cuda::std::memory_order_acquire) != seq / worker_consts.frames) {
      __nanosleep(128);
    }

    if (!evt.is_active) {
      // If the event is inactive, still count it as seen.
      // The mask below records actual presence in the time bin.
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> filled {
        state.counters->filled[slot]
      };
      auto last { filled.fetch_add(1u, cuda::std::memory_order_release) };

      return;
    }

    unsigned long long maskSlot { (seq % worker_consts.frames) / 64 };
    atomicOr(&state.counters->mask[slot][maskSlot],
             (1ULL << ((seq % worker_consts.frames) % 64)));

    state.index[slot * worker_consts.frames + seq % worker_consts.frames] = *ctx.slot;
  }

  /**
   * Reduce the event frame into the staging position.
   *
   * @note Should run across the full grid.
   *
   * @param[in] ctx The DRP algorithm kernel context.
   * @param[in] state The global algorithm state view.
   * @param[in] worker_consts The at-launch configured worker constants.
   */
  __global__ void frameBinStage(DrpKernelCtx ctx,
                                FrameBinState state,
                                WorkerBinConsts worker_consts) {
    const auto evt { drp_event(ctx) };
    if (!evt.is_active) {
      return;
    }

    const unsigned long long seq { evt.seq };
    const unsigned slot { binSlot(seq, worker_consts) };
    float* dst {
      state.staging +
      (static_cast<unsigned long long>(slot) *
       worker_consts.frames + seq % worker_consts.frames) * worker_consts.outElems
    };
    roibinFrame(drp_params<ROIBinParams>(ctx), evt.in, dst);
  }

  /**
   * Publish the staged reduced frame. The reduced bin's last event waits on the rest.
   *
   * @note Should run on 1 thread.
   *
   * @param[in] ctx The DRP algorithm kernel context.
   * @param[in] state The global algorithm state view.
   * @param[in] worker_consts The at-launch configured worker constants.
   */
  __global__ void frameBinSync(DrpKernelCtx ctx,
                               FrameBinState state,
                               WorkerBinConsts worker_consts) {
    const auto evt { drp_event(ctx) };
    const unsigned long long seq { evt.seq };
    const unsigned slot { binSlot(seq, worker_consts) };

    cuda::atomic_ref<unsigned, cuda::thread_scope_device> filled {
      state.counters->filled[slot]
    };

    if (evt.is_active) {
      // The inactive events are counted in begin. Only count active here
      filled.fetch_add(1u, cuda::std::memory_order_release);
    }

    if (seq % worker_consts.frames == worker_consts.frames - 1) {
      while (filled.load(cuda::std::memory_order_acquire) != worker_consts.frames) {
        __nanosleep(128);
      }
    }
  }

  /**
   * Sum across staged frame's when the last event for the bin is reached.
   *
   * @note Should run across the full grid.
   *
   * @param[in] ctx The DRP algorithm kernel context.
   * @param[in] state The global algorithm state view.
   * @param[in] worker_consts The at-launch configured worker constants.
   */
  __global__ void frameBinEmit(DrpKernelCtx ctx,
                               FrameBinState state,
                               WorkerBinConsts worker_consts,
                               unsigned long long outBytes,
                               unsigned* doneBlocks) {
    const auto evt { drp_event(ctx) };

    const unsigned long long seq { evt.seq };
    const unsigned slot { binSlot(seq, worker_consts) };

    const bool emit { seq % worker_consts.frames == worker_consts.frames - 1 };
    if (!emit) {
      if (!evt.is_active) {
        return;
      }

      if (blockIdx.x == 0 && threadIdx.x == 0) {
        drp_commit(ctx, { 0ULL, 0 });
      }

      return;
    } else if (!evt.is_active) {
      // Last event of the bin was supposed to emit it. But since we're inactive
      // the event was vetoed. Just reset our counters
      __threadfence();
      __syncthreads();
      if (threadIdx.x != 0) {
        return;
      }

      if (atomicAdd(&doneBlocks[worker_consts.worker], 1u) != gridDim.x - 1) {
        return;
      }

      const unsigned lost {
        static_cast<unsigned>(__popcll(state.counters->mask[slot][0]) +
                              __popcll(state.counters->mask[slot][1]))
      };
      if (lost > 0) {
        atomicAdd(&state.counters->binsDropped, 1ULL);
        atomicAdd(&state.counters->framesDropped, static_cast<unsigned long long>(lost));
      }

      state.counters->filled[slot] = 0;
      state.counters->mask[slot][0] = 0;
      state.counters->mask[slot][1] = 0;

      doneBlocks[worker_consts.worker] = 0;
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> owner {
        state.counters->owner[slot]
      };
      owner.store(seq / worker_consts.frames + kBinSlots, cuda::std::memory_order_release);

      // Can setup a metric?
      // if (blockIdx.x == 0 && threadIdx.x == 0) {
      //   // ...
      // }

      return;
    }

    const unsigned long long presence[2] {
      state.counters->mask[slot][0],
      state.counters->mask[slot][1]
    };

    const float* src {
      state.staging + static_cast<unsigned long long>(slot) * worker_consts.frames * worker_consts.outElems
    };
    DrpAlgos::algorithms::sum_present_frames(src,
                                             outData(evt),
                                             worker_consts.outElems,
                                             worker_consts.frames,
                                             2,
                                             presence);

    __threadfence();
    __syncthreads();
    if (threadIdx.x != 0) {
      return;
    }

    if (atomicAdd(&doneBlocks[worker_consts.worker], 1u) != gridDim.x - 1) {
      return;
    }
    doneBlocks[worker_consts.worker] = 0;

    const unsigned* idx { state.index + slot * worker_consts.frames };
    unsigned wraps { 0 };
    unsigned prev { ~0u };
    for (unsigned k = 0; k < worker_consts.frames; ++k) {
      unsigned long long p_idx { k / 64 }; // 8 byte bit-masks
      if (!((presence[p_idx] >> (k % 64)) & 1ULL)) {
        continue; // Skip graps
      }

      if (prev != ~0u && idx[k] <= prev) {
        ++wraps;
      }

      prev = idx[k];
    }

    ROIBinHeader* hdr { evt.out_as<ROIBinHeader>() };
    hdr->bin = seq / worker_consts.frames;
    hdr->frames = __popcll(state.counters->mask[slot][0]) + __popcll(state.counters->mask[slot][1]);
    if (hdr->frames < worker_consts.frames) {
      atomicAdd(&state.counters->binsPartial, 1ULL);
    }
    hdr->flags = (wraps > 1) ? kFlagOrderSuspect : 0u;
    hdr->presence[0] = state.counters->mask[slot][0];
    hdr->presence[1] = state.counters->mask[slot][1];

    state.counters->filled[slot] = 0;
    state.counters->mask[slot][0] = 0;
    state.counters->mask[slot][1] = 0;
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> owner {
      state.counters->owner[slot]
    };
    owner.store(seq / worker_consts.frames + kBinSlots, cuda::std::memory_order_release);

    drp_commit(ctx, { outBytes, 0 });
  }

  /**
   * Host-side tracker of shared state.
   *
   * The FrameBinGroup manages state as the DRP supports running multiple instances
   * of a Reducer.
   *
   * The groups are organized by key, which is currently set as a JSON string of the
   * CONFIGURE parameters. In principle, you can have 1 group per unique key.
   */
  struct FrameBinStorage {
    DeviceUnique<FrameBinCounters> counters; ///< Owned counters tracking buffers reduced across instances
    DeviceUnique<unsigned[]> index;          ///< Index of the DRP slot where each staged frame is
    DeviceUnique<float[]> staging;           ///< Buffer for staging the reduced time binned data.
    DeviceUnique<unsigned[]> doneBlocks;
    FrameBinState state {};                  ///< Non-owning state view

    FrameBinStorage(unsigned frames, unsigned outElems)
      : counters(deviceAlloc<FrameBinCounters>(1, "[ROIBin 0.2.0] cudaMalloc(counters)"))
      , index(deviceAlloc<unsigned[]>(static_cast<std::size_t>(kBinSlots) * frames,
                                      "[ROIBin 0.2.0] cudaMalloc(index)"))
      , staging(deviceAlloc<float[]>(static_cast<std::size_t>(kBinSlots) * frames * outElems,
                                     "[ROIBin 0.2.0] cudaMalloc(staging)"))
      , doneBlocks(deviceAlloc<unsigned[]>(static_cast<std::size_t>(kMaxWorkers),
                                           "[ROIBin 0.2.0] cudaMalloc(doneBlocks)"))
    {
      FrameBinCounters init {};
      for (unsigned i = 0; i < kBinSlots; ++i) {
        init.owner[i] = i;
      }

      check(cudaMemcpy(counters.get(), &init, sizeof(init), cudaMemcpyHostToDevice),
            "[ROIBin 0.2.0] memcpy(counters)");
      check(cudaMemset(doneBlocks.get(), 0, sizeof(unsigned) * kMaxWorkers),
            "[ROIBin 0.2.0] memset(doneBlocks)");

      state = { counters.get(), index.get(), staging.get() };
    }
  };

  class ROIBinV0_2_0 final
    : public DrpAlgos::TypedKernelDescriptor<ROIBinParams>
    , public DrpAlgos::IMetrics_v1 {
  public:
    void* query(std::uint32_t iface_id) override {
      if (iface_id == IMetrics_v1::id && m_storage) {
        return static_cast<IMetrics_v1*>(this);
      }

      return nullptr;
    }

    std::size_t metricCount() const override { return m_storage ? m_metrics.size() : 0; }
    const Metric* metrics() const override { return m_metrics.data(); }

    std::size_t payloadSize() const override {
      return sizeof(ROIBinHeader) + std::size_t { outElems() } * sizeof(float);
    }

    bool variableSize() const override { return m_params.frames > 1; }

    void setWorkers(unsigned nWorkers) override {
      if (nWorkers == 0 || nWorkers > kMaxWorkers) {
        throw std::invalid_argument("[ROIBin 0.2.0] Unsupported number of Reducer workers.");
      }

      m_nWorkers = nWorkers;
    }

    void launchTyped(const DrpCtx& ctx, cudaStream_t stream) override {
      if (m_nextWorker >= m_nWorkers) {
        throw std::runtime_error("[ROIBin 0.2.0] More launches than Reducer workers.");
      }
      const unsigned worker { m_nextWorker++ };

      constexpr unsigned threads { 256 };
      // Need at least 1 block - empty ROI is allowed to write a header and advance time bins
      const unsigned blocks { std::max(1u, (outElems() + threads - 1) / threads) };
      const DrpCtx pctx { withParams(ctx) };

      if (m_params.frames == 1) {
        roibinDirect<<<blocks, threads, 0, stream>>>(pctx);
        check(cudaGetLastError(), "[ROIBin 0.2.0] launch(direct)");
        return;
      }

      const WorkerBinConsts worker_consts {
        worker,
        m_nWorkers,
        m_params.frames,
        outElems()
      };
      const FrameBinState& state { m_storage->state };

      frameBinBegin<<<1, 1, 0, stream>>>(pctx, state, worker_consts);
      frameBinStage<<<blocks, threads, 0, stream>>>(pctx, state, worker_consts);
      frameBinSync<<<1, 1, 0, stream>>>(pctx, state, worker_consts);
      frameBinEmit<<<blocks, threads, 0, stream>>>(pctx,
                                                   state,
                                                   worker_consts,
                                                   payloadSize(),
                                                   m_storage->doneBlocks.get());

      check(cudaGetLastError(), "[ROIBin 0.2.0] launch(frame bin)");
    }

  protected:
    void onConfigure() override {
      // Check elements for all ROIs
      for (unsigned n = 0; n < m_params.num_rois; ++n) {
        unsigned long long num_elem_in { 1 };
        unsigned long long num_elem_out { 1 };
        for (int dim = 0; dim < 3; ++dim) {
          // Validate all dimensions
          if (m_params.bin[n][dim] == 0) {
            // Allow a zero roi_shape for an empty ROI
            throw std::invalid_argument("[ROIBin 0.2.0] bin must be non-zero");
          }

          unsigned start { m_params.roi_start[n][dim] };
          unsigned extent { m_params.roi_shape[n][dim] };
          if (static_cast<std::uint64_t>(start) + extent > m_params.in_shape[dim]) {
            throw std::invalid_argument("[ROIBin 0.2.0] ROI exceeds in_shape");
          }

          if (extent % m_params.bin[n][dim] != 0) {
            throw std::invalid_argument("[ROIBin 0.2.0] roi_shape must be divisible by bin "
                                        "in every dimension");
          }

          num_elem_in *= m_params.in_shape[dim];
          num_elem_out *= extent / m_params.bin[n][dim];
        }

        // Currently use 32-bit indexing - 64-bit could be used but maybe at perf cost.
        // In the time-bining dimension, though, indices are converted to 64-bit.
        // In any case, it's unlikely to be an issue on single-panel scale.
        if (num_elem_in >= 4294967296ULL) {
          throw std::invalid_argument("[ROIBin 0.2.0] Input will be too large for 32-bit indexing");
        }
        if (num_elem_out >= 4294967296ULL) {
          throw std::invalid_argument("[ROIBin 0.2.0] Output will be too large for 32-bit indexing");
        }
      }

      if (m_params.frames == 0 || m_params.frames > 128) {
        throw std::invalid_argument("[ROIBin 0.2.0] Frames must be >= 1 annd < 128 (for time binning)");
      }

      // Require that in_shape rows and columns can split evenly across the ASIC grid
      // and an ASIC index is used exactly once, for panel tiling
      if (m_params.in_shape[1] % m_params.asic_grid[0] != 0 ||
          m_params.in_shape[2] % m_params.asic_grid[1] != 0) {
        throw std::invalid_argument("[ROIBin 0.2.0] in_shape rows/cols must divide evenly "
                                    "into asic_grid");
      }

      const unsigned n_asics { m_params.asic_grid[0] * m_params.asic_grid[1] };
      bool seen[kMaxAsics] {};
      for (unsigned i = 0; i < n_asics; ++i) {
        const unsigned asic { m_params.asic_layout[i].asic };
        if (asic >= n_asics || seen[asic]) {
          throw std::invalid_argument("[ROIBin 0.2.0] asic_layout must use each ASIC "
                                      "index in [0, n_asics) exactly once");
        }
        seen[asic] = true;
      }

      m_nextWorker = 0;
      if (m_params.frames > 1) {
        m_storage = std::make_unique<FrameBinStorage>(m_params.frames, outElems());
      } else {
        m_storage.reset();
      }

      if (m_storage) {
        const auto base { reinterpret_cast<std::uintptr_t>(m_storage->counters.get()) };
        auto dev = [base](std::size_t off) {
          return reinterpret_cast<const unsigned long long*>(base + off);
        };

        m_metrics = {{
            { "ROIBin_binsDropped", 0, dev(offsetof(FrameBinCounters, binsDropped)) },
            { "ROIBin_framesDropped", 0, dev(offsetof(FrameBinCounters, framesDropped)) },
            { "ROIBin_binsPartial", 0, dev(offsetof(FrameBinCounters, binsPartial)) },
        }};
      }
    }

  private:
    unsigned outElems() const {
      unsigned totalOut { 0 };
      for (unsigned n = 0; n < m_params.num_rois; ++n) {
        totalOut +=
          (m_params.roi_shape[n][0] / m_params.bin[n][0]) *
          (m_params.roi_shape[n][1] / m_params.bin[n][1]) *
          (m_params.roi_shape[n][2] / m_params.bin[n][2]);
      }

      return totalOut;
    }

    std::array<Metric, 3> m_metrics {};

    std::unique_ptr<FrameBinStorage> m_storage; ///< Time-binning state (shared by all Reducer workers)
    unsigned m_nWorkers { 1 };                  ///< The number of Reducer workers (as set by the DRP)
    unsigned m_nextWorker { 0 };                ///< The next worker recorded by `launch`
  };

  REGISTER_KERNEL("ROIBin", 0, 2, 0, ROIBinV0_2_0)
} // anonymous namespace
