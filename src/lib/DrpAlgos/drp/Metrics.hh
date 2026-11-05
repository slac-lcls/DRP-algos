#ifndef DRP_ALGOS_DRP_METRICS_HH
#define DRP_ALGOS_DRP_METRICS_HH

#include <cstddef>
#include <cstdint>

namespace DrpAlgos {

  // NOTE: The `kind` and similar fields are intentionally fixed-width and not e.g., enums
  //       -> The `id` provides versioning for the metrics
  //       -> The metrics versioning can be moved independently of the ABI versioning
  //          so is maintained separately

  struct IMetrics_v1 {
  public:
    static constexpr unsigned id { 0x4D455431 }; ///< Identifier for metric - `MET1` as bytes

    struct Metric {
      const char* name;
      std::uint32_t kind;            ///< 0 = Counter, 1 = Gauge
      const unsigned long long* dev; ///< Device-side word for the DAQ
    };

    virtual std::size_t metricCount() const = 0;
    virtual const Metric* metrics() const = 0;

  protected:
    ~IMetrics_v1() = default;
  };
} // namespace DrpAlgos

#endif // DRP_ALGOS_DRP_METRICS_HH
