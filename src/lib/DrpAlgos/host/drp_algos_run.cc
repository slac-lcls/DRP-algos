#include "DrpAlgos/drp/Version.hh"
#include "DrpAlgos/host/AlgorithmHost.hh"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  struct ArrF32 {
    std::vector<std::size_t> shape; ///< Array shape
    std::vector<float> data;        ///< Contiguous array elements
  };

  std::vector<std::size_t> parseShape(const std::string& s) {
    std::vector<std::size_t> shape;

    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      shape.push_back(std::stoull(tok));
    }

    return shape;
  }

  std::string need(const std::map<std::string, std::string>& args,
                   const std::string& key) {
    const auto it { args.find(key) };
    if (it == args.end()) {
      throw std::invalid_argument("Missing --" + key);
    }

    return it->second;
  }

  std::string opt(const std::map<std::string, std::string>& args,
                  const std::string& key,
                  const std::string& def) {
    const auto it { args.find(key) };

    return it == args.end() ? def : it->second;
  }

  void usage() {
    const std::string help_msg { R"help(
drp-algos-run --alg <alg_name> --version <version> [--soname libDrpAlgos.so]
              --params <params.json>
              --frame-shape <frame shape>
              [--events N] [--slots K] [--bench N]


Run a DRP algorithm standalone, with the DRP's execution model. Algorithms must be those
that have actually been registered with DRP-algos. Provide the algorithm name,
the algorithm version, and a parameter JSON document matching the types of parameters you
would provide if running in the DAQ. A frame shape must be provided in comma-delimited form
which will be used to construct random input frames. Slots can be configured to emulate the
number of DRP worker buffers (default 4).
)help"};

    std::printf("%s", help_msg.c_str());
    exit(0);
  }

  std::map<std::string, std::string> parseArgs(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
      const std::string key { argv[i] };

      if (key == "--help" || key == "-h") {
        usage();
        exit(0);
      }

      if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
        throw std::invalid_argument("Bad argument: " + key);
      }
      args[key.substr(2)] = argv[++i];
    }

    return args;
  }
} // anonymous namespace

int main(int argc, char** argv) {
  try {
    const auto args { parseArgs(argc, argv) };

    const std::string alg { need(args, "alg") };
    const auto version { DrpAlgos::Version::parse(need(args, "version")) };
    const std::string soname { opt(args, "soname", "libDrpAlgos.so") };

    std::ifstream pf(need(args, "params"));
    if (!pf) {
      throw std::runtime_error("Cannot open params file");
    }
    // Cannot use { } initializer here, wrong nlohmann contructor chosen.
    const nlohmann::json params = nlohmann::json::parse(pf);

    // Input frames
    ArrF32 frames;

    const auto frameShape { parseShape(need(args, "frame-shape")) };
    std::size_t nElems { 1 };
    for (const auto d : frameShape) {
      nElems *= d;
    }

    const std::size_t n { 16 };
    frames.shape = { n };
    frames.shape.insert(frames.shape.end(), frameShape.begin(), frameShape.end());
    frames.data.resize(n * nElems);
    std::mt19937 rng { 12345 };
    std::uniform_real_distribution<float> dist { -100.0f, 1000.0f };
    for (auto& v : frames.data) {
      v = dist(rng);
    }

    if (frames.shape.size() < 2) {
      throw std::runtime_error("Input must be shaped (n_frames, ...)");
    }

    const std::size_t nFrames { frames.shape[0] };
    const std::size_t frameElems { frames.data.size() / nFrames };
    const std::size_t nEvents {
      std::stoull(opt(args, "events", std::to_string(nFrames)))
    };

    DrpAlgos::HostConfig cfg;
    cfg.slots = static_cast<unsigned>(std::stoul(opt(args, "slots", "4")));
    cfg.inElems = frameElems;

    DrpAlgos::AlgorithmHost host(alg, version, soname, params, cfg);
    std::printf("%s %s: payload %zu B%s, frame %zu elems, %zu events\n",
                alg.c_str(),
                version.toString().c_str(),
                host.payloadSize(),
                host.variableSize() ? " (variable)" : "",
                frameElems,
                nEvents);

    // Run events through the ring
    std::vector<DrpAlgos::EventResult> results;
    results.reserve(nEvents);
    double deviceMs { 0.0 };
    for (std::size_t i = 0; i < nEvents; ++i) {
      results.push_back(host.process(frames.data.data() + (i % nFrames) * frameElems));
      deviceMs += results.back().deviceMs;
    }
    if (nEvents) {
      std::printf("mean device time per event: %.2f us\n", 1000.0 * deviceMs / nEvents);
    }

    // Benchmark timing
    const std::string b { opt(args, "bench", "") };
    if (!b.empty()) {
      const auto bench { host.benchmark(std::stoull(b)) };
      std::printf("bench (%zu iters): mean %.2f us, min %.2f us, max %.2f us  -> %.1f kHz\n",
                  bench.iterations,
                  bench.meanUs,
                  bench.minUs,
                  bench.maxUs,
                  bench.meanUs > 0 ? 1000.0 / bench.meanUs : 0.0);
    }

    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "drp-algos-run: %s\n", e.what());
    return 1;
  }
}
