#include "DrpAlgos/drp/JITKernelDescriptor.hh"

#include "DrpAlgos/KernelCtx.hh"
#include "DrpAlgos/drp/BuildInfo.hh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace DrpAlgos {
  /**
   * Embedded contents of KernelCtx.hh and ABI version implementations
   */
  extern const char* const kDrpKernelCtxHeaderSource;
  extern const char* const kDrpKernelCtxV0HeaderSource;
  extern const char* const kDrpKernelCtxV1HeaderSource;

  struct JITKernel {
    CUlibrary library { nullptr }; ///< Never unloaded: lives for the process (like dlopen'd libs)
    CUkernel kernel { nullptr };   ///< Context-independent handle (usable on green-context streams)
  };

  const char* jitNvrtcVersion() {
    auto get_ver = [] () {
      int major { 0 };
      int minor { 0 };

      nvrtcVersion(&major, &minor);

      return std::to_string(major) + "." + std::to_string(minor);
    };

    const static std::string version { get_ver() };

    return version.c_str();
  }

  namespace {
    /// --- Error checking --- ///

    /**
     * Check on driver calls.
     */
    void check(CUresult r, const std::string& what) {
      if (r != CUDA_SUCCESS) {
        const char* name { nullptr };
        const char* msg { nullptr };
        cuGetErrorName(r, &name);
        cuGetErrorString(r, &msg);

        throw std::runtime_error(what + ": " + (name ? name : "?") + " (" + (msg ? msg : "") + ")");
      }
    }

    /**
     * Check on runtime calls.
     */
    void check(cudaError_t e, const std::string& what) {
      if (e != cudaSuccess) {
        throw std::runtime_error(what + ": " + cudaGetErrorString(e));
      }
    }

    /**
     * Check on NVRTC calls.
     */
    void check(nvrtcResult r, const std::string& what) {
      if (r != NVRTC_SUCCESS) {
        throw std::runtime_error(what + ": " + nvrtcGetErrorString(r));
      }
    }

    /// --- Generation of the prelude (for parameter baking) --- ///

    constexpr std::size_t kMaxBakedArray { 1024 };

    enum class Kind { Bool, U64, I64, F64 };

    bool isIdentifier(const std::string& s) {
      if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) {
        return false;
      }

      for (char c : s) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
          return false;
        }
      }

      return true;
    }

    const char* cType(Kind k) {
      switch (k) {
      case Kind::Bool: {
        return "bool";
      }
      case Kind::U64: {
        return "unsigned long long";
      }
      case Kind::I64: {
        return "long long";
      }
      default: {
        return "double";
      }
      }
    }

    Kind scalarKind(const nlohmann::json& v, const std::string& path) {
      if (v.is_boolean()) {
        return Kind::Bool;
      }

      if (v.is_number_unsigned()) {
        return Kind::U64;
      }

      if (v.is_number_integer()) {
        return v.get<long long>() >= 0 ? Kind::U64 : Kind::I64;
      }

      if (v.is_number_float()) {
        return Kind::F64;
      }

      throw std::invalid_argument("JIT parameter '" + path + "': unsupported value " + v.dump());
    }

    Kind arrayKind(const nlohmann::json& arr, const std::string& path) {
      bool anyBool { false };
      bool anyNum { false };
      Kind k { Kind::U64 };
      for (const auto& e : arr) {
        if (e.is_array() || e.is_object()) {
          throw std::invalid_argument("JIT parameter '" + path + "': arrays must be flat");
        }
        const Kind ek { scalarKind(e, path) };
        if (ek == Kind::Bool) {
          anyBool = true;
          continue;
        }

        anyNum = true;
        if (ek == Kind::F64 || (ek == Kind::I64 && k != Kind::F64)) {
          k = ek;
        }
      }

      if (anyBool && anyNum) {
        throw std::invalid_argument("JIT parameter '" + path + "': mixes booleans and numbers");
      }

      if (anyBool) {
        return Kind::Bool;
      }

      if (k == Kind::I64) {
        for (const auto& e : arr) {
          if (e.is_number_unsigned() &&
              e.get<unsigned long long>() > static_cast<unsigned long long>(LLONG_MAX)) {
            throw std::invalid_argument("JIT parameter '" + path + "': mixes negatives and values > INT64_MAX");
          }
        }
      }

      return k;
    }

    std::string doubleLiteral(double v, const std::string& path) {
      if (!std::isfinite(v)) {
        throw std::invalid_argument("JIT parameter '" + path + "': non-finite value");
      }

      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.17g", v);
      std::string s { buf };
      if (s.find_first_of(".eE") == std::string::npos) {
        s += ".0";
      }

      return s;
    }

    std::string literal(const nlohmann::json& v, Kind k, const std::string& path) {
      switch (k) {
      case Kind::Bool: {
        return v.get<bool>() ? "true" : "false";
      }
      case Kind::U64: {
        return std::to_string(v.get<unsigned long long>()) + "ULL";
      }
      case Kind::I64: {
        const long long x { v.get<long long>() };
        if (x == LLONG_MIN) {
          return "(-9223372036854775807LL - 1)";
        }

        return std::to_string(x) + "LL";
      }
      default: {
        return doubleLiteral(v.get<double>(), path);
      }
      }
    }

    void emit(std::ostringstream& os,
              const nlohmann::json& obj,
              const std::string& path,
              int depth) {
      const std::string ind(2 * depth, ' ');

      for (auto it = obj.begin(); it != obj.end(); ++it) {
        const std::string& name { it.key() };
        const nlohmann::json& v { it.value() };
        const std::string p { path.empty() ? name : path + "." + name };

        if (!isIdentifier(name)) {
          throw std::invalid_argument("JIT parameter '" + p + "': name is not a valid C++ identifier");
        }

        if (v.is_object()) {
          os << ind << "namespace " << name << " {\n";
          emit(os, v, p, depth + 1);
          os << ind << "} // namespace " << name << "\n";
        } else if (v.is_array()) {
          if (v.empty() || v.size() > kMaxBakedArray) {
            throw std::invalid_argument("JIT parameter '" + p + "': arrays must have 1.."
                                        + std::to_string(kMaxBakedArray) + " elements");
          }

          const Kind k { arrayKind(v, p) };
          os << ind << "constexpr unsigned long long " << name << "_size = " << v.size() << "ULL;\n";
          os << ind << "__host__ __device__ constexpr " << cType(k) << " " << name
             << "(unsigned long long i) {\n";
          os << ind << "  constexpr " << cType(k) << " v_[] = { ";

          for (std::size_t i = 0; i < v.size(); ++i) {
            os << (i ? ", " : "") << literal(v[i], k, p);
          }

          os << " };\n";
          os << ind << "  return v_[i];\n";
          os << ind << "}\n";
        } else if (v.is_string()) {
          for (unsigned char c : v.get_ref<const std::string&>()) {
            if (c < 0x20 || c == 0x7f) {
              throw std::invalid_argument("JIT parameter '" + p + "': control characters not allowed");
            }
          }
          os << ind << "__host__ __device__ constexpr const char* " << name << "() { return "
             << v.dump() << "; }\n";
        } else {
          const Kind k { scalarKind(v, p) };
          os << ind << "constexpr " << cType(k) << " " << name << " = " << literal(v, k, p) << ";\n";
        }
      }
    }

    /// --- Compile and load (+ cache) JIT kernels --- ///

    std::shared_ptr<const JITKernel> compileAndLoad(const std::string& program,
                                                    const std::string& entry,
                                                    const std::vector<JITHeader>& extraHeaders) {
      int dev { 0 };
      int major { 0 };
      int minor { 0 };
      check(cudaGetDevice(&dev), "cudaGetDevice");
      check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev),
            "cudaDeviceGetAttribute");
      check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev),
            "cudaDeviceGetAttribute");

      const std::vector<std::string> opts {
        "-arch=sm_" + std::to_string(major) + std::to_string(minor),
        "-std=c++20",
      };

      std::string key;
      for (const auto& o : opts) {
        key += o + '\n';
      }
      for (const auto& h : extraHeaders) {
        key += std::string(h.name) + '\n' + h.source + '\n';
      }
      key += entry + '\n' + program;

      static std::mutex mtx;
      static std::map<std::string, std::shared_ptr<const JITKernel>> cache;
      std::lock_guard<std::mutex> lock(mtx);

      if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
      }

      nvrtcProgram prog;
      std::vector<const char*> headers {
        kDrpKernelCtxHeaderSource,
        kDrpKernelCtxV0HeaderSource,
        kDrpKernelCtxV1HeaderSource
      };
      std::vector<const char*> includeNames {
        "DrpAlgos/KernelCtx.hh",
        // And the actual ABI implementations
        "DrpAlgos/abi/KernelCtx_v0.hh",
        "DrpAlgos/abi/KernelCtx_v1.hh"
      };
      for (const auto& h : extraHeaders) {
        headers.push_back(h.source);
        includeNames.push_back(h.name);
      }

      check(nvrtcCreateProgram(&prog,
                               program.c_str(),
                               (entry + ".cu").c_str(),
                               static_cast<int>(headers.size()),
                               headers.data(),
                               includeNames.data()),
            "nvrtcCreateProgram");

      struct ProgramGuard {
        nvrtcProgram* p;
        ~ProgramGuard() { nvrtcDestroyProgram(p); }
      } guard { &prog };

      std::vector<const char*> cOpts;
      for (const auto& o : opts) {
        cOpts.push_back(o.c_str());
      }

      // Pass along the ABI version information
      const std::string abiOpt {
        "-DDRP_KERNEL_CTX_ABI=" + std::to_string(DRP_KERNEL_CTX_ABI)
      };
      cOpts.push_back(abiOpt.c_str());

      const nvrtcResult res {
        nvrtcCompileProgram(prog, static_cast<int>(cOpts.size()), cOpts.data())
      };

      std::size_t logSize { 0 };
      nvrtcGetProgramLogSize(prog, &logSize);
      std::string log(logSize, '\0');
      if (logSize > 1) {
        nvrtcGetProgramLog(prog, log.data());
      }

      if (res != NVRTC_SUCCESS) {
        throw std::runtime_error("NVRTC compile of '" + entry + "' failed (" + nvrtcGetErrorString(res)
                                 + "):\n" + log);
      }

      std::size_t cubinSize { 0 };
      check(nvrtcGetCUBINSize(prog, &cubinSize), "nvrtcGetCUBINSize");
      std::vector<char> cubin(cubinSize);
      check(nvrtcGetCUBIN(prog, cubin.data()), "nvrtcGetCUBIN");

      check(cuInit(0), "cuInit");
      auto kernel = std::make_shared<JITKernel>();
      check(cuLibraryLoadData(&kernel->library,
                              cubin.data(),
                              nullptr,
                              nullptr,
                              0,
                              nullptr,
                              nullptr,
                              0),
            "cuLibraryLoadData");
      check(cuLibraryGetKernel(&kernel->kernel, kernel->library, entry.c_str()),
            "cuLibraryGetKernel('" + entry + "'); the entry point must be extern \"C\" __global__");

      cache.emplace(std::move(key), kernel);
      return kernel;
    }
  } // anonymous namespace

  std::string JITKernelDescriptor::makePrelude(const nlohmann::json& params) {
    if (!params.is_object()) {
      throw std::invalid_argument("JIT parameters must be a JSON object");
    }

    std::ostringstream os;
    os << "// ---- Generated by DrpAlgos::JITKernelDescriptor from the algorithm parameters ----\n";
    os << "namespace jit_params {\n";
    emit(os, params, "", 1);
    os << "} // namespace jit_params\n";

    return os.str();
  }

  //void JITKernelDescriptor::configure(const nlohmann::json& params) {
  void JITKernelDescriptor::configure(const char* paramsStr, std::size_t paramsLen) {
    const auto params = nlohmann::json::parse(paramsStr, paramsStr + paramsLen);
    onConfigure(params);

    const JITSource src { source() };
    if (src.code.empty() || src.entry.empty()) {
      throw std::invalid_argument("JIT source and entry point must be non-empty");
    }

    // #line: NVRTC errors point at lines of the algorithm's own source
    const std::string program {
      makePrelude(params) + "#line 1 \"" + src.entry + ".cu\"\n" + src.code
    };
    m_kernel = compileAndLoad(program, src.entry, src.headers);

    int dev { 0 };
    int sms { 0 };
    check(cudaGetDevice(&dev), "cudaGetDevice");
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    // Whole device - a green context just queues extra blocks
    m_defaultGrid = 4u * static_cast<unsigned>(sms);
  }

  void JITKernelDescriptor::launch(const void* ctx, cudaStream_t stream) {
    if (!m_kernel) {
      throw std::logic_error("JITKernelDescriptor::launch() called before configure()");
    }

    // ctx.params = nullptr; // Parameters are compiled in
    void* args[] = { const_cast<void*>(ctx) };
    check(cuLaunchKernel(reinterpret_cast<CUfunction>(m_kernel->kernel), // CUkernel is accepted here
                         gridSize(), 1, 1,
                         blockSize(), 1, 1,
                         0,
                         reinterpret_cast<CUstream>(stream),
                         args,
                         nullptr),
          "cuLaunchKernel");
  }
} // namespace DrpAlgos
