#include "DrpAlgos/drp/KernelRegistry.hh"
#include "DrpAlgos/drp/Version.hh"

#include <dlfcn.h>

#include <sstream>
#include <stdexcept>

namespace DrpAlgos {
  KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry reg;
    return reg;
  }

  void KernelRegistry::registerKernel(const std::string& name,
                                      const Version& version,
                                      KernelFactoryFn factory) {
    Key key { name, version };
    if (m_factories.count(key)) {
      // Can't throw here: we're inside a static initializer during dlopen.
      // Save the error for later
      m_pendingErrors.push_back("duplicate registration of '" + name + "' " + version.toString());
      return;
    }

    m_factories.emplace(std::move(key), std::move(factory));
  }

  std::shared_ptr<IKernelDescriptor> KernelRegistry::create(const std::string& name,
                                                            const Version& version,
                                                            const std::string& soname) {
    if (soname.empty()) {
      throw std::runtime_error("No soname given for '" + name + "' " +
                               version.toString());
    }

    if (auto err = m_libErrors.find(soname); err != m_libErrors.end()) {
      throw std::runtime_error(err->second);
    }

    if (!m_libs.count(soname)) {
      m_pendingErrors.clear();

      void* handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!handle) {
        throw std::runtime_error("dlopen('" + soname + "') failed: " + dlerror());
      }
      m_libs.emplace(soname, handle);

      if (!m_pendingErrors.empty()) {
        std::ostringstream msg;
        msg << "Errors registering kernels from '" << soname << "':";
        for (const auto& e : m_pendingErrors) {
          msg << "\n  - " << e;
        }

        m_libErrors.emplace(soname, msg.str());
        m_pendingErrors.clear();
        throw std::runtime_error(m_libErrors.at(soname));
      }
    }

    auto it = m_factories.find(Key { name, version });
    if (it == m_factories.end()) {
      std::ostringstream msg;
      msg << "No kernel '" << name << "' version " << version.toString()
          << " after loading '" << soname << "'. Registered:";

      for (const auto& [n, v] : registered()) {
        msg << "\n  - " << n << " " << v.toString();
      }

      throw std::runtime_error(msg.str());
    }

    auto desc = it->second();
    if (!desc) {
      throw std::runtime_error("Factory for '" + name + "' " +
                               version.toString() + " returned null");
    }

    return desc;
  }

  std::vector<std::pair<std::string, Version>> KernelRegistry::registered() const {
    std::vector<std::pair<std::string, Version>> out;
    out.reserve(m_factories.size());
    for (const auto& [key, factory] : m_factories) {
      out.push_back(key);
    }

    return out;
  }
} // namespace DrpAlgos
