#include "DrpAlgos/drp/KernelRegistry.hh"

#include <cstdio>
#include <exception>

// Load each plugin into the registry - duplicate descriptors will fail.
int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      DrpAlgos::KernelRegistry::instance().load(argv[i]);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  return 0;
}
