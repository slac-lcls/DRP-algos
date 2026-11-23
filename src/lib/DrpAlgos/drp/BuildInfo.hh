#ifndef DRP_ALGOS_DRP_BUILD_INFO_HH
#define DRP_ALGOS_DRP_BUILD_INFO_HH

#include "DrpAlgos/export_macro.hh"

namespace DrpAlgos {
  // NOTE: gitBuildVersion compiles into libDrpAlgosCore via BuildInfo.cc
  //       jitNvrtcVersion compiles into libDrpAlgosJit via JITKernelDescriptor.cc

  DRP_ALGOS_API const char* gitBuildVersion();  ///< `git describe` of the DRP-algos build (libDrpAlgosCore)
  DRP_ALGOS_API const char* jitNvrtcVersion();  ///< NVRTC used for JIT kernels, "MAJOR.MINOR" (libDrpAlgosJit)
} // namespace DrpAlgos

#endif // DRP_ALGOS_DRP_BUILD_INFO_HH
