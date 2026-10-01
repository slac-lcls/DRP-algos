DrpAlgosCore_lib = shared_library(
    'DrpAlgosCore',
    [
        'drp/KernelRegistry.cc',
    ],
    include_directories: [drp_algos_include],
    install: true,
    install_rpath: lib_install_rpath,
    link_with: [],
    dependencies: deps_for_drp_algos,
    gnu_symbol_visibility: 'inlineshidden',
)

DrpAlgosCore_dep = declare_dependency(
    link_with: [
        DrpAlgosCore_lib,
    ],
    dependencies: deps_for_drp_algos,
    include_directories: drp_algos_include,
)

# Utility script for JIT header embeds. Defined in top-level meson.build
# The JIT library requires KernelCtx.hh
kernel_ctx_cc = custom_target(
    'KernelCtxHdr',
    input: 'drp/KernelCtx.hh',
    output: 'KernelCtxHeader.cc',
    command: [
        py3, jit_header_embed_script, '@INPUT@', '@OUTPUT@', 'kDrpKernelCtxHeaderSource'
    ],
)

DrpAlgosJit_lib = shared_library(
    'DrpAlgosJit',
    [
        'drp/JITKernelDescriptor.cc',
    ] + kernel_ctx_cc,
    include_directories: [drp_algos_include],
    install: true,
    install_rpath: lib_install_rpath,
    cuda_args: drp_algos_cuda_args,
    dependencies: deps_for_drp_algos + DrpAlgosCore_dep,
    gnu_symbol_visibility: 'inlineshidden',
)

DrpAlgosJit_dep = declare_dependency(
    link_with: [
        DrpAlgosJit_lib,
    ],
    dependencies: deps_for_drp_algos + DrpAlgosCore_dep,
    include_directories: drp_algos_include,
)

# The Binning JIT v0.2.0 library requires algorithms/binning.cuh
binning_v020_cc = custom_target(
    'BinningV020Hdr',
    input: 'algorithms/binning.cuh',
    output: 'BinningV020Header.cc',
    command: [
        py3, jit_header_embed_script, '@INPUT@', '@OUTPUT@', 'kBinningKernelHeaderSource'
    ],
)

DrpAlgos_lib = shared_library('DrpAlgos',
  [
      'drp/BinningV0.cu',
      'drp/BinningV020.cc',
  ] + binning_v020_cc,
  include_directories: [drp_algos_include],
  install: true,
  install_rpath: lib_install_rpath,
  cuda_args: drp_algos_cuda_args,
  dependencies: deps_for_drp_algos + DrpAlgosCore_dep + DrpAlgosJit_dep,
  gnu_symbol_visibility: 'inlineshidden',
)

# Setup dependency for the consumers using core libs only
# DrpAlgos is dlopend.
DrpAlgos_dep = declare_dependency(
    link_with: [
        DrpAlgosCore_lib,
        DrpAlgosJit_lib,
    ],
    dependencies: deps_for_drp_algos,
    include_directories: drp_algos_include,
)
meson.override_dependency('DrpAlgos', DrpAlgos_dep)

DrpAlgosHost_lib = shared_library(
    'DrpAlgosHost',
    [
        'host/AlgorithmHost.cu',
    ],
    include_directories: [drp_algos_include],
    install: true,
    install_rpath: lib_install_rpath,
    cuda_args: drp_algos_cuda_args,
    dependencies: deps_for_drp_algos + DrpAlgosCore_dep,
    gnu_symbol_visibility: 'inlineshidden',
)

DrpAlgosHost_dep = declare_dependency(
    link_with: [DrpAlgosHost_lib],
    dependencies: deps_for_drp_algos + DrpAlgosCore_dep,
    include_directories: drp_algos_include,
)
meson.override_dependency('DrpAlgosHost', DrpAlgosHost_dep)

subdir('algorithms')
subdir('drp')
subdir('host')

# Install the headers for usage

install_headers(
    'export_macro.hh',
    subdir: 'DrpAlgos',
)
