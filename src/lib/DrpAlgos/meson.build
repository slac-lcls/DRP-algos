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
    input: 'KernelCtx.hh',
    output: 'KernelCtxHeader.cc',
    command: [
        py3, jit_header_embed_script, '@INPUT@', '@OUTPUT@', 'kDrpKernelCtxHeaderSource'
    ],
)
## Include the specific ABI version implementations
kernel_ctx_v0_cc = custom_target(
    'KernelCtxV0Hdr',
    input: 'abi/KernelCtx_v0.hh',
    output: 'KernelCtxV0Header.cc',
    command: [
        py3, jit_header_embed_script, '@INPUT@', '@OUTPUT@', 'kDrpKernelCtxV0HeaderSource'
    ],
)
kernel_ctx_v1_cc = custom_target(
    'KernelCtxV1Hdr',
    input: 'abi/KernelCtx_v1.hh',
    output: 'KernelCtxV1Header.cc',
    command: [
        py3, jit_header_embed_script, '@INPUT@', '@OUTPUT@', 'kDrpKernelCtxV1HeaderSource'
    ],
)

kernel_ctx_abi_impls = [
    kernel_ctx_cc,
    kernel_ctx_v0_cc,
    kernel_ctx_v1_cc,
]

DrpAlgosJit_lib = shared_library(
    'DrpAlgosJit',
    [
        'drp/JITKernelDescriptor.cc',
    ] + kernel_ctx_abi_impls,
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

plugin_symbol_map = meson.current_source_dir() / 'drpalgos_plugin.map'

###############################
# ABI-Pinned Plugin Libraries #
###############################
# Selection of ABI libs to build comes from meson option

if '0' in get_option('ctx_abis')
    abi_v0 = ['-DDRP_KERNEL_CTX_ABI=0']

    DrpAlgosV0_lib = shared_library(
        'DrpAlgos.abiV0',
        [
            'drp/BinningV0.cu',
            'drp/BinningV020.cc',
            'drp/ROIBinV010.cu',
        ] + binning_v020_cc,
        include_directories: [drp_algos_include],
        install: true,
        install_rpath: lib_install_rpath,
        link_args: ['-Wl,--version-script=' + plugin_symbol_map],
        link_depends: plugin_symbol_map,
        cpp_args: abi_v0,
        cuda_args: drp_algos_cuda_args + abi_v0,
        dependencies: deps_for_drp_algos + DrpAlgosCore_dep + DrpAlgosJit_dep,
        gnu_symbol_visibility: 'inlineshidden',
    )
endif # Build ABI V0

if '1' in get_option('ctx_abis')
    abi_v1 = ['-DDRP_KERNEL_CTX_ABI=1']

    DrpAlgosV1_lib = shared_library(
        'DrpAlgos.abiV1',
        [
            'drp/BinningV011.cu',
            'drp/BinningV021.cc',
            'drp/ROIBinV020.cu',
        ] + binning_v020_cc,
        include_directories: [drp_algos_include],
        install: true,
        install_rpath: lib_install_rpath,
        link_args: ['-Wl,--version-script=' + plugin_symbol_map],
        link_depends: plugin_symbol_map,
        cpp_args: abi_v1,
        cuda_args: drp_algos_cuda_args + abi_v1,
        dependencies: deps_for_drp_algos + DrpAlgosCore_dep + DrpAlgosJit_dep,
        gnu_symbol_visibility: 'inlineshidden',
    )
endif # Build ABI V1

########################
# End Plugin Libraries #

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

subdir('abi')
subdir('algorithms')
subdir('drp')
subdir('host')

# Install the headers for usage

install_headers(
    'export_macro.hh',
    'KernelCtx.hh',
    subdir: 'DrpAlgos',
)
