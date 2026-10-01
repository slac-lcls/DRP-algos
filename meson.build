project(
    'DrpAlgos',
    'cpp',
    version: '0.1.0',
    meson_version: '>=1.8.0',
    # meson_version: '>=1.10.0',
    default_options: [
        'cpp_std=c++23,c++latest',
        'cuda_std=c++20',
        'optimization=3',
        'pkgconfig.relocatable=true',
    ],
)
add_languages('cpp', native: true, required: true)

# Only build with CUDA support for now
add_languages('cuda', required: true)

cuda = meson.get_compiler('cuda')
cuda_version = cuda.version()

cpp = meson.get_compiler('cpp')

# Check compiler compatibility
if cpp.get_id() == 'gcc'
    gcc_version = cpp.version()
    if not gcc_version.version_compare('>=13')
        if gcc_version.version_compare('>=12')
            warning('GCC >=13 is recommended, however, GCC 12+ should work.')
        else
            error('Builds with GCC require version >=12')
        endif
    endif

    # Too new a compiler for CUDA support also causes problems
    if cuda_version.version_compare('<12.4') and not gcc_version.version_compare('<13')
        error('For CUDA < 12.4 builds, must use GCC < 13')
    elif cuda_version.version_compare('<12.8') and not gcc_version.version_compare('<14')
        error('For CUDA < 12.8 builds, must use GCC < 14')
    elif cuda_version.version_compare('<13.0') and not gcc_version.version_compare('<15')
        error('For CUDA 12.8-12.9 builds, must use GCC < 15')
    elif cuda_version.version_compare('<13.3') and not gcc_version.version_compare('<16')
        error('For CUDA 13.0-13.2 builds, must use GCC < 16')
    endif
elif cpp.get_id() == 'clang'
    clang_version = cpp.version()
    if not clang_version.version_compare('>=16')
        error('Builds with clang require version >=16')
    endif

    # Too new a compiler for CUDA support also causes problems
    if cuda_version.version_compare('<12.4') and not clang_version.version_compare('<17')
        error('For CUDA < 12.4 builds, must use clang < 17')
    elif cuda_version.version_compare('<12.8') and not clang_version.version_compare('<19')
        error('For CUDA < 12.8 builds, must use clang < 19')
    elif cuda_version.version_compare('<13.0') and not clang_version.version_compare('<20')
        error('For CUDA 12.8-12.9 builds, must use clang < 20')
    elif cuda_version.version_compare('<13.1') and not clang_version.version_compare('<21')
        error('For CUDA 13.0 builds, must use clang < 21')
    elif cuda_version.version_compare('<13.3') and not clang_version.version_compare('<22')
        error('For CUDA 13.1-13.2 builds, must use clang < 22')
    endif
endif

drp_algos_cuda_args = [
    '--std=c++20',
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    '-Xcompiler=-fPIC',
    '-Xfatbin','-compress-all',
]
# Generate PTX for forward compatibility on the Hopper build
drp_algos_cuda_args += [
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_90,code=[sm_90,compute_90]',
]

lib_install_rpath = join_paths(get_option('prefix'), get_option('libdir'))
exe_install_rpath = join_paths(get_option('prefix'), get_option('bindir'))

drp_algos_include = include_directories('src/lib')

#### Python Setup ####
######################

pymod = import('python')

py3 = pymod.find_installation(pure: false)
py3_dep = py3.dependency()

min_python_ver = '3.8'
py_ver = py3.language_version()
if py_ver.version_compare(f'<@min_python_ver@')
    error(f'Minimum supported Python version is @min_python_ver@, found @py_ver@')
endif

# For JIT usage, any included headers are pre-compiled into strings
# We use a small helper utillity to do this
jit_header_embed_script = files(meson.project_source_root() + '/utilities/embed_header.py')

#### Conda and Other Env Setup ####
###################################

conda_base = get_option('conda_prefix')

if conda_base != ''
    add_project_link_arguments(
      [
        '-Wl,-O2',
        '-Wl,--sort-common',
        '-Wl,--as-needed',
        '-Wl,--gc-sections',
        '-Wl,-z,relro',
        '-Wl,-z,now',
        '-Wl,--disable-new-dtags',
        '-Wl,--allow-shlib-undefined',
        '-Wl,-rpath,' + join_paths(conda_base, 'lib'),
        '-Wl,-rpath-link,' + join_paths(conda_base, 'lib'),
        '-L' + join_paths(conda_base, 'lib'),
        '-L' + join_paths(conda_base, 'targets/x86_64-linux/lib'),
        '-L' + join_paths(conda_base, 'targets/x86_64-linux/lib/stubs'),
      ],
      language: 'cpp'
    )

    add_project_link_arguments(
      [
        '-Xlinker=-O2',
        '-Xlinker=--as-needed',
        '-Xlinker=--gc-sections',
        '-Xlinker=-z,relro',
        '-Xlinker=-z,now',
        '-Xlinker=-rpath,' + join_paths(conda_base, 'lib'),
        '-L' + join_paths(conda_base, 'lib'),
        '-L' + join_paths(conda_base, 'targets/x86_64-linux/lib'),
      ],
      language: 'cuda'
    )
endif # conda_base != ''

#### Core Dependencies ####
###########################

# CUDA: Requires NVRTC and both runtime and driver API
cuda_dep = dependency('cuda', modules: ['cudart', 'nvrtc'])

# Can be hard to get driver lib
cuda_path = cuda_dep.get_variable(
    internal: 'cuda_path',
    default_value: '/usr/local/cuda',
    configtool: 'cuda_path',
)
cuda_lib_dirs = [
    join_paths(cuda_path, 'targets/x86_64-linux/lib'),
    join_paths(cuda_path, 'lib/'),
    join_paths(cuda_path, 'lib64/'),
]
cuda_driver = cpp.find_library(
    'cuda',
    dirs: [cuda_lib_dirs, join_paths(cuda_path, 'lib64/stubs'),],
    required: true,
)
cuda_driver_dep = declare_dependency(
    include_directories: include_directories([
        join_paths(cuda_path, 'targets/x86_64-linux/include'),
        join_paths(cuda_path, 'targets/x86_64-linux/include/cccl'),
    ]),
    dependencies: [cuda_driver],
)

# Add libdl as needed for JIT if not merged directly into the cstdlib
dl_dep = cpp.find_library('dl', required: true)

nlohmann_json_dep = dependency('nlohmann_json', required: false)
if not nlohmann_json_dep.found()
    message('nlohmann JSON not found - using subproject')
    nlohmann_json_proj = subproject('nlohmann_json')
    nlohmann_json_dep = nlohmann_json_proj.get_variable('nlohmann_json_dep')
endif # not nlohmann_json_dep.found()

deps_for_drp_algos = [cuda_dep, cuda_driver_dep, dl_dep, nlohmann_json_dep]

subdir('src')

# Setup pkg-config for simplicity, AFTER the subdirs are traversed
pkg = import('pkgconfig')
pc_libs = [DrpAlgosCore_lib, DrpAlgosJit_lib]
pc_headers = ['.', 'DrpAlgos']
pkg.generate(
    name: 'DrpAlgos',
    description: 'DRP reduction algorithms, registries, and descriptors.',
    libraries: pc_libs,
    subdirs: pc_headers,
    version: meson.project_version(),
)
