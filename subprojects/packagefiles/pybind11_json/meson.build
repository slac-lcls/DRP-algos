project(
    'pybind11_json',
    'cpp',
    version: '0.2.15',
    # meson_version: '>=1.8.0',
    # meson_version: '>=1.10.0',
    #default_options: [
    #    'cpp_std=c++23,c++latest',
    #    'cuda_std=c++20',
    #    'optimization=3',
    #    'pkgconfig.relocatable=true',
    #],
)
add_languages('cpp', native: true, required: true)

# Its a single header-only library
pybind11_json_dep = declare_dependency(
    include_directories: include_directories('include'),
)
