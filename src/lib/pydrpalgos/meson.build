fs = import('fs')

py_install_dir = py3.get_install_dir(pure: false)
pydrpalgos_install_dir = py_install_dir / 'drpalgos'

# Relative path from '$PREFIX/lib/pythonMAJ.MIN/site-packages/drpalgos' to the shared libraries
lib_dir_abs = get_option('prefix') / get_option('libdir')
py_to_lib_rel = fs.relative_to(lib_dir_abs, pydrpalgos_install_dir)

py_to_lib_install_rpath = '$ORIGIN' + ':' + ('$ORIGIN' / py_to_lib_rel) + ':' + lib_install_rpath

deps_for_py_mod = [DrpAlgosHost_dep, py3_dep, pybind11_dep]

py_mod = py3.extension_module(
    '_drpalgos',
    [
        'pydrpalgos.cc',
    ],
    install: true,
    install_dir: pydrpalgos_install_dir,
    include_directories: [drp_algos_include],
    dependencies: deps_for_py_mod,
    install_rpath: py_to_lib_install_rpath,
    build_rpath: py_to_lib_install_rpath,
)
