py_install_dir = py3.get_install_dir(pure: false)
pydrpalgos_install_dir = py_install_dir / 'drpalgos'

deps_for_py_mod = [DrpAlgosHost_dep, py3_dep, pybind11_dep]

# Relative path from '$PREFIX/lib/pythonMAJ.MIN/site-packages/drpalgos' to '$PREFIX/lib'
py_to_lib_path = '../../../'
abs_lib_rpath = lib_install_rpath / py_to_lib_path

py_to_lib_install_rpath = '$ORIGIN' + ':' + ('$ORIGIN' / py_to_lib_path) + ':' + abs_lib_rpath

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
