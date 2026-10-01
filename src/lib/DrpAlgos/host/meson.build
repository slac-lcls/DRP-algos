install_headers(
    'AlgorithmHost.hh',
    subdir: 'DrpAlgos/host',
)

drp_algos_run = executable(
    'drp-algos-run',
    'drp_algos_run.cc',
    include_directories: [drp_algos_include],
    dependencies: [DrpAlgosHost_dep],
    install: true,
    install_rpath: lib_install_rpath,
)
