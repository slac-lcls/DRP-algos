# Only include the actually built libraries for the test binary
registry_libs = []
if '0' in get_option('ctx_abis')
    registry_libs += DrpAlgosV0_lib.full_path()
endif
if '1' in get_option('ctx_abis')
    registry_libs += DrpAlgosV1_lib.full_path()
endif

test_registry_unique = executable(
    'test_registry_unique',
    ['test_registry_unique.cc'],
    dependencies: [DrpAlgosCore_dep,],
    include_directories: drp_algos_include,
)
test(
    'Running test for unique registrations in all libraries.',
    test_registry_unique,
    args: registry_libs,
    verbose: true,
)
