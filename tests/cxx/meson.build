test_registry_unique = executable(
    'test_registry_unique',
    ['test_registry_unique.cc'],
    dependencies: [DrpAlgosCore_dep,],
    include_directories: drp_algos_include,
)
test(
    'Running test for unique registrations in all libraries.',
    test_registry_unique,
    args: [DrpAlgosV0_lib.full_path(), DrpAlgosV1_lib.full_path()],
    verbose: true,
)
