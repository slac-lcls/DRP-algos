# DrpAlgos

The `DrpAlgos` project provides implementations of algorithms to run in the LCLS2 DAQ's Data Reduction Pipeline (or `DRP`). Some algorithms are entirely implemented within this project, and, additionally the project provides the interface code to run the algorithms as integrations within the DRP Reducer framework.

## Project Organization
Where, and to the extent possible, the pure algorithm code is separated out from the portions of the libraries which are used directly by the DAQ. This allows the algorithms themselves to be used outside of the DRP, and with standalone frameworks without DAQ access. An implementation of a standalone framework is also included in the project as an example, and to aid with testing while integrating new algorithms.

The project source code is organized as follows:

```
DRP-algos/
   |---- src/
   |      |
   |      |---- lib/                         # All C++ and CUDA code for compiled libraries
   |      |      |---- DrpAlgos/             # C++ and CUDA code for algorithms and core libraries
   |      |      |        |
   |      |      |        |---- algorithms   # Standalone algorithm implementations
   |      |      |        |---- drp          # Interface code for the DAQ
   |      |      |        |---- host         # Standalone framework for running without the DAQ
   |      |      |
   |      |      |---- pydrpalgos/           # C++ code for Python bindings
   |      |
   |      |---- drpalgos                     # Pure Python code for the bindings
   |
   |---- utilities/                          # Various build utilities
   |
   |---- tests/                              # Tests
```

## Building the Project

The project can be built using standard `meson` `setup/compile/install` commands. When integrated into the DAQ through the `lcls2` repo, it is built as a meson subproject.

Using the `build.sh` script, it may be possible to build on many linux systems if `meson` is available. If running on S3DF/DRP nodes with access to an LCLS2 environment, you can source the standard environment first, and then run `build.sh`. It will create an installation inside an `install` folder at the top of the repo.

The `build.sh` script does not do modifications of your environment. Some executables should work automatically (e.g. `drp-algos-run`) via the RPATH configuration. However, you will likely need to modify `PYTHONPATH` to expose the modules if you use this script to work from Python.

## Algorithm Integration

`DRP-algos` supports running algorithms which are are ahead-of-time (AOT) or just-in-time (JIT) compiled.

### AOT

... coming soon ...

### JIT

... coming soon ...
