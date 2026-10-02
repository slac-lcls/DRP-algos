# Running Example Algorithms

To get started with `DRP-algos`, you can run some example algorithms using the `AlgorithmHost` standalone framework.

The framework comes with a companion command-line utility called `drp-algos-run`.

## Initial Setup

You must first clone and build the project. On an S3DF node with GPUs and CUDA toolkit access (or appropriate CUDA environment variables set) you can clone and build as follows:

```bash
git clone git@github.com:slac-lcls/DRP-algos.git
cd DRP-algos
./build.sh
```

This will build the core libraries, and the standalone framework with command-line utility at `DRP-algos/install`.

## Running Example Algorithms

Algorithms in `DRP-algos` consist of the algorithm kernels themselves, in addition to some interface code. The latter requires that the algorithm be "registered" under a specific name and version identifier. Once this is done, the algorithm schema, is immutable. This means that that version of the algorithm will always take parameters of the same types.

Additionally, once the algorithm has been registered, while its code and symbols are hidden (the core libraries are opaque in this regard), they can be looked up by simple name and version identifier. This is how the command-line utility (`drp-algos-run`) works.

Using `drp-algos-run`, we can provide a name of an algorithm and a version, along with a small JSON file that contains values for the parameters that match what the algorithm of that version expects. There are two basic example cases:
- `Binning v0.1.0`
- `Binning v0.2.0`

These two algorithms actually take the same parameters. The difference is that `v0.2.0` uses the JIT compilation pathway, while `v0.1.0` is using the AOT compilation pathway.

Both the algorithm descriptors are accessed via the `libDrpAlgos.so` library. This is the default for the command-line utility.

If you have built the project using `build.sh` you should now be at the root of `DRP-algos`. You can then run the examples using:
```bash
# Inside the root of the DRP-algos repo
# An install sub-directory should've been created if you used build.sh
./install/bin/drp-algos-run --alg Binning --version 0.2.0 --params schemas/binning_v020.defaults.json --frame-shape 6,168,192 --events 1000 --bench 1000
```

This command will run version `v0.2.0` (the JIT compiled one) using the parameters in the example document. It will process 1000 events, and do further benchmarking on another 1000. The event processing path uses a setup that closely mimimcs the real DRP (the secondary benchmark pathway is more for timing the algorithm in isolation). `--frame-shape` allows providing a comma-delimited input array shape.

## Using `drp-algos-run` for Development

After running the basic examples you can use these tools to work on new algorithms. Have a look at `src/lib/DrpAlgos/drp/BinningV0.cu` (AOT) and `src/lib/DrpAlgos/drp/BinningV020.cc` (JIT) source files for starting points.

When working locally, you can simply rebuild the project with your new source code files and algorithms integrated into the libraries. If you have used the `REGISTER_ALGORITHM` macro found in these two example source code files, you will then be able to use `drp-algos-run` to test their performance.
