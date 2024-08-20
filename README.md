# XCGD: cut-cell continuous Galerkin difference method

XCGD uses continuous Galerkin difference method to solve partial differential equations and perform structural topology optimization applications.
XCGD is a header-only library.

## Dependencies
-  [A2D](https://github.com/smdogroup/a2d)
- [SparseUtils](https://github.com/smdogroup/sparse-utils)
- [algoim](https://github.com/jehicken/algoim)
- [ParOpt](https://github.com/smdogroup/paropt)

## Installation

As XCGD is a header-only library, installation merely means moving headers and
dependency metadata maintained in .cmake to a destination of the installation:
```
mkdir build_for_install && cd build_for_install && cmake .. -DXCGD_BUILD_TESTS=OFF -DXCGD_BUILD_EXAMPLES=OFF && make install
```
This installs XCGD headers and CMake files into
```${HOME}/installs/xcgd```.

To change this location, modify ```XCGD_INSTALL_DIR```:
```cmake -DXCGD_INSTALL_DIR=<new location> <other args>```.


## Build examples
```
mkdir build && cd build && cmake .. && make -j
```

## Test
```
cd build && ctest . -j <num_procs>
```

## CMake variables
| Variable | Description | Default | Choices |
|----------|-------------|---------|---------|
|XCGD_SPARSE_UTILS_DIR|path to a SparseUtils installation|```${HOME}/installs/sparse-utils```|a path|
|XCGD_A2D_DIR|path to an A2D installation|```${HOME}/installs/a2d```|a path|
|XCGD_ALGOIM_DIR|path to algoim source code|```${HOME}/git/algoim```|a path|
|XCGD_PAROPT_DIR|path to a ParOpt installation|```${HOME}/git/paropt```|a path|
|XCGD_BUILD_TESTS|build unit tests or not|```ON```|```ON```, ```OFF```|
|XCGD_BUILD_EXAMPLES|build examples or not|```ON```|```ON```, ```OFF```|
|CMAKE_BUILD_TYPE|build type|N/A|```Release```, ```Debug```|
|XCGD_INSTALL_DIR|destination of the installation|${HOME}/installs/xcgd|a path|