# xcgd: cut-cell continuous Galerkin difference method

xcgd uses continuous Galerkin difference method to solve partial differential equations and perform structural topology optimization applications.

## Dependencies
-  [A2D](https://github.com/smdogroup/a2d)
- [SparseUtils](https://github.com/smdogroup/sparse-utils)
- [algoim](https://github.com/jehicken/algoim)
- [ParOpt](https://github.com/smdogroup/paropt)

## Installation

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
|XCGD_BUILD_TESTS|build unit tests or not|```OFF```|```ON```, ```OFF```|
|CMAKE_BUILD_TYPE|build type|N/A|```Release```, ```Debug```|