# AMD HPC Training Examples Repo

Welcome to AMD's HPC Training Examples Repo!
Here you will find a variety of examples to showcase the capabilities of AMD's GPU software stack.
Please be aware that the repo is continuously updated to keep up with the most recent releases of the AMD software.

## Repository Structure

Please refer to these instructions to locate the exercises you are interested in sorted by topic. 

1. [`HIP`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP)
   1. Basic Examples
      1. [`Stream_Overlap`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/Stream_Overlap): this example showcases how to share the workload of a GPU offload compation using several overlapping streams. The result is an additional gain in terms of time of execution due to the additional parallelism provided by the overlapping streams. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIP/Stream_Overlap/README.md).
      2. [`dgemm`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/dgemm): A (d)GEMM application created as an exercise to showcase simple matrix-matrix multiplications on AMD GPUs. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/dgemm/README.md).
      3. [`exercises`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/exercises): a collection of introductory exercises such as device to host data transfer and basic GPU kernel implementation. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/exercises/README.md).
      4. [`hip_stream`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/hip-stream): modification of the STREAM benchmark for HIP. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIP/hip-stream/README.md).
      5. [`jacobi`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/jacobi): distributed Jacobi solver, using GPUs to perform the computation and MPI for halo exchanges. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIP/jacobi/README.md).
      6. [`matrix_addition`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/matrix_addition): example of a HIP kernel performing a matrix addition. 
      7. [`saxpy`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/saxpy): example of a HIP kernel performing a saxpy operation. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/saxpy/README.md)
      8. [`stencil`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/stencil): example of a stencil operation with a HIP kernel.
      9. [`stencil_timers`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/stencil_timers): same as `stencil` but with timers.
      10. [`stencil_overlap`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/stencil_overlap): similar to `stencil` but using asyncronous copies with `hipMemcpyAsync`.
      11. [`stencil_overlap_timers`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/stencil_overlap_timers): same as `stencil_overlap` but with timers.
      12. [`vectorAdd`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/vectorAdd): example of a HIP kernel to perform a vector add. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/vectorAdd/README.md).
      13. [`vector_addition`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/vector_addition): another example of a HIP kernel to perform vector addition, including different versions such as one using shared memory, one with timers, and a CUDA one to try hipify and hipifly tools on. The examples in this directory are not part of the HIP test suite.
   2. CUDA to HIP Porting
      1. [`HIPIFY`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPIFY): example to show how to port CUDA code to HIP with hipify tools. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIPIFY/README.md).
      2. [`hipifly`](https://github.com/amd/HPCTrainingExamples/tree/main/hipifly): example to show how to port CUDA code to HIP with hipifly tools. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/hipifly/vector_add/README.md).
   3. [`HIP-Optimizations`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP-Optimizations): a daxpy HIP kernel is used to show how an initial version can be optimized to improve performance. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP-Optimizations/daxpy/README.md).
   4. [`HIPFort`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPFort): a gemm example in Fortran using hipfort.
   5. [`HIPStdPar`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar): several examples showing C++ Std Parallelism on AMD GPUs. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIPStdPar/CXX/README.md).
2. [`MPI-examples`](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples)
   1. Benchmarks: benchmarks (`collective.cpp` and `pt2pt.cpp`) to assess the performance of the communication libraries. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/MPI-examples/README.md).
   2. [`GhostExchange`](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange): slimmed down example of an actual physics application where the solution is initialized on a square domain discretized with a Cartesian grid, and then advanced in parallel using MPI communications. **NOTE**: detailed [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/README.md) files are provided here for the different versions of the `GhostExchange_ArrayAssign` code, that showcase how to use `Omnitrace` to profile this application.
3. [`ManagedMemory`](https://github.com/amd/HPCTrainingExamples/tree/main/ManagedMemory): programming model exercises: topics covered are for instance APU programming model, OpenMP, performance protability frameworks (Kokkos and Raja) and discrete GPU programming model. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/ManagedMemory/README.md).
4. [`MLExamples`](https://github.com/amd/HPCTrainingExamples/tree/main/MLExamples): a variation of PyTorch's MNIST example code. 
5. [`Occupancy`](https://github.com/amd/HPCTrainingExamples/tree/main/Occupancy): example on modifying thread occupancy, using several variants of a matrix vector multiplication leveraging shared memory and launch bounds.
6. [`OmniperfExamples`](https://github.com/amd/HPCTrainingExamples/tree/main/OmniperfExamples): several examples showing how to leverage Omniperf to perform kernel level optimization. **NOTE**: detailed READMEs are provided on each subdirectory. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/OmniperfExamples/README.md).
7. [`Omnitrace`](https://github.com/amd/HPCTrainingExamples/tree/main/Omnitrace)
   1. Omnitrace on Jacobi: Omnitrace used on the Jacobi solver example. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Omnitrace/omnitrace_on_Jacobi.md). 
   2. Omnitrace by Example: Omnitrace used on several versions of the Ghost Exchange example. [`READMEs`](https://github.com/amd/HPCTrainingExamples/blob/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign) available for each of the different versions of the example code.
8. [`Pragma_Examples`](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples): OpenMP (in Fortran, C, and C++) and OpenACC examples. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples)


## Run the Tests

Most of the exercises in this repo can be run as a test suite by doing:

```
git clone https://github.com/amd/HPCTrainingExamples && \
cd HPCTrainingExamples && \
cd tests && \
./runTests.sh
```
You can also run a subset of the whole test suite by specifying the subset you are interested in as an input to the `runTests.sh` script. For instance: `./runTests.sh --pytorch`. To see a full list of the possible subsets that can be run, do `./runTests.sh --help`.

**NOTE**: tests can also be run manually from their respective directories, provided the necessary modules have been loaded and they have been compiled appropriately.

## Feedback
We welcome your feedback and contributions, feel free to use this repo to bring up any issues or submit pull requests.
The software made available here is released under the MIT license, more details can be found in `LICENSE.md`.
