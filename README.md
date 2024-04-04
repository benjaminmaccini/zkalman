# zkalman

A simple Kalman filter written in Zig, works with an arbitrary number of


### TODO

- Add code docs and clean up naming conventions
- Add pre-allocated memory
- Optimize (Unroll and SIMD) for matrix multiplication
  - https://en.algorithmica.org/hpc/algorithms/matmul/#generalizations
  - https://svaniksharma.github.io/posts/2023-05-07-optimizing-matrix-multiplication-with-zig/
- Add optimized matrix modules for small filters (2/3 variables)
- Package and distribute

