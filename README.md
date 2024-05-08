# zkalman

A simple Kalman filter written in Zig, works with an arbitrary number of variables.


### TODO

- Add code docs and clean up naming conventions
- Add pre-allocated memory instead of stack-based
- Optimize (Unroll and SIMD) for matrix multiplication
  - https://en.algorithmica.org/hpc/algorithms/matmul/#generalizations
  - https://svaniksharma.github.io/posts/2023-05-07-optimizing-matrix-multiplication-with-zig/
- Package and distribute

