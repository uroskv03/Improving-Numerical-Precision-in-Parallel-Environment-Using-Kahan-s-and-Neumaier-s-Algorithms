# Improving-Numerical-Precision-in-Parallel-Environment-Using-Kahan-s-and-Neumaier-s-Algorithms


## Problem Overview

Floating-point addition under the IEEE-754 standard is **non-associative** because of finite-precision rounding errors that occur at each step. In parallel environments (OpenMP), thread execution scheduling and partial sum groupings are non-deterministic, causing identical runs to produce inconsistent results.

This project addresses this challenge by **implementing Kahan and Neumaier compensated summation algorithms** into multi-threaded reduction workflows to track and correct floating-point rounding errors at each iteration.

## Execution Environment & Target Platform

All performance benchmarks and accuracy evaluations were executed on **rtidev5**, a high-performance remote development server at the School of Electrical Engineering, University of Belgrade.

* **Hardware Specs:** Intel Core i7-11700F CPU @ 2.50 GHz (11th Gen, 8 physical cores / 16 threads via Hyper-Threading, max boost frequency of 4.90 GHz) with 62 GiB RAM, and L1d/L1i/L2/L3 cache capacities of 384 KiB, 256 KiB, 4 MiB, and 16 MiB, respectively.
* **Software Environment:** Ubuntu 22.04.5 LTS (Linux Kernel 5.15.0-186-generic) running GCC 11.4.0 supporting OpenMP 4.5.
* **Compilation Command:** All test applications were compiled using the identical optimized command flags:
  gcc -fopenmp -O3 <fileName>.c -o <programName> -lm -lquadmath


## Test Applications

To measure numerical stability and speedup, three benchmarks of increasing structural complexity were implemented, each tested across three input complexity profiles and averaged over multiple runs:

* **Array Summation (`sumArray`):** Summing large-scale `double` arrays (10,000,000 elements).
* **Matrix Multiplication (`matMul`):** Double-precision matrix multiplication where accuracy is evaluated per element of the resulting matrix.
* **Molecular Dynamics (`md`):** An N-body force and energy simulation evaluated over multiple time steps; non-associative accumulation of interaction forces across iterations causes significant multi-step rounding error propagation.

## Implementation Details

* **Precision Reference:** High-precision ground truth was computed using extended quad-precision floating-point arithmetic (`__float128` via `libquadmath`), which was subsequently rounded to `double` for direct, fair comparisons.
* **Custom OpenMP Reductions:** Kahan and Neumaier algorithms were integrated using custom inline functions alongside OpenMP reduction operators (`#pragma omp declare reduction`) operating over a dedicated `KahanAcc` state structure.


## Measured Metrics

To evaluate the algorithms, the following key performance and accuracy metrics were tracked across executions:

* **Relative Error**
* **Global Relative Error:** - sum of absolute errors divided by the sum of absolute exact values
* **Maximum Relative and Absolute Error**
* **Total Force Sum (for MD simulation):** - theoretically expected to sum to 0.0.
* **Execution Time & Speedup**
