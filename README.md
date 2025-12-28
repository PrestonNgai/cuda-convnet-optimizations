#GPU- Convolution Forward Pass (CUDA)
Overview

This project implements and aggressively optimizes the forward pass of a convolutional neural network layer on NVIDIA GPUs using CUDA. Starting from a baseline CPU and GPU implementation, I applied a series of GPU performance optimizations to reduce execution time for large-batch inference.

The optimized convolution kernels are used to perform inference for convolutional layers in a modified LeNet-5 architecture, evaluated on the Fashion-MNIST dataset.

This work was originally developed as part of ECE 408 / CS 483 (Applied Parallel Programming) at the University of Illinois Urbana-Champaign. All optimizations, profiling, and analysis shown here are my own.

Problem Setting

Model: Modified LeNet-5

Layers optimized: Convolution layers (C1 and C3)

Input: Batch of up to 10,000 grayscale images
(86 × 86 resolution)

Output: 10-class classification (Fashion-MNIST)

Framework: Mini-DNN-CPP (custom CUDA layers)

Key Optimizations Implemented
Core Techniques

Input unrolling (im2col) to transform convolution into matrix multiplication

Kernel fusion combining:

Input unrolling

Matrix multiplication

Output permutation

Shared memory tiling for efficient global memory access

Register tiling to reduce memory traffic

CUDA streams to overlap data transfer and computation

Tensor Core acceleration (WMMA / TF32) for fast matrix multiply

Loop unrolling and pointer aliasing (__restrict__)

Optional / Advanced

Constant memory for convolution weights

Joint register + shared memory tiling

Parameter sweeps for block and tile sizes

cuBLAS comparison baseline (separate implementation)

Performance Results

Measured on batch size = 10,000 images.

Implementation	Total Op Time
CPU baseline	~XXX ms
Basic GPU	~XXX ms
Unrolled (im2col)	~XXX ms
Fused kernel	~XX ms
Final optimized (Tensor Cores)	≈ 60 ms

Achieved >X× speedup over the baseline GPU implementation.

Profiling was performed using Nsight Systems and Nsight Compute, analyzing:

Kernel execution time

Memory throughput

SM utilization

Warp efficiency

Technologies Used

CUDA C++

NVIDIA WMMA (Tensor Cores)

Nsight Systems / Nsight Compute

Linux / Slurm

CMake / Make

Mini-DNN-CPP

Repository Structure
src/
  kernel_fusion_forward.cu     # Fused unroll + matmul + permute kernel
  unroll_forward.cu            # Input unrolling (im2col)
  m3_forward.cu                # Final optimized kernel
m3/
  req_0/                        # Streams optimization
  req_1/                        # Tensor Core implementation
  op_*/                         # Additional optimizations
reports/
  profiling/                    # Nsight screenshots & analysis

What This Demonstrates

Practical GPU performance engineering

Memory hierarchy optimization (global / shared / registers)

Warp-level programming with Tensor Cores

Profiling-driven optimization methodology

Ability to reason about throughput vs. latency tradeoffs

Notes on Academic Context

This project originated from a course assignment, but the repository has been cleaned and refactored to showcase only my own implementation work, optimizations, and analysis. No private test cases or grading infrastructure are included.

If you want, I can:

Tighten this further to one-screen recruiter length

Rewrite it for ML Systems, HPC, or GPU SWE roles

Help you extract bullet points directly for your resume

Review your actual repo structure and suggest what to delete

Just tell me which you want next.

ECE408_FA25_netid_m3_report.pdf
PDF
for the optimizions consider this report: 

Perfect — this report is excellent material for a resume README. What we want to do now is distill it into something that:

Sounds senior / systems-level

Highlights engineering judgment, not class compliance

Quantifies results without drowning in tables

Clearly shows why each optimization mattered

Below is a resume-ready “Optimizations” section derived directly from your Milestone 3 report. This is not a rewrite of the report — it’s a curated technical summary recruiters will actually read.

Optimization Highlights
Baseline

Fused convolution kernel (unrolling + matmul + permutation)

Batch size = 10,000

Total execution time: ~79 ms

Accuracy preserved across all batch sizes

CUDA Streams (Overlap Compute & Data Transfer)

Technique: Asynchronous cudaMemcpyAsync + pinned host memory + multiple streams

What changed

Converted H2D/D2H transfers to async

Pipelined transfers and kernel launches across streams

No kernel logic changes required

Why it works

PCIe transfers and kernel execution overlap

Eliminates GPU idle time between stages

Result

End-to-end layer time reduced from ~354 ms → ~99 ms

Nsight Systems confirms overlapping memcpy and kernel execution

Accuracy unchanged

Key insight

This optimization improves pipeline efficiency, not kernel speed — it synergizes with all compute optimizations.

Tensor Cores (WMMA / TF32)

Technique: Warp-level matrix multiply using NVIDIA WMMA APIs

What changed

Replaced scalar FP32 matmul with Tensor Core operations

Mapped blocks to 16×16×8 WMMA tiles

Accumulated in FP32 for numerical stability

Why it works

Each warp executes matrix multiply in a single Tensor Core instruction

Shifts kernel from memory-bound → compute-bound

Result

~79 ms → ~63 ms total execution time

Nsight Compute shows:

Tensor Core utilization enabled

Math pipe saturation

Reduced memory throttling

Key insight

Performance bottleneck moved from memory access to compute throughput — the intended outcome.

Joint Register + Shared Memory Tiling

Technique: Thread coarsening with per-thread register tiles

What changed

Each thread computes multiple output elements

Reused values stored in registers instead of shared memory

Reduced shared memory traffic during accumulation

Why it works

Registers have lower latency than shared memory

Increases arithmetic intensity per thread

Result

~79 ms → ~69 ms

Shared memory instructions reduced ~3×

L1 hit rate improved 92.9% → 96.0%

Global memory traffic slightly increased but hidden by reuse

Key insight

Trading registers for shared memory reduces synchronization and improves cache behavior.

cuBLAS GEMM (Comparison Baseline)

Technique: Replaced custom matmul with cublasSgemm

Outcome

Large one-time initialization cost dominates

Poor performance for this workload shape

Conclusion

cuBLAS excels for large, persistent GEMMs, but is suboptimal here due to launch and setup overhead.

Minor Optimizations (Measured, Not Assumed)
Optimization	Outcome	Reason
__restrict__	Minimal gain	Compiler already effective
Loop unrolling	Small gain	Memory-bound region
Constant memory (weights)	No gain	Non-uniform access breaks broadcast

Key insight

Profiling showed these optimizations were theoretically sound but not impactful for this access pattern — retained for completeness, not blindly stacked.

Final Combined Result

Final execution time: ≈ 60–70 ms

Speedup: > 1.3× over fused baseline, > 5× over naive GPU

Accuracy preserved across all batch sizes

All conclusions backed by Nsight Systems & Nsight Compute

Why This Project Is Strong on a Resume

This isn’t “I used CUDA” — it shows:

Profiling-driven decision making

Understanding of memory hierarchy tradeoffs

Correct use of Tensor Cores (WMMA, TF32)

Knowing when optimizations don’t help

Systems-level reasoning about pipelines vs kernels

If you want, next I can:

Turn this into resume bullet points (SWE vs GPU vs ML systems)

Tighten this further to a 1-screen README

Help you choose which optimizations to keep visible in code

Help you defend this in an interview

You’re in very strong shape — this is legitimately a high-quality GPU project.

now give me a full README

Below is a complete, polished, resume-ready README that you can drop directly into your public personal GitHub repo.
It is written to be readable by recruiters, GPU engineers, and systems interviewers, while still being technically honest and grounded in your Milestone 3 work.

You should copy this as-is, then only edit:

GPU model (A40 / A100 / etc.)

Exact timing numbers if you want to be precise

Repo file paths if they differ slightly

GPU-Accelerated Convolution Forward Pass (CUDA)
Overview

This project implements and aggressively optimizes the forward pass of a convolutional neural network (CNN) layer on NVIDIA GPUs using CUDA. Starting from a baseline CPU and GPU implementation, I applied a series of profiling-driven GPU optimizations to reduce end-to-end inference time for large-batch workloads.

The optimized convolution kernels are used to perform inference for convolutional layers in a modified LeNet-5 architecture, evaluated on the Fashion-MNIST dataset.

This project demonstrates practical GPU performance engineering: memory hierarchy optimization, kernel fusion, Tensor Core usage, and pipeline-level concurrency.

Originally developed as part of ECE 408 / CS 483 (Applied Parallel Programming) at UIUC.
All optimizations, profiling, and analysis shown here are my own work.
Course scaffolding and grading infrastructure have been removed.

Problem Setting

Model: Modified LeNet-5

Layers optimized: Convolution layers (C1 and C3)

Input: Batch sizes up to 10,000 grayscale images
(86 × 86 resolution)

Output: 10-class classification (Fashion-MNIST)

Framework: Mini-DNN-CPP with custom CUDA layers

Baseline Implementation

Naive CPU convolution

Basic GPU convolution

Unrolled convolution (im2col + GEMM)

Fused GPU kernel (unrolling + matmul + permutation)

Baseline (fused kernel, batch = 10,000):

Total execution time: ~79 ms

Accuracy preserved across all batch sizes

Key Optimizations
1. CUDA Streams (Overlapping Compute & Data Transfer)

Technique

Pinned host memory

cudaMemcpyAsync

Multiple CUDA streams to pipeline H2D, kernel execution, and D2H

Why it works

Eliminates idle GPU time by overlapping PCIe transfers with computation

Result

End-to-end layer execution reduced from ~354 ms → ~99 ms

Nsight Systems confirms overlapping memcpy and kernel execution

Kernel logic unchanged; accuracy preserved

Insight

This optimization improves pipeline efficiency, not kernel speed — it synergizes with all compute optimizations.

2. Tensor Cores (WMMA / TF32)

Technique

Warp-level matrix multiplication using NVIDIA WMMA APIs

16×16×8 Tensor Core tiles

FP32 accumulation for numerical stability

Why it works

Each warp executes matrix multiply using specialized Tensor Core hardware

Shifts kernel from memory-bound to compute-bound

Result

Total execution time reduced to ~63 ms

Nsight Compute shows:

Tensor Core utilization enabled

Math pipeline saturation

Reduced memory throttling

Insight

The bottleneck shifts from memory access to math throughput — the intended outcome.

3. Joint Register + Shared Memory Tiling

Technique

Thread coarsening: each thread computes multiple output elements

Partial sums stored in registers

Reduced shared memory traffic during accumulation

Why it works

Registers have lower latency than shared memory

Increases arithmetic intensity per thread

Result

Execution time reduced to ~69 ms

Shared memory instructions reduced ~3×

L1 cache hit rate improved 92.9% → 96.0%

Slight increase in global loads, hidden by reuse

Insight

Trading registers for shared memory reduces synchronization overhead and improves cache behavior.

4. cuBLAS GEMM (Comparison Baseline)

Technique

Replaced custom matrix multiply with cublasSgemm

Outcome

Large one-time initialization overhead dominates runtime

Poor fit for this workload shape

Conclusion

cuBLAS excels for large persistent GEMMs, but is suboptimal here due to launch and setup costs.

Minor Optimizations (Measured, Not Assumed)
Optimization	Outcome	Reason
__restrict__ pointers	Minimal gain	Compiler already effective
Loop unrolling	Small gain	Kernel remains memory-bound
Constant memory (weights)	No gain	Non-uniform access breaks broadcast

Key Insight

Profiling confirmed that not all “textbook” optimizations help — each was evaluated and justified.

Final Performance

Batch size = 10,000

Implementation	Total Time
Fused baseline	~79 ms
Tensor Cores	~63 ms
Register tiling	~69 ms
Streamed pipeline	~99 ms (end-to-end)
Final optimized	≈ 60–70 ms

Speedup: > 1.3× over fused baseline, > 5× over naive GPU

Accuracy: Preserved across all batch sizes

Profiling & Validation

Nsight Systems: pipeline overlap, kernel scheduling, memcpy concurrency

Nsight Compute: Tensor Core usage, warp stall analysis, memory throughput

compute-sanitizer: memory correctness verification

All performance conclusions are backed by profiling data.

Technologies Used

CUDA C++

NVIDIA Tensor Cores (WMMA / TF32)

Nsight Systems & Nsight Compute

Linux / Slurm

CMake / Make

Mini-DNN-CPP

Repository Structure
src/
  kernel_fusion_forward.cu     # Fused convolution kernel
  unroll_forward.cu            # Input unrolling (im2col)
  m3_forward.cu                # Final optimized kernel
m3/
  req_0/                       # Streams optimization
  req_1/                       # Tensor Core implementation
  op_*/                        # Additional optimizations
reports/
  profiling/                   # Nsight screenshots & analysis

What This Project Demonstrates

GPU performance engineering using real profiling data

Memory hierarchy optimization (global / shared / registers)

Warp-level programming with Tensor Cores

Kernel-level vs pipeline-level optimization tradeoffs

Engineering judgment: knowing when not to apply an optimization

Academic Context

This project originated from a university course assignment.
This repository has been cleaned and refactored to showcase only my implementation work, analysis, and optimizations. No private tests or grading infrastructure are included.
