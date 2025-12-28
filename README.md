# GPU-Accelerated Convolution Forward Pass (CUDA)

## Overview
This project implements and aggressively optimizes the **forward pass of a convolutional neural network (CNN) layer** on NVIDIA GPUs using **CUDA**. Starting from a baseline CPU and GPU implementation, I applied a series of **profiling-driven GPU optimizations** to reduce end-to-end inference time for large-batch workloads.

The optimized convolution kernels are used to perform inference for convolutional layers in a **modified LeNet-5 architecture**, evaluated on the **Fashion-MNIST** dataset.

This project demonstrates practical GPU performance engineering: memory hierarchy optimization, kernel fusion, Tensor Core usage, and pipeline-level concurrency.

> Originally developed as part of **ECE 408 / CS 483 (Applied Parallel Programming)** at UIUC.  
> All optimizations, profiling, and analysis shown here are my own work.  
> Course scaffolding and grading infrastructure have been removed.

---

## Problem Setting
- **Model:** Modified LeNet-5
- **Layers optimized:** Convolution layers (C1 and C3)
- **Input:** Batch sizes up to **10,000** grayscale images  
  (86 × 86 resolution)
- **Output:** 10-class classification (Fashion-MNIST)
- **Framework:** Mini-DNN-CPP with custom CUDA layers

---

## Baseline Implementation
- Naive CPU convolution
- Basic GPU convolution
- Unrolled convolution (im2col + GEMM)
- Fused GPU kernel (unrolling + matmul + permutation)

**Baseline (fused kernel, batch = 10,000):**
- Total execution time: **~79 ms**
- Accuracy preserved across all batch sizes

---

## Key Optimizations

### 1. CUDA Streams (Overlapping Compute & Data Transfer)
**Technique**
- Pinned host memory
- `cudaMemcpyAsync`
- Multiple CUDA streams to pipeline H2D, kernel execution, and D2H

**Why it works**
- Eliminates idle GPU time by overlapping PCIe transfers with computation

**Result**
- End-to-end layer execution reduced from **~354 ms → ~99 ms**
- Nsight Systems confirms overlapping memcpy and kernel execution
- Kernel logic unchanged; accuracy preserved

**Insight**
> This optimization improves *pipeline efficiency*, not kernel speed — it synergizes with all compute optimizations.

---

### 2. Tensor Cores (WMMA / TF32)
**Technique**
- Warp-level matrix multiplication using NVIDIA WMMA APIs
- 16×16×8 Tensor Core tiles
- FP32 accumulation for numerical stability

**Why it works**
- Each warp executes matrix multiply using specialized Tensor Core hardware
- Shifts kernel from memory-bound to compute-bound

**Result**
- Total execution time reduced to **~63 ms**
- Nsight Compute shows:
  - Tensor Core utilization enabled
  - Math pipeline saturation
  - Reduced memory throttling

**Insight**
> The bottleneck shifts from memory access to math throughput — the intended outcome.

---

### 3. Joint Register + Shared Memory Tiling
**Technique**
- Thread coarsening: each thread computes multiple output elements
- Partial sums stored in registers
- Reduced shared memory traffic during accumulation

**Why it works**
- Registers have lower latency than shared memory
- Increases arithmetic intensity per thread

**Result**
- Execution time reduced to **~69 ms**
- Shared memory instructions reduced **~3×**
- L1 cache hit rate improved **92.9% → 96.0%**
- Slight increase in global loads, hidden by reuse

**Insight**
> Trading registers for shared memory reduces synchronization overhead and improves cache behavior.

---

### 4. cuBLAS GEMM (Comparison Baseline)
**Technique**
- Replaced custom matrix multiply with `cublasSgemm`

**Outcome**
- Large one-time initialization overhead dominates runtime
- Poor fit for this workload shape

**Conclusion**
> cuBLAS excels for large persistent GEMMs, but is suboptimal here due to launch and setup costs.

---

### Minor Optimizations (Measured, Not Assumed)

| Optimization | Outcome | Reason |
|-------------|--------|--------|
| `__restrict__` pointers | Minimal gain | Compiler already effective |
| Loop unrolling | Small gain | Kernel remains memory-bound |
| Constant memory (weights) | No gain | Non-uniform access breaks broadcast |

**Key Insight**
> Profiling confirmed that not all “textbook” optimizations help — each was evaluated and justified.

---

## Final Performance
**Batch size = 10,000**

| Implementation | Total Time |
|---------------|-----------|
| Fused baseline | ~79 ms |
| Tensor Cores | ~63 ms |
| Register tiling | ~69 ms |
| Streamed pipeline | ~99 ms (end-to-end) |
| **Final optimized** | **≈ 60–70 ms** |

- **Speedup:** > **1.3×** over fused baseline, > **5×** over naive GPU
- **Accuracy:** Preserved across all batch sizes

---

## Profiling & Validation
- **Nsight Systems:** pipeline overlap, kernel scheduling, memcpy concurrency
- **Nsight Compute:** Tensor Core usage, warp stall analysis, memory throughput
- **compute-sanitizer:** memory correctness verification

All performance conclusions are backed by profiling data.

---

## Technologies Used
- **CUDA C++**
- **NVIDIA Tensor Cores (WMMA / TF32)**
- **Nsight Systems & Nsight Compute**
- **Linux / Slurm**
- **CMake / Make**
- **Mini-DNN-CPP**

---

## Repository Structure
src/
kernel_fusion_forward.cu # Fused convolution kernel
unroll_forward.cu # Input unrolling (im2col)
m3_forward.cu # Final optimized kernel
m3/
req_0/ # Streams optimization
req_1/ # Tensor Core implementation
op_*/ # Additional optimizations
reports/
profiling/ # Nsight screenshots & analysis

---

## What This Project Demonstrates
- GPU performance engineering using real profiling data
- Memory hierarchy optimization (global / shared / registers)
- Warp-level programming with Tensor Cores
- Kernel-level vs pipeline-level optimization tradeoffs
- Engineering judgment: knowing when *not* to apply an optimization

---

## Academic Context
This project originated from a university course assignment.  
This repository has been **cleaned and refactored** to showcase only my implementation work, analysis, and optimizations. No private tests or grading infrastructure are included.

