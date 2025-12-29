#include <cmath>
#include <iostream>
#include <mma.h>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

using namespace nvcuda;

#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    int Height_out = Height - K + 1;
    int Width_out  = Width - K + 1;

    // Number of input features per output pixel 
    // (width of the filter matrix | height of the input feature map)
    int K_size = Channel * K * K;
    // Number of total output pixels across batch
    int N_size = Batch * Height_out * Width_out;

    int tid = threadIdx.x;

    // Calculate which 16x16 tile this block is responsible for
    int warp_row = blockIdx.y * WMMA_M; 
    int warp_col = blockIdx.x * WMMA_N;  

    if (warp_row >= Map_out || warp_col >= N_size) {
        return;
    }

    int HoutWout = Height_out * Width_out;
    int HxW      = Height * Width;
    int KK       = K * K;
    
    // Declare WMMA fragments for TF32 (16x16x8)
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ float shared_mask[WMMA_M][WMMA_K];    
    __shared__ float shared_input[WMMA_K][WMMA_N];   
    __shared__ float shared_output[WMMA_M][WMMA_N];  
    __shared__ int n_batch[WMMA_N];
    __shared__ int n_h[WMMA_N];
    __shared__ int n_w[WMMA_N];

    if (tid < WMMA_N) {
        int global_n = warp_col + tid;
        if (global_n < N_size) {
            int currBatch = global_n / HoutWout;
            int remB      = global_n - currBatch * HoutWout;   
            int h_out     = remB / Width_out;
            int w_out     = remB - h_out * Width_out;          
            n_batch[tid]  = currBatch;
            n_h[tid]      = h_out;
            n_w[tid]      = w_out;
        } else {
            n_batch[tid] = -1;
            n_h[tid] = n_w[tid] = 0;
        }
    }
    __syncthreads();

    for (int k_base = 0; k_base < K_size; k_base += WMMA_K) {

        //  load mask tile into shared memory | 16x8 = 128 elements
        if (tid < 128) {
            int tile_row   = tid / WMMA_K;  
            int tile_col   = tid % WMMA_K;  
            int global_row = warp_row + tile_row;
            int global_col = k_base + tile_col;

            if (global_row < Map_out && global_col < K_size) {
                shared_mask[tile_row][tile_col] = mask[global_row * K_size + global_col];
            } else {
                shared_mask[tile_row][tile_col] = 0.0f;
            }
        }

        //  load and unroll input tile
        if (tid < 128) {
            int tile_row = tid / WMMA_N; 
            int tile_col = tid % WMMA_N; 

            int global_k = k_base + tile_row;
            int global_n = warp_col + tile_col;

            //  unrolled position (global_k, global_n) to input position
            if (global_k < K_size && global_n < N_size) {
                int currBatch = n_batch[tile_col];
                if (currBatch >= 0) {
                    int h_out = n_h[tile_col];
                    int w_out = n_w[tile_col];

                    int currChannel = global_k / KK;
                    int remC        = global_k - currChannel * KK;
                    int p           = remC / K;
                    int q           = remC - p * K;

                    int h_in = h_out + p;
                    int w_in = w_out + q;

                    shared_input[tile_row][tile_col] =
                        input[currBatch * (Channel * HxW) +
                              currChannel * HxW +
                              h_in * Width + w_in];
                } else {
                    shared_input[tile_row][tile_col] = 0.0f;
                }
            } else {
                shared_input[tile_row][tile_col] = 0.0f;
            }
        }

        __syncthreads();

        // Only the first 32 threads do WMMA ops
        if (tid < 32) {
            wmma::load_matrix_sync(a_frag, &shared_mask[0][0], WMMA_K);
            wmma::load_matrix_sync(b_frag, &shared_input[0][0], WMMA_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    // Store result
    if (tid < 32) {
        wmma::store_matrix_sync(&shared_output[0][0],
                                acc_frag, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    // All threads write to output
    for (int idx = tid; idx < WMMA_M * WMMA_N; idx += blockDim.x) {
        int tile_row   = idx / WMMA_N; 
        int tile_col   = idx % WMMA_N;
        int global_row = warp_row + tile_row;
        int global_col = warp_col + tile_col;

        if (global_row < Map_out && global_col < N_size) {
            int currBatch = global_col / HoutWout;
            int rem       = global_col - currBatch * HoutWout;
            int h_out     = rem / Width_out;
            int w_out     = rem - h_out * Width_out;

            output[currBatch * (Map_out * HoutWout) +
                   global_row * HoutWout +
                   h_out * Width_out + w_out] =
                shared_output[tile_row][tile_col];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    int Width_out = Width - K + 1;
    int Height_out = Height - K + 1;
    size_t output_size = Batch * Map_out * Width_out * Height_out * sizeof(float);

    cudaMalloc((void **)device_input_ptr, input_size);
    cudaMalloc((void **)device_mask_ptr, mask_size);
    cudaMalloc((void **)device_output_ptr, output_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel
    dim3 blockDim(128, 1, 1);
    // Output matrix dimension: M by N | M is the number of output feature maps | N is the number of total number of output pixels across all images
    int Width_out = Width - K + 1;
    int Height_out = Height - K + 1;
    int M = Map_out;
    int N = Batch * Width_out * Height_out;

    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M, 1);

    matmul_conv_fused<<<gridDim, blockDim>>>(device_mask, device_input, device_output,
                                  Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    int Width_out = Width - K + 1;
    int Height_out = Height - K + 1;
    size_t output_size = Batch * Map_out * Width_out * Height_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}