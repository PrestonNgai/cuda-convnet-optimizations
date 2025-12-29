#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"
#include <cublas_v2.h>

#define PERMUTE_BLOCK_SIZE 256
#define UNROLL_BLOCK_SIZE 256
__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here

    // Thread's linear index -- the thread handles only one batch, channel, h_out, w_out
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    // width of the unrolled feature matrix (one image)
    int W_unroll = Height_out * Width_out;
    size_t W_unroll_total = (size_t)Batch * (size_t)W_unroll; // width after batching

    size_t thread_total = (size_t)Channel * (size_t)Batch * (size_t)W_unroll;

    if (t < thread_total) {
        // get indices
        int b = t / (Channel * W_unroll);
        int rem = t % (Channel * W_unroll);
        int c = rem / W_unroll;
        int w_unroll = rem % W_unroll;

        int h_out = w_unroll / Width_out;
        int w_out = w_unroll % Width_out;
        // Starting row index for unrolled section in the channel
        int h_base = c * K * K;
        size_t w_col = (size_t)b * (size_t)W_unroll + (size_t)w_unroll;

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = h_base + p * K + q;

                size_t output_idx = (size_t)h_unroll * W_unroll_total + w_col;
                float val = in_4d(b, c, h_out + p, w_out + q);
                output[output_idx] = val;
            }
        }
    }

    #undef in_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

static cublasHandle_t cublas_handle = nullptr;

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    size_t input_size = (size_t)Batch * (size_t)Channel * (size_t)Height * (size_t)Width * sizeof(float);
    size_t mask_size = (size_t)Map_out * (size_t)Channel * (size_t)K * (size_t)K * sizeof(float);
    size_t output_size = (size_t)Batch * (size_t)Map_out * (size_t)Height_out * (size_t)Width_out * sizeof(float);

    cudaMalloc((void **)device_input_ptr, input_size);
    cudaMalloc((void **)device_mask_ptr, mask_size);
    cudaMalloc((void **)device_output_ptr, output_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    if (cublas_handle == nullptr) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            exit(-1);
        }
    }
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
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    size_t num_threads = (size_t)Batch * (size_t)Channel * (size_t)Height_out * (size_t)Width_out;
    size_t num_blocks = (num_threads + UNROLL_BLOCK_SIZE - 1) / UNROLL_BLOCK_SIZE;
    dim3 gridDim(num_blocks, 1, 1);
    dim3 blockDim(UNROLL_BLOCK_SIZE, 1, 1);
    matrix_unrolling_kernel<<<gridDim, blockDim>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);


    // Matrix multiplication and permutation. Do not modify.
    // Multiply the mask with the unrolled matrix
    /*dim3 matmul_grid_dim((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1,
                         (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim>>>(
        device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled,
        Height_unrolled, Width_unrolled, Map_out, Width_unrolled
    ); */

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Width_unrolled,     
        Map_out,            
        Height_unrolled,    
        &alpha,
        unrolled_matrix,    
        Width_unrolled,     
        device_mask,        
        Height_unrolled,    
        &beta,
        matmul_output,      
        Width_unrolled      
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        exit(-1);
    }
    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t output_size = (size_t)(Batch * Map_out * Height_out * Width_out) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    
    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
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