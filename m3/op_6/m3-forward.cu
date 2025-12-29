#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define REG_TILE_M 1
#define REG_TILE_N 4

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
    __shared__ float tileM[TILE_WIDTH][TILE_WIDTH]; // Mask Tile (M x K)
    __shared__ float tileN[TILE_WIDTH][TILE_WIDTH]; // Input Tile (K x N)

    // Block computes a TILE_WIDTH x TILE_WIDTH tile of the output (M x N)
    int blockRow = blockIdx.y * TILE_WIDTH;
    int blockCol = blockIdx.x * TILE_WIDTH;

    // Each thread is responsible for one row and computes REG_TILE_N columns
    int row = blockRow + threadIdx.y;
    int col_base = blockCol + threadIdx.x * REG_TILE_N;

    int Height_out = Height - K + 1;
    int Width_out  = Width  - K + 1;

    // Number of input features per output pixel 
    int K_size = Channel * K * K;
    // Number of total output pixels across batch
    int N_size = Batch * Height_out * Width_out;

    // Number of tiles needed to span the K dimension
    int numTiles = (K_size + TILE_WIDTH - 1) / TILE_WIDTH;

    // Joint register tiling: each thread keeps REG_TILE_N accumulators
    float total[REG_TILE_N];
    #pragma unroll
    for (int ni = 0; ni < REG_TILE_N; ++ni) {
        total[ni] = 0.0f;
    }

    for (int t = 0; t < numTiles; t++) {
        // Load tileM from mask - each thread loads REG_TILE_N elements
        #pragma unroll
        for (int i = 0; i < REG_TILE_N; ++i) {
            int maskRow = blockRow + threadIdx.y;
            int maskCol = t * TILE_WIDTH + threadIdx.x * REG_TILE_N + i;

            if (maskRow < Map_out && maskCol < K_size) {
                tileM[threadIdx.y][threadIdx.x * REG_TILE_N + i] = mask[maskRow * K_size + maskCol];
            } else {
                tileM[threadIdx.y][threadIdx.x * REG_TILE_N + i] = 0.0f;
            }
        }

        // Load tileN from input (implicitly unrolled)
        // Each thread loads REG_TILE_N elements
        #pragma unroll
        for (int i = 0; i < REG_TILE_N; ++i) {
            int unrollRow = t * TILE_WIDTH + threadIdx.y;
            int unrollCol = blockCol + threadIdx.x * REG_TILE_N + i;

            if (unrollRow < K_size && unrollCol < N_size) {
                int currBatch = unrollCol / (Height_out * Width_out);
                int remB = unrollCol % (Height_out * Width_out);
                int h_out = remB / Width_out;
                int w_out = remB % Width_out;

                int currChannel = unrollRow / (K * K);
                int remC = unrollRow % (K * K);
                int p = remC / K;
                int q = remC % K;

                int h_in = h_out + p;
                int w_in = w_out + q;

                tileN[threadIdx.y][threadIdx.x * REG_TILE_N + i] = input[
                    currBatch * (Channel * Height * Width) +
                    currChannel * (Height * Width) +
                    h_in * Width +
                    w_in
                ];
            } else {
                tileN[threadIdx.y][threadIdx.x * REG_TILE_N + i] = 0.0f;
            }
        }

        __syncthreads();

        // Compute: Load S elements from M row into registers, multiply with entire N tile
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            // Load one element from M (this thread's row, column k)
            float m_val = tileM[threadIdx.y][k];
            
            // Multiply with REG_TILE_N elements from N and accumulate
            #pragma unroll
            for (int ni = 0; ni < REG_TILE_N; ++ni) {
                int nCol = threadIdx.x * REG_TILE_N + ni;
                total[ni] += m_val * tileN[k][nCol];
            }
        }

        __syncthreads();
    }

    // Write to corresponding indices in output 
    // Reminder: col is equivalent to the index of the output feature map
    #pragma unroll
    for (int ni = 0; ni < REG_TILE_N; ++ni) {
        int col = col_base + ni;
        
        if (row < Map_out && col < N_size) {
            int currBatch = col / (Height_out * Width_out);
            int rem = col % (Height_out * Width_out);
            int h_out = rem / Width_out;
            int w_out = rem % Width_out;

            output[
                currBatch * (Map_out * Height_out * Width_out) +
                row * (Height_out * Width_out) +
                h_out * Width_out +
                w_out
            ] = total[ni];
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
    dim3 blockDim(TILE_WIDTH / REG_TILE_N, TILE_WIDTH, 1);
    // Output matrix dimension: M by N | M is the number of output feature maps | N is the number of total number of output pixels across all images
    int Width_out = Width - K + 1;
    int Height_out = Height - K + 1;
    int M = Map_out;
    int N = Batch * Width_out * Height_out;

    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);

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