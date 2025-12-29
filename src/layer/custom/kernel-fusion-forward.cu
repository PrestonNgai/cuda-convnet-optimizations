#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
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
    __shared__ float tileM[TILE_WIDTH][TILE_WIDTH]; // Mask Tile
    __shared__ float tileN[TILE_WIDTH][TILE_WIDTH]; // Input Tile

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    float total = 0.0f;
    // Number of input features per output pixel 
    // (width of the filter matrix | height of the input feature map)
    int K_size = Channel * K * K;
    // Number of total output pixels across batch
    int N_size = Batch * Height_out * Width_out;


    // Number of tiles needed to span the K dimension
    int numTiles = (K_size + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        // Load mask tile
        int maskRow = row;
        int maskCol = t * TILE_WIDTH + threadIdx.x;

        if (maskRow < Map_out && maskCol < K_size) {
            tileM[threadIdx.y][threadIdx.x] = mask[maskRow * K_size + maskCol];
        } else {
            tileM[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load input tile
        // Corresponds to a batch, h_out, w_out index
        int unrollRow = t * TILE_WIDTH + threadIdx.y;
        // Corresponds to a channel, p, q index in the kernel
        int unrollCol = col;

        if (unrollRow < K_size && unrollCol < N_size) {
            int currBatch = unrollCol / (Height_out * Width_out);
            // h_out and w_out correspond to the output "spatial" position within the currBatch 
            int remB = unrollCol % (Height_out * Width_out);
            int h_out = remB / Width_out;
            int w_out = remB % Width_out;

            int currChannel = unrollRow / (K * K);
            // Similarly p, q corresponds to the pixel's location within the current Channel
            int remC = unrollRow % (K * K);
            int p = remC / K;
            int q = remC % K;

            // Position in the original input;
            int h_in = h_out + p;
            int w_in = w_out + q;

            tileN[threadIdx.y][threadIdx.x] = input[currBatch * (Channel * Height * Width) + 
                currChannel * (Height * Width) + 
                h_in * Width + 
                w_in];
        } else {
            tileN[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            total += tileM[threadIdx.y][i] * tileN[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write to corresponding index in output 
    // Reminder: col is equivelent to the inndex of the output feature map
    if (row < Map_out && col < N_size) {
        int currBatch = col / (Height_out * Width_out);
        int rem = col % (Height_out * Width_out);
        int h_out = rem / Width_out;
        int w_out = rem % Width_out;

        output[currBatch * (Map_out * Height_out * Width_out) + 
            row * (Height_out * Width_out) +
            h_out * Width_out + 
            w_out] = total;
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
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
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