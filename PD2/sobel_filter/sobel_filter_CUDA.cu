/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2022 Riga Technical University
 */
#include "sobel_filter_CUDA.h"

#include <iostream>
#include <cmath>

#include "device_launch_parameters.h"
#include "sobel_filter_common.h"

// #define BLOCK_SIZE	256
#define bSize 256
#define widthStep 1024

__constant__ DWORD palette[MAX_PALETTE_SIZE];

texture<BYTE> byteTexture;
texture<DWORD> dwordTexture;

__global__ void sobelFilter8BPPPaletteKernel(BYTE* output, const BYTE* input, int imgW, int imgH) {
    

    //computing with multiple threads
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIndex + yIndex * imgW;




    //gradient in x and y direction
    int Gx = 0;
    int Gy = 0;

    while (offset < (imgH - 2) * (imgW - 2)) {
        //gradient in x direction
        Gx = -1 * input[(yIndex)*imgW + xIndex] + -2 * input[(yIndex + 1) * imgW + xIndex]
            + 1 * input[(yIndex + 2) * imgW + xIndex] + 0 * input[(yIndex)*imgW + xIndex + 1]
            + 0 * input[(yIndex + 1) * imgW + xIndex + 1] + 0 * input[(yIndex + 2) * imgW + xIndex + 1]
            + 1 * input[(yIndex)*imgW + xIndex + 2] + 2 * input[(yIndex + 1) * imgW + xIndex + 2]
            + 1 * input[(yIndex + 2) * imgW + xIndex + 2];

        //gradient in y direction
        Gy = 1 * input[(yIndex)*imgW + xIndex] + 2 * input[(yIndex + 1) * imgW + xIndex]
            + 1 * input[(yIndex + 2) * imgW + xIndex] + 0 * input[(yIndex)*imgW + xIndex + 1]
            + 0 * input[(yIndex + 1) * imgW + xIndex + 1] + 0 * input[(yIndex + 2) * imgW + xIndex + 1]
            + -1 * input[(yIndex)*imgW + xIndex + 2] + -2 * input[(yIndex + 1) * imgW + xIndex + 2]
            + -1 * input[(yIndex + 2) * imgW + xIndex + 2];

        int sum = abs(Gx) + abs(Gy);
        // constrain the sum with 255
        //if (sum > 255) {
        //    sum = 255;
        //}
        output[offset] = sum;
        xIndex += blockDim.x * gridDim.x;
        if (xIndex > imgW - 2) {
            yIndex += blockDim.y * gridDim.y;
            xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        }

        offset = xIndex + yIndex * imgW;
    }
}

__global__ void sobelFilterRGBAKernel(DWORD* output, const DWORD* input, int imgW, int imgH) {

    //computing with multiple threads
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIndex + yIndex * imgW;




    //gradient in x and y direction
    float Gx = 0;
    float Gy = 0;

    while (offset < (imgH - 2) * (imgW - 2)) {
        //gradient in x direction
        Gx = -1 * input[(yIndex)*imgW + xIndex] + -2 * input[(yIndex + 1) * imgW + xIndex]
            + 1 * input[(yIndex + 2) * imgW + xIndex] + 0 * input[(yIndex)*imgW + xIndex + 1]
            + 0 * input[(yIndex + 1) * imgW + xIndex + 1] + 0 * input[(yIndex + 2) * imgW + xIndex + 1]
            + 1 * input[(yIndex)*imgW + xIndex + 2] + 2 * input[(yIndex + 1) * imgW + xIndex + 2]
            + 1 * input[(yIndex + 2) * imgW + xIndex + 2];

        //gradient in y direction
        Gy = 1 * input[(yIndex)*imgW + xIndex] + 2 * input[(yIndex + 1) * imgW + xIndex]
            + 1 * input[(yIndex + 2) * imgW + xIndex] + 0 * input[(yIndex)*imgW + xIndex + 1]
            + 0 * input[(yIndex + 1) * imgW + xIndex + 1] + 0 * input[(yIndex + 2) * imgW + xIndex + 1]
            + -1 * input[(yIndex)*imgW + xIndex + 2] + -2 * input[(yIndex + 1) * imgW + xIndex + 2]
            + -1 * input[(yIndex + 2) * imgW + xIndex + 2];

        float sum = sqrt(Gx*Gx + Gy*Gy);

        
        // constrain the sum with 255
        //if (sum > 255) {
        //    sum = 255;
        //}
        output[offset] = sum;
        xIndex += blockDim.x * gridDim.x;
        if (xIndex > imgW - 2) {
            yIndex += blockDim.y * gridDim.y;
            xIndex = threadIdx.x + blockIdx.x * blockDim.x;
        }

        offset = xIndex + yIndex * imgW;
    }
}

extern "C" cudaError_t sobelFilterGrayscaleCUDA(const BitmapImage & inputImage, const BitmapImage & outputImage, int maxThreadPerBlock) {
    using namespace std;

    const size_t dataSize = inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE ? inputImage.getWidth() * inputImage.getHeight() : inputImage.getWidth() * inputImage.getHeight() * sizeof(RGBA);

    const int BLOCK_SIZE = sqrt((double)maxThreadPerBlock);

    //dim3 blocks(ceil((double)inputImage.getWidth() / BLOCK_SIZE), ceil((double)inputImage.getHeight() / BLOCK_SIZE));
    
    dim3 blocks(ceil((double)inputImage.getWidth() / BLOCK_SIZE), ceil((double)inputImage.getHeight() / BLOCK_SIZE));
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    BYTE* dev_input = NULL;
    BYTE* dev_output = NULL;

    cudaEvent_t start, stop;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?";
        goto Error;
    }

    // Allocate GPU buffers for vectors (input, output)    .
    cudaStatus = cudaMalloc(&dev_output, dataSize);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_input, dataSize);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!";
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_input, inputImage.getRawData(), dataSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy for input failed!" << endl;
        goto Error;
    }

    if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {
        cudaMemcpyToSymbol(palette, inputImage.getPaletteColors(), inputImage.getNumberOfColors() * sizeof(RGBA), 0, cudaMemcpyHostToDevice);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<BYTE>();
        cudaBindTexture(NULL, byteTexture, dev_input, desc, dataSize);
    }
    else {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<DWORD>();
        cudaBindTexture(NULL, dwordTexture, dev_input, desc, dataSize);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {
        sobelFilter8BPPPaletteKernel << <blocks, threads >> > (dev_output, dev_input, inputImage.getWidth(), inputImage.getHeight());
    }
    else {
        sobelFilterRGBAKernel << <blocks, threads >> > ((DWORD*)dev_output, (DWORD*)dev_input, inputImage.getWidth(), inputImage.getHeight());
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching kernel!" << endl;;
        goto Error;
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << endl;
        goto Error;
    }

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "---------------------" << endl;
    cout << "Elapsed image processing time on GPU: " << elapsedTime << " milliseconds" << endl;
    cout << "---------------------" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outputImage.getRawData(), dev_output, dataSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy for output failed!" << endl;
        goto Error;
    }

    if (inputImage.getNumberOfColors() <= MAX_PALETTE_SIZE) {
        cudaUnbindTexture(byteTexture);
    }
    else {
        cudaUnbindTexture(dwordTexture);
    }

Error:
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaStatus;
}