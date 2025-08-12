#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

// 矩阵乘法核函数极简版，只支持N*N矩阵乘法
__global__ void matrixMul_simple_kernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMul_simple(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.size(0);
    dim3 block(16, 16);
    dim3 grid((N + 16 -1)/16, (N + 16 -1)/16);
    matrixMul_simple_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
}


// 矩阵乘法核函数优化版，使用共享内存，还是只支持N*N矩阵乘法
#define TILE_WIDTH 16

__global__ void matrixMul_shared_kernel(float *A, float *B, float *C, int N) {
    // 使用共享内存，每次从A，B中加载一个TILE_WIDTH*TILE_WIDTH的子矩阵
    __shared__ float SMemA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float SMemB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;
    // 分N/TILE_WIDTH次加载，每个线程加载自己对应位置
    for (int i = 0; i < (N + TILE_WIDTH - 1)/TILE_WIDTH; i++) {
        if (row < N && (i*TILE_WIDTH + tx) < N)
            // A每次去对应行加载对应TILE段的数据
            SMemA[ty][tx] = A[row*N + i*TILE_WIDTH + tx];
        else SMemA[ty][tx] = 0.0f;
        if (col < N && (i*TILE_WIDTH + ty) < N)
            // B每次去对应列加载对应TILE段的数据
            SMemB[ty][tx] = B[(i*TILE_WIDTH + ty)*N + col];
        else SMemB[ty][tx] = 0.0f;
        // 需要确保块中所有线程都加载完成，才能进行计算，写后读(read-after-write)依赖
        __syncthreads();
        // 计算对应位置的乘积和
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += SMemA[ty][k] * SMemB[k][tx];
        }
        // 确保块中所有线程都计算完成，才能进行下一轮加载，读后写(write-after-read)依赖
        __syncthreads();
    }
    if (row < N && col < N)
        C[row*N + col] = sum;
}

void matrixMul_shared(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.size(0);
    dim3 block(16, 16);
    dim3 grid((N + 16 -1)/16, (N + 16 -1)/16);
    matrixMul_shared_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
}


// 矩阵乘法核函数近一步优化版，使用共享内存，支持M*N和N*P矩阵乘法

__global__ void matrixMul_shared_kernel_2(float *A, float *B, float *C, int M, int N, int P) {
    __shared__ float SMemA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float SMemB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    for (int i =0 ; i < (N + TILE_WIDTH - 1)/TILE_WIDTH; i++) {
        if (row < M && (i*TILE_WIDTH + tx) < N)
            SMemA[ty][tx] = A[row*N + i*TILE_WIDTH + tx];
        else SMemA[ty][tx] = 0.0f;
        if (col < P && (i*TILE_WIDTH + ty) < N)
            SMemB[ty][tx] = B[(i*TILE_WIDTH + ty)*P + col];
        else SMemB[ty][tx] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += SMemA[ty][k] * SMemB[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < P)
        C[row*P + col] = sum;
}

void matrixMul_shared_2(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int N = A.size(1);
    int P = B.size(1);
    dim3 block(16, 16);
    // !!!!!!x维对应列，y维对应行
    dim3 grid((P + 16 -1)/16, (M + 16 -1)/16);
    matrixMul_shared_kernel_2<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, P);
}

// 矩阵乘法核函数优化版3，使用共享内存，支持M*N和N*P矩阵乘法，通过读取设备能力确认TILE大小
int calculate_max_tile_size(int device_id = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    // 限制1：块的共享内存限制
    // 矩阵乘法中，每个块需要加载2个TILE_WIDTH*TILE_WIDTH(float)的子矩阵
    // 内存需求：2 * TILE_WIDTH * TILE_WIDTH * sizeof(float)
    size_t shared_mem_per_block = prop.sharedMemPerBlock;
    size_t available_shared_mem = shared_mem_per_block * 0.9;  // 保留10%的共享内存用于其他用途
    int max_tile_width_from_shared_mem = (int)std::sqrt(available_shared_mem / (2 * sizeof(float)));

    // 限制2: 线程数限制
    // 每个block需要TILE_WIDTH×TILE_WIDTH个线程
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_tile_width_from_threads = (int)std::sqrt(max_threads_per_block);

    int theoretical_max_tile_width = std::min(max_tile_width_from_shared_mem, max_tile_width_from_threads);

    // 限制3: warp对齐优化
    // TILE_WIDTH应该是warp size的倍数以获得最佳性能
    int warp_size = prop.warpSize;
    int max_tile_width = (theoretical_max_tile_width / warp_size) * warp_size;
    if (max_tile_width < warp_size)
        max_tile_width = warp_size;
    printf("max_tile_width: %d\n", max_tile_width);
    return max_tile_width;
}

template<int tile_width>
__global__ void matrixMul_shared_kernel_3(float *A, float *B, float *C, int M, int N, int P) {
    __shared__ float SMemA[tile_width][tile_width];
    __shared__ float SMemB[tile_width][tile_width];

    int tx = threadIdx.x, ty = threadIdx.y;

    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + tx;

    float sum = 0.0f;
    for (int i =0 ; i < (N + tile_width - 1)/tile_width; i++) {
        if (row < M && (i*tile_width + tx) < N)
            SMemA[ty][tx] = A[row*N + i*tile_width + tx];
        else SMemA[ty][tx] = 0.0f;
        if (col < P && (i*tile_width + ty) < N)
            SMemB[ty][tx] = B[(i*tile_width + ty)*P + col];
        else SMemB[ty][tx] = 0.0f;
        __syncthreads();
        for (int k = 0; k < tile_width; k++) {
            sum += SMemA[ty][k] * SMemB[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < P)
        C[row*P + col] = sum;
}

void matrixMul_shared_3(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int N = A.size(1);
    int P = B.size(1);
    int max_tile_width = calculate_max_tile_size();
    // 由于模板参数必须是编译时常量，我们需要使用固定值或者switch语句来确定TILE_WIDTH
    if (max_tile_width >= 32) {
        dim3 block(32, 32);
        dim3 grid((P + 32 -1)/32, (M + 32 -1)/32);
        matrixMul_shared_kernel_3<32><<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, P);
    } else if (max_tile_width >= 16) {
        dim3 block(16, 16);
        dim3 grid((P + 16 -1)/16, (M + 16 -1)/16);
        matrixMul_shared_kernel_3<16><<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, P);
    } else {
        dim3 block(8, 8);
        dim3 grid((P + 8 -1)/8, (M + 8 -1)/8);
        matrixMul_shared_kernel_3<8><<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, P);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrixMul_simple", &matrixMul_simple, "Matrix Multiplication");
    m.def("matrixMul_shared", &matrixMul_shared, "Matrix Multiplication");
    m.def("matrixMul_shared_2", &matrixMul_shared_2, "Matrix Multiplication");
    m.def("matrixMul_shared_3", &matrixMul_shared_3, "Matrix Multiplication");
}

