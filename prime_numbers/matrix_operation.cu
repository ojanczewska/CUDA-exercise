#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <time.h>

const int maxSize = 5000;

struct Matrix
{
	int* elements;

};

void generateMatrix(Matrix array, int arraySize);
void showMstrix(Matrix array, int arraySize);

__device__ void kernel_add(Matrix arrayA, Matrix arrayB, Matrix arrayX)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;


	if (col < maxSize && row < maxSize) {
		arrayX.elements[row * maxSize + col] = arrayA.elements[row * maxSize + col] + arrayB.elements[row * maxSize + col];
	}
}

__device__ void kernel_add_wsp(Matrix arrayA, Matrix arrayB, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__  int tab[32][32];
	__shared__  int tab2[32][32];

	tab[threadIdx.y][threadIdx.x] = arrayA.elements[row * arraySize + col];
	tab2[threadIdx.y][threadIdx.x] = arrayB.elements[row * arraySize + col];

	__syncthreads();

	if (col < arraySize && row < arraySize) {
		arrayX.elements[row * arraySize + col] = tab[threadIdx.y][threadIdx.x] + tab2[threadIdx.y][threadIdx.x];
	}

}

__device__ void kernel_substract(Matrix arrayA, Matrix arrayB, Matrix arrayX)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < maxSize && row < maxSize) {
		arrayX.elements[row * maxSize + col] = arrayA.elements[row * maxSize + col] - arrayB.elements[row * maxSize + col];
	}
}

__device__ void kernel_substract_wsp(Matrix arrayA, Matrix arrayB, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__  int tab[32][32];
	__shared__  int tab2[32][32];

	tab[threadIdx.y][threadIdx.x] = arrayA.elements[row * arraySize + col];
	tab2[threadIdx.y][threadIdx.x] = arrayB.elements[row * arraySize + col];

	__syncthreads();

	if (col < arraySize && row < arraySize) {
		arrayX.elements[row * arraySize + col] = tab[threadIdx.y][threadIdx.x] - tab2[threadIdx.y][threadIdx.x];
	}
}

__device__ void kernel_transpose(Matrix arrayA, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;


	if (col < arraySize && row < arraySize) {
		arrayX.elements[row * arraySize + col] = arrayA.elements[col * arraySize + row];
	}

}

__device__ void kernel_transpose_wsp(Matrix arrayA, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ int tab[32][32];
	tab[threadIdx.y][threadIdx.x] = arrayA.elements[col * arraySize + row];
	__syncthreads();

	if (col < arraySize && row < arraySize) {
		arrayX.elements[row * arraySize + col] = tab[threadIdx.y][threadIdx.x];
	}

}

__device__ void kernel_scalar(Matrix arrayA, Matrix arrayX, int x, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	arrayX.elements[row * arraySize + col] = x * arrayA.elements[row * arraySize + col];
}

__device__ void kernel_scalar_wsp(Matrix arrayA, Matrix arrayX, int x, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ int tab[32][32];
	__shared__ int tab2[32][32];
	tab[threadIdx.y][threadIdx.x] = arrayA.elements[row * arraySize + col];
	tab2[threadIdx.y][threadIdx.x] = x;
	__syncthreads();

	arrayX.elements[row * arraySize + col] = tab2[threadIdx.y][threadIdx.x] * tab[threadIdx.y][threadIdx.x];
}

__device__ void kernel_multiply(Matrix arrayA, Matrix arrayB, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	int sum = 0;

	if (col < arraySize && row < arraySize) {
		for (int i = 0; i < arraySize; i++) {
			sum += arrayA.elements[row * arraySize + i] * arrayB.elements[i *arraySize + col];
		}

		arrayX.elements[row * arraySize + col] = sum;
	}
}

__device__ void kernel_multiply_wsp(Matrix arrayA, Matrix arrayB, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ int tab[32][32];
	__shared__ int tab2[32][32];

	int sum = 0;
	// int idx;

	if (col < arraySize && row < arraySize) {
		for (int i = 0; i < arraySize; i++) {

			tab[threadIdx.y][threadIdx.x] = arrayA.elements[row * arraySize + i];
			tab2[threadIdx.y][threadIdx.x] = arrayB.elements[i *arraySize + col];
			__syncthreads();

			sum += tab[threadIdx.y][threadIdx.x] * tab2[threadIdx.y][threadIdx.x];
			__syncthreads();
		}

		arrayX.elements[row * arraySize + col] = sum;
		//__syncthreads();
	}
}

__global__ void kernel(Matrix arrayA, Matrix arrayB, Matrix arrayX, Matrix arrayY, int arraySize, int w, int u) {

	// X = A^T
	kernel_transpose(arrayA, arrayX, arraySize);
	__syncthreads();
	// X = u * A^T
	kernel_scalar(arrayX, arrayX, u, arraySize);
	__syncthreads();
	// X = u * A^T + A
	kernel_add(arrayX, arrayA, arrayX);
	__syncthreads();

	// Y = w * B
	kernel_scalar(arrayB, arrayY, w, arraySize);
	__syncthreads();
	// X = A * B + u * A^T + A - w * B
	kernel_substract(arrayX, arrayY, arrayX);
	__syncthreads();

	//// Y = A * B
	kernel_multiply(arrayA, arrayB, arrayY, arraySize);
	__syncthreads();
	//// X = A * B + u * A^T + A
	kernel_add(arrayX, arrayY, arrayX);
	__syncthreads();
	
}

__global__ void kernel_wsp(Matrix arrayA, Matrix arrayB, Matrix arrayX, Matrix arrayY, int arraySize, int w, int u) {

	// X = A^T
    kernel_transpose_wsp( arrayA,  arrayX,  arraySize);
    __syncthreads();
	// X = u * A^T
    kernel_scalar_wsp(arrayX, arrayX,  u, arraySize);
    __syncthreads();
	// X = u * A^T + A
    kernel_add_wsp( arrayX,  arrayA,  arrayX,  arraySize);
    __syncthreads();

	// Y = w * B
    kernel_scalar_wsp(arrayB, arrayY,  w, arraySize);
    __syncthreads();
	// X = A * B + u * A^T + A - w * B
    kernel_substract_wsp(arrayX, arrayY, arrayX,  arraySize);
	__syncthreads();
	// Y = A * B
	kernel_multiply_wsp(arrayA, arrayB, arrayY, arraySize);
	//kernel_add_wsp(arrayA, arrayB, arrayX, arraySize);
	//__syncthreads();
	// X = A * B + u * A^T + A
	//kernel_add_wsp( arrayX,  arrayY,  arrayX,  arraySize);
   //__syncthreads();
	
}


void matrix_GPU(Matrix arrayA, Matrix  arrayB, Matrix  arrayX, Matrix  arrayY, int arraySize, int w, int u) {

	Matrix d_a, d_b, d_x, d_y;
	int cuda_malloc2 = sizeof(int) * maxSize*maxSize;

	cudaMalloc(&d_a.elements, cuda_malloc2);
	cudaMalloc(&d_b.elements, cuda_malloc2);
	cudaMalloc(&d_x.elements, cuda_malloc2);
	cudaMalloc(&d_y.elements, cuda_malloc2);

	cudaMemcpy(d_a.elements, arrayA.elements, cuda_malloc2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b.elements, arrayB.elements, cuda_malloc2, cudaMemcpyHostToDevice);

	int z = arraySize + 1 / 32;
	dim3 block(32, 32);
	dim3 grid(z, z);

	kernel << < grid, block >> > (d_a, d_b, d_x, d_y, arraySize, w, u);

	cudaMemcpy(arrayX.elements, d_x.elements, cuda_malloc2, cudaMemcpyDeviceToHost);

	cudaFree(d_a.elements); cudaFree(d_b.elements); cudaFree(d_x.elements); cudaFree(d_y.elements);

}

void matrix_WSP(Matrix arrayA, Matrix  arrayB, Matrix  arrayX, Matrix  arrayY, int arraySize, int w, int u) {

	Matrix d_a, d_b, d_x, d_y;
	int cuda_malloc2 = sizeof(int) * maxSize*maxSize;

	cudaMalloc(&d_a.elements, cuda_malloc2);
	cudaMalloc(&d_b.elements, cuda_malloc2);
	cudaMalloc(&d_x.elements, cuda_malloc2);
	cudaMalloc(&d_y.elements, cuda_malloc2);

	cudaMemcpy(d_a.elements, arrayA.elements, cuda_malloc2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b.elements, arrayB.elements, cuda_malloc2, cudaMemcpyHostToDevice);

	int z = arraySize + 1 / 32;
	dim3 block(32, 32);
	dim3 grid(z, z);

	kernel_wsp << < grid, block >> > (d_a, d_b, d_x, d_y, arraySize, w, u);

	cudaMemcpy(arrayX.elements, d_x.elements, cuda_malloc2, cudaMemcpyDeviceToHost);
	//cudaMemcpy(arrayY.elements, d_y.elements, cuda_malloc2, cudaMemcpyDeviceToHost);

	cudaFree(d_a.elements); cudaFree(d_b.elements); cudaFree(d_x.elements); cudaFree(d_y.elements);

}

void muliplyMatrix(int** arrayA, int**   arrayB, int**  arrayX, int arraySize) {
	
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = 0;
			for (int k = 0; k < arraySize; k++) {
				arrayX[i][j] += arrayA[i][k] * arrayB[k][j];
			}
		}
	}
}

void addMatrix(int**  arrayA, int** arrayB, int**   arrayX, int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = arrayA[i][j] + arrayB[i][j];
		}
	}
}
void substractMatrix(int**  arrayA, int** arrayB, int**   arrayX, int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = arrayA[i][j] - arrayB[i][j];
		}
	}
}
void transpositionMatrix(int**  arrayA, int**   arrayX, int arraySize) {
	
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = arrayA[j][i];
		}
	}
}
void scalar_multiplicationMatrix(int** arrayA, int**  arrayX, int arraySize, int x) {
	
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = x * arrayA[i][j];
		}
	}
}

void matrix_CPU(int** arrayA, int** arrayB, int** arrayX, int**  arrayY, int arraySize, int w, int u){

	// X = A *B
	muliplyMatrix(arrayA, arrayB, arrayX, arraySize);
	// A^T
	transpositionMatrix(arrayA, arrayY, arraySize);
	// u * a^t
	scalar_multiplicationMatrix(arrayY, arrayY, arraySize, u);
	// x = a * b + u * a^t
	addMatrix(arrayX, arrayY, arrayX, arraySize);
	// x = a * b + u * a^t + a
	addMatrix(arrayX, arrayA, arrayX, arraySize);
	// w* b
	scalar_multiplicationMatrix(arrayB, arrayY, arraySize, w);
	// x = a * b + u * a^t + a - w *b
	substractMatrix(arrayX, arrayY, arrayX, arraySize);

}

int main() {

	Matrix a, b, x_cpu, x_gpu, x_wsp, y, y2;
	srand(time(NULL));
	int size = 3;
	int w = 2;
	int u = 6;
	int cuda_malloc = sizeof(int) * maxSize*maxSize;
	a.elements = (int*)malloc(cuda_malloc);
	b.elements = (int*)malloc(cuda_malloc);
	x_cpu.elements = (int*)malloc(cuda_malloc);
	x_gpu.elements = (int*)malloc(cuda_malloc);
	x_wsp.elements = (int*)malloc(cuda_malloc);
	y.elements = (int*)malloc(cuda_malloc);
	y2.elements = (int*)malloc(cuda_malloc);

	generateMatrix(a, size);
	printf("\nMacierz A:\n ");
	showMstrix(a, size);

	generateMatrix(b, size);
	printf("\nMacierz B:\n ");
	showMstrix(b, size);

	matrix_GPU(a, b, x_cpu, y, size, w, u);

	printf("\n----------------CPU-------------- :\n ");
	printf("\nMacierz X:\n ");
	showMstrix(x_cpu, size);


	printf("\nMacierz Y :\n ");
	showMstrix(y, size);

	printf("\n----------------GPU-------------- :\n ");

	matrix_GPU(a, b, x_gpu,y2, size, w,  u);

	//matrix_WSP(a, b, x_wsp, y2, size, w, u);

    printf("\nMacierz X:\n ");
    showMstrix(x_gpu, size);

	printf("\nMacierz Y :\n ");
	showMstrix(y2, size);

	printf("\n----------------GPU WSP-------------- :\n ");

	matrix_WSP(a, b, x_wsp, y2, size, w, u);

	printf("\nMacierz X WSP:\n ");
	showMstrix(x_wsp, size);

	return 0;
}


void generateMatrix(Matrix array, int arraySize)
{
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			*(array.elements + i * arraySize + j) = (rand() % (10 + 1 + 10) - 10);
		}
	}
}
void showMstrix(Matrix array, int arraySize)
{
	for (int i = 0; i < arraySize; i++) {
		printf("[");
		for (int j = 0; j < arraySize; j++) {
			printf(" %d ", *(array.elements + i * arraySize + j));
		}
		printf("]\n");
	}
}