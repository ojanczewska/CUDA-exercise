
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <chrono>

using namespace std;

const int maxSize = 5000;

struct Matrix
{
	float* elements;
};

void txtfile(float* array, int arraySize, FILE * fPtr) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {

			fprintf(fPtr, "%f", array[i * arraySize + j]);
			fputs("; ", fPtr);
		}
		fputs("\n", fPtr);
	}
	fclose(fPtr);
}

void checkFile(FILE * fPtr) {
	/* fopen() return NULL if last operation was unsuccessful */
	if (fPtr == NULL)
	{
		/* File not created hence exit */
		printf("Unable to create file.\n");
		exit(EXIT_FAILURE);
	}
}


void generateMatrix(Matrix array, int arraySize)
{
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			*(array.elements + i * arraySize + j) = (float)rand() / (float)(RAND_MAX / 100);
		}
	}
}
void showMstrix(Matrix array, int arraySize)
{
	for (int i = 0; i < arraySize; i++) {
		printf("[");
		for (int j = 0; j < arraySize; j++) {
			printf(" %f ", *(array.elements + i * arraySize + j));
		}
		printf("]\n");
	}
}

void addMatrix(float* arrayA, float* arrayB, float*  arrayX, int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i * arraySize + j] = arrayA[i * arraySize + j] + arrayB[i * arraySize + j];
		}
	}
}

void substractMatrix(float* arrayA, float* arrayB, float*  arrayX, int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i * arraySize + j] = arrayA[i * arraySize + j] - arrayB[i * arraySize + j];
		}
	}
}

void transpositionMatrix(float* arrayA, float*  arrayX, int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i * arraySize + j] = arrayA[j * arraySize + i];
		}
	}
}

void scalar_multiplicationMatrix(float* arrayA, float* arrayX, int arraySize, int x) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i * arraySize + j] = x * arrayA[i * arraySize + j];
		}
	}
}


void muliplyMatrix(float* arrayA, float*  arrayB, float* arrayX, int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i * arraySize + j] = 0;
			for (int k = 0; k < arraySize; k++) {
				arrayX[i * arraySize + j] += arrayA[i * arraySize + k] * arrayB[k * arraySize + j];
			}
		}
	}
}

void matrix_CPU(float* arrayA, float* arrayB, float* arrayX, float*  arrayY, int arraySize, int w, int u) {

	//// X = A *B
	muliplyMatrix(arrayA, arrayB, arrayX, arraySize);
	//// A^T
	transpositionMatrix(arrayA, arrayY, arraySize);
	//// u * a^t
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
__device__ void kernel_add(Matrix arrayA, Matrix arrayB, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;


	if (col < arraySize && row < arraySize) {
		arrayX.elements[row * arraySize + col] = arrayA.elements[row * arraySize + col] + arrayB.elements[row * arraySize + col];
	}
}


__device__ void kernel_substract(Matrix arrayA, Matrix arrayB, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < arraySize && row < arraySize) {
		arrayX.elements[row * arraySize + col] = arrayA.elements[row * arraySize + col] - arrayB.elements[row * arraySize + col];
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

__device__ void kernel_scalar(Matrix arrayA, Matrix arrayX, int x, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	arrayX.elements[row * arraySize + col] = x * arrayA.elements[row * arraySize + col];
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


__global__ void kernel(Matrix arrayA, Matrix arrayB, Matrix arrayX, Matrix arrayY, int arraySize, int w, int u) {

	// X = A^T
	kernel_transpose(arrayA, arrayX, arraySize);
	__syncthreads();
	// X = u * A^T
	kernel_scalar(arrayX, arrayX, u, arraySize);
	__syncthreads();
	// X = u * A^T + A
	kernel_add(arrayX, arrayA, arrayX, arraySize);
	__syncthreads();
	//// Y = A * B
	kernel_multiply(arrayA, arrayB, arrayX, arraySize);
	__syncthreads();
	//// X = A * B + u * A^T + A
	kernel_add(arrayX, arrayY, arrayX, arraySize);
	__syncthreads();
	// Y = w * B
	kernel_scalar(arrayB, arrayY, w, arraySize);
	__syncthreads();
	// X = A * B + u * A^T + A - w * B
	kernel_substract(arrayX, arrayY, arrayX, arraySize);
	__syncthreads();

}


__device__ void kernel_multiply_wsp(Matrix arrayA, Matrix arrayB, Matrix arrayX, int arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ float tabA[32][32];
	__shared__ float tabB[32][32];

	int tmp = 0;

	for (int i = 0; i < gridDim.x; i++) {
		if (row * arraySize + i * blockDim.x + threadIdx.x < arraySize*arraySize) {
			tabA[threadIdx.y][threadIdx.x] = arrayA.elements[row * arraySize + i * blockDim.x + threadIdx.x];
		}
		else {
			tabA[threadIdx.y][threadIdx.x] = 0;
		}
		if (i* blockDim.y + threadIdx.y + col * arraySize < arraySize*arraySize) {
			tabB[threadIdx.y][threadIdx.x] = arrayB.elements[(i* blockDim.y + threadIdx.y)*arraySize + col];
		}
		else {
			tabB[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();

		for (int k = 0; k < blockDim.x; k++) {
			tmp += tabA[threadIdx.y][k] * tabB[k][threadIdx.x];
		}
		__syncthreads();


	}

	if (row < arraySize && col < arraySize)
	{
		arrayX.elements[row *arraySize + col] = tmp;
	}


}




__device__ void kernel_transpose_wsp(Matrix arrayA, Matrix arrayX, int arraySize)
{

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ float tabX[32][32];

	tabX[threadIdx.y][threadIdx.x] = arrayA.elements[col * arraySize + row];

	__syncthreads();

	if (col < arraySize && row < arraySize) {

		arrayX.elements[row * arraySize + col] = tabX[threadIdx.y][threadIdx.x];
	}

}




__global__ void kernel_wsp(Matrix arrayA, Matrix arrayB, Matrix arrayX, Matrix arrayY, int arraySize, int w, int u) {


	// X = A^T
	kernel_transpose_wsp(arrayA, arrayX, arraySize);
	__syncthreads();
	// X = u * A^T
	kernel_scalar(arrayX, arrayX, u, arraySize);
	__syncthreads();
	// X = u * A^T + A
	kernel_add(arrayX, arrayA, arrayX, arraySize);
	__syncthreads();
	//// Y = A * B
	kernel_multiply_wsp(arrayA, arrayB, arrayY, arraySize);
	__syncthreads();
	//// X = A * B + u * A^T + A
	kernel_add(arrayX, arrayY, arrayX, arraySize);
	__syncthreads();
	// Y = w * B
	kernel_scalar(arrayB, arrayY, w, arraySize);
	__syncthreads();
	// X = A * B + u * A^T + A - w * B
	kernel_substract(arrayX, arrayY, arrayX, arraySize);
	__syncthreads();


}
void matrix_GPU(Matrix arrayA, Matrix  arrayB, Matrix  arrayX, Matrix  arrayY, int arraySize, int w, int u) {

	Matrix d_a, d_b, d_x, d_y;
	int cuda_malloc2 = sizeof(float) * maxSize*maxSize;

	cudaMalloc(&d_a.elements, cuda_malloc2);
	cudaMalloc(&d_b.elements, cuda_malloc2);
	cudaMalloc(&d_x.elements, cuda_malloc2);
	cudaMalloc(&d_y.elements, cuda_malloc2);

	cudaMemcpy(d_a.elements, arrayA.elements, cuda_malloc2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b.elements, arrayB.elements, cuda_malloc2, cudaMemcpyHostToDevice);

	int blockSize = arraySize;
	if (arraySize > 32) {
		for (int i = 1; i <= 32; i++) {
			if (arraySize%i == 0) {
				blockSize = i;
			}
		}
	}

	dim3 block(blockSize, blockSize);
	dim3 grid(arraySize / block.x, arraySize / block.y);

	kernel << < grid, block >> > (d_a, d_b, d_x, d_y, arraySize, w, u);

	cudaMemcpy(arrayX.elements, d_x.elements, cuda_malloc2, cudaMemcpyDeviceToHost);
	cudaFree(d_a.elements); cudaFree(d_b.elements); cudaFree(d_x.elements); cudaFree(d_y.elements);

}

__global__ void dummy() {

}

void matrix_WSP(Matrix arrayA, Matrix  arrayB, Matrix  arrayX, Matrix  arrayY, int arraySize, int w, int u) {

	Matrix d_a, d_b, d_x, d_y;
	int cuda_malloc2 = sizeof(float) * maxSize*maxSize;


	cudaMalloc(&d_a.elements, cuda_malloc2);
	cudaMalloc(&d_b.elements, cuda_malloc2);
	cudaMalloc(&d_x.elements, cuda_malloc2);
	cudaMalloc(&d_y.elements, cuda_malloc2);

	cudaMemcpy(d_a.elements, arrayA.elements, cuda_malloc2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b.elements, arrayB.elements, cuda_malloc2, cudaMemcpyHostToDevice);

	int blockSize = arraySize;
	if (arraySize > 32) {
		for (int i = 1; i <= 32; i++) {
			if (arraySize%i == 0) {
				blockSize = i;
			}
		}
	}

	dim3 block(blockSize, blockSize);
	dim3 grid(arraySize / block.x, arraySize / block.y);

	kernel_wsp << < grid, block >> > (d_a, d_b, d_x, d_y, arraySize, w, u);

	cudaMemcpy(arrayX.elements, d_x.elements, cuda_malloc2, cudaMemcpyDeviceToHost);
	cudaFree(d_a.elements); cudaFree(d_b.elements); cudaFree(d_x.elements); cudaFree(d_y.elements);

}


int main() {

	Matrix a, b, x_cpu, x_gpu, x_wsp, y, y2;
	srand(time(NULL));
	int w = 2;
	int u = 6;
	FILE * fPtr;
	int size;

	printf("Podaj rozmiar macierzy : ");
	scanf("%d", &size);

	dummy << <1, 1 >> > ();

	int cuda_malloc = sizeof(float) * maxSize*maxSize;
	a.elements = (float*)malloc(cuda_malloc);
	b.elements = (float*)malloc(cuda_malloc);
	x_cpu.elements = (float*)malloc(cuda_malloc);
	x_gpu.elements = (float*)malloc(cuda_malloc);
	x_wsp.elements = (float*)malloc(cuda_malloc);
	y.elements = (float*)malloc(cuda_malloc);
	y2.elements = (float*)malloc(cuda_malloc);

	generateMatrix(a, size);
	//printf("\nMacierz A:\n ");
	//showMstrix(a, size);

	generateMatrix(b, size);
	//printf("\nMacierz B:\n ");
	//showMstrix(b, size);

	printf("\n----------------CPU-------------- :\n ");

	auto CPUstart = chrono::steady_clock::now();
	matrix_CPU(a.elements, b.elements, x_cpu.elements, y.elements, size, w, u);
	auto CPUend = chrono::steady_clock::now();
	chrono::duration<double> elapsedCPU = CPUend - CPUstart;

	printf("\nCode executed in %f ms.\n", elapsedCPU.count());
	//printf("\nMacierz X:\n ");
	//showMstrix(x_cpu, size);

	printf("\n----------------GPU-------------- :\n ");

	auto GPUstart = chrono::steady_clock::now();
	matrix_GPU(a, b, x_gpu, y2, size, w, u);
	auto GPUend = chrono::steady_clock::now();
	chrono::duration<double> elapsedGPU = GPUend - GPUstart;

	printf("\nCode executed in %f ms.\n", elapsedGPU.count());
	//printf("\nMacierz X GPU:\n ");
	//showMstrix(x_gpu, size);


	printf("\n----------------GPU WSP-------------- :\n ");

	auto WSPstart = chrono::steady_clock::now();
	matrix_WSP(a, b, x_wsp, y2, size, w, u);
	auto WSPend = chrono::steady_clock::now();
	chrono::duration<double> elapsedWSP = WSPend - WSPstart;

	printf("\nCode executed in %f ms.\n\n", elapsedWSP.count());
	//printf("\nMacierz X WSP:\n ");
	//showMstrix(x_wsp, size);


	//fPtr = fopen("A.txt", "w");
	//checkFile(fPtr);
	//txtfile(a.elements, size, fPtr);
	//fclose(fPtr);

	//fPtr = fopen("B.txt", "w");
	//checkFile(fPtr);
	//txtfile(b.elements, size, fPtr);
	//fclose(fPtr);

	//fPtr = fopen("X_CPU.txt", "w");
	//checkFile(fPtr);
	//txtfile(x_cpu.elements, size, fPtr);
	//fclose(fPtr);

	//fPtr = fopen("X_WSP.txt", "w");
	//checkFile(fPtr);
	//txtfile(x_wsp.elements, size, fPtr);
	//fclose(fPtr);

	//fPtr = fopen("X_GPU.txt", "w");
	//checkFile(fPtr);
	//txtfile(x_gpu.elements, size, fPtr);
	//fclose(fPtr);

	return 0;
}
