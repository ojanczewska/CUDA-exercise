#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <time.h>


const int maxSize = 50;

void generateMatrix(int array[][maxSize], int arraySize)
{
	
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {

			array[i][j] = (rand() % (10 + 1 + 10) - 10);
		}
	}
}

void showMstrix(int array[maxSize][maxSize], int arraySize) {


	for (int i = 0; i < arraySize; i++) {
		printf("[");
		for (int j = 0; j < arraySize; j++) {
			printf(" %d ", array[i][j]);
		}
		printf("]\n");
	}

}


void muliplyMatrix(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			for (int k = 0; k < arraySize; k++) {
				arrayX[i][j] += arrayA[i][k] * arrayB[k][j];
			}
		}
	}
}

__global__ void kernel_multiply(int* arrayA, int* arrayB, int* arrayX, int arraySize)
{
	int col = blockDim.x * blockDim.x + threadIdx.x;
	int row = blockDim.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < arraySize; i++) {
		arrayX[row * arraySize + col] += arrayA[row * arraySize + i] * arrayB[i * arraySize + col];
	}

}

void addMatrix(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = arrayA[i][j] + arrayB[i][j];
		}
	}
}

__global__ void kernel_add(int* arrayA, int* arrayB, int* arrayX, int arraySize)
{
	int col = blockDim.x * blockDim.x + threadIdx.x;
	int row = blockDim.y * blockDim.y + threadIdx.y;
	if (col < arraySize && row < arraySize) {
		arrayX[row * arraySize + col] = arrayA[row * arraySize + col] + arrayB[row * arraySize + col];
	}
}

void transpositionMatrix(int arrayA[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = arrayA[j][i];
		}
	}
}

__global__ void kernel_transpose(int* arrayA, int* arrayX, int arraySize)
{
	int col = blockDim.x * blockDim.x + threadIdx.x;
	int row = blockDim.y * blockDim.y + threadIdx.y;
	if (col < arraySize && row < arraySize) {
		arrayX[row * arraySize + col] = arrayA[col * arraySize + row];
	}
}

void scalar_multiplicationMatrix(int arrayA[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize, int x) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = x * arrayA[i][j];
		}
	}
}

__global__ void kernel_scalar(int* arrayA, int* arrayX, int arraySize, int x)
{
	int col = blockDim.x * blockDim.x + threadIdx.x;
	int row = blockDim.y * blockDim.y + threadIdx.y;
	if (col < arraySize && row < arraySize) {
		arrayX[row * arraySize + col] = x * arrayA[row * arraySize + col];
	}
}


void GPU_version(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arrayY[maxSize][maxSize], int arraySize, int w, int u);


int main()
{
	const int arraySize = 50;
	srand(time(NULL));

	int a[maxSize][maxSize];
	int b[maxSize][maxSize];
	int x[maxSize][maxSize];
	int y[maxSize][maxSize];
	int u = 6;
	int w = -2;
	int size;
	printf("Podaj rozmiar macierzy : ");
	scanf("%d", &size);


	generateMatrix(a, size);
	showMstrix(a, size);

	generateMatrix(b, size);
	showMstrix(b, size);


	// X = A *B
	muliplyMatrix(a, b, x, size);

	// A^T
	transpositionMatrix(a, y, size);

	// u * A^T
	scalar_multiplicationMatrix(a, y, size, u);

	// X = A * B + u * A^T
	addMatrix(x, y, x, size);

	// X = A * B + u * A^T + A
	addMatrix(x, a, x, size);

	// w* B
	scalar_multiplicationMatrix(b, y, size, w);

	// X = A * B + u * A^T + A - w *B
	addMatrix(x, y, x, size);

	showMstrix(y, size);

	
	return 0;
}
void GPU_version(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arrayY[maxSize][maxSize], int arraySize, int w, int u) {

	// CUDA

	int *d_a[maxSize][maxSize];
	int *d_b[maxSize][maxSize];
	int *d_x[maxSize][maxSize];
	int *d_y[maxSize][maxSize];
	int *d_u;
	int *d_w;
	int *d_size;
	int matSize = maxSize * maxSize;

	cudaMalloc((void**)&d_a, sizeof(int )*  matSize);
	cudaMalloc((void**)&d_b, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_x, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_y, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_u, sizeof(int));
	cudaMalloc((void**)&d_w, sizeof(int));
	cudaMalloc((void**)&d_size, sizeof(int));

	cudaMemcpy(d_a, &arrayA, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &arrayB, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, &arrayX, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, &arrayY, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, &u, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_size, &arraySize, sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(32,32);
	dim3 grid(maxSize/32, maxSize / 32);


	// X = A *B
	
	kernel_multiply << < grid, block >> > (d_a, d_b, d_x, d_size);

	//// a^t
	//kernel_transpose << <grid, block >> > (d_a, d_y, d_size);

	//// u * a^t
	//kernel_scalar << <grid, block >> >  (d_a, d_y, d_size, d_u);

	//// x = a * b + u * a^t
	//kernel_add << <grid, block >> >  (d_x, d_y, d_x, d_size);

	//// x = a * b + u * a^t + a
	//kernel_add << <grid, block >> >  (d_x, d_a, d_x, d_size);

	//// w* b
	//kernel_scalar << <grid, block >> >  (d_b, d_y, d_size, d_w);

	//// x = a * b + u * a^t + a - w *b
	//kernel_add << <grid, block >> >  (d_x, d_y, d_x, d_size);


	cudaMemcpy(&arrayA, d_a, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(&arrayB, d_b, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(&arrayX, d_x, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(&arrayY, d_y, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(&u, d_u, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&w, d_w, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&arraySize, d_size, sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_u);
	cudaFree(d_w);
	cudaFree(d_size);




}