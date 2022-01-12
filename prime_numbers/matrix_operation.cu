#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <time.h>


const int maxSize = 50;

void generateMatrix(int array[][maxSize], int arraySize);
void showMstrix(int array[maxSize][maxSize], int arraySize);
void muliplyMatrix(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize);
void transpositionMatrix(int arrayA[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize);
void scalar_multiplicationMatrix(int arrayA[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize, int x);
//void GPU_version(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arrayY[maxSize][maxSize], int arraySize, int w, int u);
void addMatrix(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize);
void txtfile(int array[][maxSize], int arraySize, FILE * fPtr);
void checkFile(FILE * fPtr);

//__global__ void kernel_add(int *arrayA, int *arrayB, int *arrayX, int arraySize)
//{
//	int col = blockDim.x * blockDim.x + threadIdx.x;
//	int row = blockDim.y * blockDim.y + threadIdx.y;
//	if (col < arraySize && row < arraySize) {
//		arrayX[row * arraySize + col] = arrayA[row * arraySize + col] + arrayB[row * arraySize + col];
//	}
//}

__global__ void kernel_add(int *arrayA,  int arraySize)
{
	int col = blockDim.x * blockDim.x + threadIdx.x;
	int row = blockDim.y * blockDim.y + threadIdx.y;
	arrayA[row * arraySize + col ] = arraySize;
	
}


int main()
{
	//const int arraySize = 500;
	srand(time(NULL));

	int a[maxSize][maxSize];
	int b[maxSize][maxSize];
	int x[maxSize][maxSize];
	int y[maxSize][maxSize];
	int u = 6;
	int w = -2;
	int size;

	printf("\nObliczana operacja: X = A * B + u * A^T + A - w *B \n ");
	printf("Podaj rozmiar macierzy : ");
	scanf("%d", &size);


	generateMatrix(a, size);
	printf("\nMacierz A:\n ");
	showMstrix(a, size);

	generateMatrix(b, size);
	printf("\nMacierz B:\n ");
	showMstrix(b, size);



	int *d_a;
	int *d_b;
	int *d_x;
	//int *d_y;
	/*int *d_u;
	int *d_w;*/
	int *d_size;
	int matSize = maxSize * maxSize;

	cudaMalloc((void**)&d_a, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_b, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_x, sizeof(int)*  matSize);
	/*cudaMalloc((void**)&d_y, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_u, sizeof(int));
	cudaMalloc((void**)&d_w, sizeof(int));*/
	cudaMalloc((void**)&d_size, sizeof(int));

	cudaMemcpy(d_a, &a, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_y, &y, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_u, &u, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);

	int z = maxSize / 32 + 1;
	dim3 block(32, 32);
	dim3 grid(z, z);

	kernel_add << < grid, block >> > (d_a,d_size);


	cudaMemcpy(&a, d_a, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(&b, d_b, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x, d_x, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(&y, d_y, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(&u, d_u, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&w, d_w, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&size, d_size, sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_x);
	//cudaFree(d_y);
	//cudaFree(d_u);
	//cudaFree(d_w);
	cudaFree(d_size);


	///* File pointer to hold reference to our file */
	//FILE * fPtr;
	///*
	// * Open file in w (write) mode.
	// * "data/file1.txt" is complete path to create file
	// */
	//fPtr = fopen("A.txt", "w");
	//checkFile(fPtr);
	//

	//txtfile(a, size, fPtr);
	//fPtr = fopen("B.txt", "w");
	//checkFile(fPtr);
	//txtfile(b, size, fPtr);
	//

	//// X = A *B
	//muliplyMatrix(a, b, x, size);

	//// A^T
	//transpositionMatrix(a, y, size);

	//// u * a^t
	//scalar_multiplicationMatrix(y, y, size, u);

	//// x = a * b + u * a^t
	//addMatrix(x, y, x, size);

	//// x = a * b + u * a^t + a
	//addMatrix(x, a, x, size);

	//// w* b
	//scalar_multiplicationMatrix(b, y, size, w);

	//// x = a * b + u * a^t + a - w *b
	//addMatrix(x, y, x, size);

	//printf("\nmacierz x:\n ");
	//showMstrix(x, size);
	//fPtr = fopen("X.txt", "w");
	//checkFile(fPtr);
	//txtfile(x, size, fPtr);

	//GPU_version(a, b, x, y, size,w, u) {

	//
	return 0;
}

void txtfile(int array[][maxSize], int arraySize, FILE * fPtr) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {

			fprintf(fPtr, "%d", array[i][j]);
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
			arrayX[i][j] = 0;
			for (int k = 0; k < arraySize; k++) {
				arrayX[i][j] += arrayA[i][k] * arrayB[k][j];
			}
		}
	}
}

void addMatrix(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = arrayA[i][j] + arrayB[i][j];
		}
	}
}

void transpositionMatrix(int arrayA[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = arrayA[j][i];
		}
	}
}

void scalar_multiplicationMatrix(int arrayA[maxSize][maxSize], int arrayX[maxSize][maxSize], int arraySize, int x) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			arrayX[i][j] = x * arrayA[i][j];
		}
	}
}

//void GPU_version(int arrayA[maxSize][maxSize], int arrayB[maxSize][maxSize], int arrayX[maxSize][maxSize], int arrayY[maxSize][maxSize], int arraySize, int w, int u) {
//
//	// CUDA
//
//	int *d_a[maxSize][maxSize];
//	int *d_b[maxSize][maxSize];
//	int *d_x[maxSize][maxSize];
//	int *d_y[maxSize][maxSize];
//	int *d_u;
//	int *d_w;
//	int *d_size;
//	int matSize = maxSize * maxSize;
//
//	cudaMalloc((void**)&d_a, sizeof(int )*  matSize);
//	cudaMalloc((void**)&d_b, sizeof(int)*  matSize);
//	cudaMalloc((void**)&d_x, sizeof(int)*  matSize);
//	cudaMalloc((void**)&d_y, sizeof(int)*  matSize);
//	cudaMalloc((void**)&d_u, sizeof(int));
//	cudaMalloc((void**)&d_w, sizeof(int));
//	cudaMalloc((void**)&d_size, sizeof(int));
//
//	cudaMemcpy(d_a, &arrayA, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, &arrayB, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_x, &arrayX, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_y, &arrayY, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_u, &u, sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_size, &arraySize, sizeof(int), cudaMemcpyHostToDevice);
//
//	dim3 block(32,32);
//	dim3 grid(maxSize/32, maxSize / 32);
//
//
//	// X = A *B
//	
//	//kernel_multiply << < grid, block >> > (d_a, d_b, d_x, d_size);
//	//kernel_add(int* arrayA, int* arrayB, int* arrayX, int arraySize)
//	
//	//// a^t
//	//kernel_transpose << <grid, block >> > (d_a, d_y, d_size);
//
//	//// u * a^t
//	//kernel_scalar << <grid, block >> >  (d_a, d_y, d_size, d_u);
//
//	//// x = a * b + u * a^t
//	//kernel_add << <grid, block >> >  (d_x, d_y, d_x, d_size);
//
//	//// x = a * b + u * a^t + a
//	//kernel_add << <grid, block >> >  (d_x, d_a, d_x, d_size);
//
//	//// w* b
//	//kernel_scalar << <grid, block >> >  (d_b, d_y, d_size, d_w);
//
//	//// x = a * b + u * a^t + a - w *b
//	//kernel_add << <grid, block >> >  (d_x, d_y, d_x, d_size);
//
//
//	cudaMemcpy(&arrayA, d_a, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(&arrayB, d_b, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(&arrayX, d_x, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(&arrayY, d_y, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
//	cudaMemcpy(&u, d_u, sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&w, d_w, sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&arraySize, d_size, sizeof(int), cudaMemcpyDeviceToHost);
//
//
//	cudaFree(d_a);
//	cudaFree(d_b);
//	cudaFree(d_x);
//	cudaFree(d_y);
//	cudaFree(d_u);
//	cudaFree(d_w);
//	cudaFree(d_size);
//
//
//
//
//}

//__global__ void kernel_multiply(int* arrayA, int* arrayB, int* arrayX, int arraySize)
//{
//	int col = blockDim.x * blockDim.x + threadIdx.x;
//	int row = blockDim.y * blockDim.y + threadIdx.y;
//
//	for (int i = 0; i < arraySize; i++) {
//		arrayX[row * arraySize + col] += arrayA[row * arraySize + i] * arrayB[i * arraySize + col];
//	}
//	__global__ void kernel_transpose(int* arrayA, int* arrayX, int arraySize)
//	{
//		int col = blockDim.x * blockDim.x + threadIdx.x;
//		int row = blockDim.y * blockDim.y + threadIdx.y;
//		if (col < arraySize && row < arraySize) {
//			arrayX[row * arraySize + col] = arrayA[col * arraySize + row];
//		}
//	}
//	__global__ void kernel_scalar(int* arrayA, int* arrayX, int arraySize, int x)
//	{
//		int col = blockDim.x * blockDim.x + threadIdx.x;
//		int row = blockDim.y * blockDim.y + threadIdx.y;
//		if (col < arraySize && row < arraySize) {
//			arrayX[row * arraySize + col] = x * arrayA[row * arraySize + col];
//		}
//	}