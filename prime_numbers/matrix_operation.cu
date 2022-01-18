#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <time.h>


const int maxSize = 400;

int* a[maxSize][maxSize];
int* b[maxSize][maxSize];


int* x_GPU[maxSize][maxSize];
int* x_CPU[maxSize][maxSize];


void generateMatrix(int* array[maxSize][maxSize], int arraySize);
void showMstrix(int* array[maxSize][maxSize], int arraySize);
void muliplyMatrix(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int* arrayX[maxSize][maxSize], int arraySize);
void transpositionMatrix(int* arrayA[maxSize][maxSize], int* arrayX[maxSize][maxSize], int arraySize);
void scalar_multiplicationMatrix(int* arrayA[maxSize][maxSize], int* arrayX[maxSize][maxSize], int arraySize, int x);
void addMatrix(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int*arrayX[maxSize][maxSize], int arraySize);
void txtfile(int* array[][maxSize], int arraySize, FILE * fPtr);
void checkFile(FILE * fPtr);


void matrix_GPU(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int arraySize, int* x[maxSize][maxSize], int w, int u);

void matrix_CPU(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int* x[maxSize][maxSize], int arraySize, int u, int w);

__device__ void kernel_multiply(int* arrayA, int* arrayB, int* arrayX, int* arraySize);
__device__ void kernel_scalar(int* arrayA, int* arrayX, int* x);
__device__ void kernel_transpose(int* arrayA, int* arrayX);
__device__ void kernel_add(int* arrayA, int* arrayB, int* arrayX);

__global__ void kernel(int* d_a, int* d_b, int* d_size, int* d_x, int* d_y, int* d_w, int* d_u, int* d_y1, int* d_y2, int* d_y3, int* d_y4 ) {

	//// y1 = A *B 
	kernel_multiply(d_a, d_b, d_y1, d_size);
	__syncthreads();

	//// y =a^t
	kernel_transpose(d_a, d_y);
	__syncthreads();

	//////  y = u * a^t
	kernel_scalar (d_y, d_y, d_u);
	__syncthreads();

	//////// y2= u * a^t + a
	kernel_add(d_y, d_a, d_y2);
	__syncthreads();

	////////w* b = y3
	kernel_scalar(d_b, d_y3,  d_w);
	__syncthreads(); 

	//////// z4= a * b + u * a^t +a
	kernel_add (d_y1, d_y2,  d_y4);
	__syncthreads();

	//////// z4 = a * b + u * a^t + a - w *b
	kernel_add (d_y4, d_y3, d_x);
	__syncthreads();
}



int main()
{
	//const int arraySize = 500;
	
	//int *a = (int *)malloc(maxSize * maxSize * sizeof(int));
	//int** a;
	//cudaMallocHost((void **)&a, sizeof(int)*maxSize*maxSize);
	//int *a = (int*)malloc(maxSize * maxSize * sizeof(int));


	cudaEvent_t start, stop;
	float elapsedTime;
	srand(time(NULL));

	//
	

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


	/* File pointer to hold reference to our file */
	FILE * fPtr;
	/*
	 * Open file in w (write) mode.
	 * "data/file1.txt" is complete path to create file
	 */
	fPtr = fopen("A.txt", "w");
	checkFile(fPtr);
	

	txtfile(a, size, fPtr);
	fPtr = fopen("B.txt", "w");
	checkFile(fPtr);
	txtfile(b, size, fPtr);
	

	//CPU VERSION

	//cudaEventCreate(&start);
	//cudaEventRecord(start, 0);

	matrix_CPU(a,b,x_CPU,size,u,w);

	//cudaEventCreate(&stop);
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);

	//cudaEventElapsedTime(&elapsedTime, start, stop);
	//printf("CPU Elapsed time : %f ms\n", elapsedTime);

	printf("\nMacierz X CPU\n ");
	showMstrix(x_CPU, size);

	//GPU VERSION 

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	matrix_GPU(a,b, size,x_GPU,w,u);

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf(" GPU Elapsed time : %f ms\n", elapsedTime);

	printf("\nMacierz X GPU\n ");
	showMstrix(x_GPU, size);

	fPtr = fopen("X.txt", "w");
	checkFile(fPtr);
	txtfile(x_GPU, size, fPtr);

	//printf("\nMacierz Y: u * A^t\n ");
	//showMstrix(y, size);

	//printf("\nMacierz Y1 = a*b+u*a^t\n ");
	//showMstrix(y1, size);

	//printf("\nMacierz Y2: A * B + u * A^T + A \n ");
	//showMstrix(y2, size);

	//printf("\nMacierz Y3:- w *B \n ");
	//showMstrix(y3, size);

	//printf("\nMacierz Y4: A * B + u * A^T + A - w *B \n ");
	//showMstrix(y4, size);

	

	

	//printf("\nmacierz x:\n ");
	//showMstrix(x, size);


	//GPU_version(a, b, x, y, size,w, u) {

	//
	return 0;
}

void matrix_GPU(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int arraySize, int* x[maxSize][maxSize], int w, int u) {


	int *d_a;
	int *d_b;
	int *d_x;
	int *d_y;
	int *d_y1;
	int *d_y2;
	int *d_y3;
	int *d_y4;
	int *d_size;
	int matSize = maxSize * maxSize;
	int *d_w;
	int *d_u;

	int y[maxSize][maxSize];
	int y1[maxSize][maxSize];
	int y2[maxSize][maxSize];
	int y3[maxSize][maxSize];
	int y4[maxSize][maxSize];

	cudaMalloc((void**)&d_a, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_b, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_size, sizeof(int));
	cudaMalloc((void**)&d_w, sizeof(int));
	cudaMalloc((void**)&d_u, sizeof(int));
	cudaMalloc((void**)&d_x, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_y, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_y1, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_y2, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_y3, sizeof(int)*  matSize);
	cudaMalloc((void**)&d_y4, sizeof(int)*  matSize);

	cudaMemcpy(d_a, arrayA, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, arrayB, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_size, &arraySize, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, &u, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y3, y3, sizeof(int)*  matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y4, y4, sizeof(int)*  matSize, cudaMemcpyHostToDevice);

	int z = arraySize + 1 / 32;
	dim3 block(32, 32);
	dim3 grid(z, z);

	kernel << < grid, block >> > (d_a, d_b, d_size, d_x, d_y, d_w, d_u, d_y1, d_y2, d_y3, d_y4);

	//cudaMemcpy(arrayA, d_a, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(arrayB, d_b, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(&arraySize, d_size, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&w, d_w, sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(&u, d_u, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x, d_x, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(y, d_y, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(y1, d_y1, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(y2, d_y2, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(y3, d_y3, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(y4, d_y4, sizeof(int)*  matSize, cudaMemcpyDeviceToHost);



	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_size);
	cudaFree(d_w);
	cudaFree(d_u);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_y1);
	cudaFree(d_y2);
	cudaFree(d_y3);
	cudaFree(d_y4);
}

void txtfile(int* array[][maxSize], int arraySize, FILE * fPtr) {

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

void generateMatrix(int* array[][maxSize], int arraySize)
{

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {

			*array[i][j] = (rand() % (10 + 1 + 10) - 10);
		}
	}
}

void showMstrix(int* array[maxSize][maxSize], int arraySize) {


	for (int i = 0; i < arraySize; i++) {
		printf("[");
		for (int j = 0; j < arraySize; j++) {
			printf(" %d ", array[i][j]);
		}
		printf("]\n");
	}

}

void muliplyMatrix(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int* arrayX[maxSize][maxSize], int arraySize) {

	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			*arrayX[i][j] = 0;
			for (int k = 0; k < arraySize; k++) {
				*arrayX[i][j] += *arrayA[i][k] * *arrayB[k][j];
			}
		}
	}
}

void addMatrix(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int* arrayX[maxSize][maxSize], int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			*arrayX[i][j] = *arrayA[i][j] + *arrayB[i][j];
		}
	}
}

void transpositionMatrix(int* arrayA[maxSize][maxSize], int* arrayX[maxSize][maxSize], int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			*arrayX[i][j] = *arrayA[j][i];
		}
	}
}

void scalar_multiplicationMatrix(int* arrayA[maxSize][maxSize], int* arrayX[maxSize][maxSize], int arraySize, int x) {
	for (int i = 0; i < arraySize; i++) {
		for (int j = 0; j < arraySize; j++) {
			*arrayX[i][j] = x * *arrayA[i][j];
		}
	}
}

void matrix_CPU(int* arrayA[maxSize][maxSize], int* arrayB[maxSize][maxSize], int* x[maxSize][maxSize], int arraySize,
	int u, int w)
{

	int* y[maxSize][maxSize];

	// X = A *B
	muliplyMatrix(arrayA, arrayB, x, arraySize);

	// A^T
	transpositionMatrix(arrayA, y, arraySize);

	// u * a^t
	scalar_multiplicationMatrix(y, y, arraySize, u);

	// x = a * b + u * a^t
	addMatrix(x, y, x, arraySize);

	// x = a * b + u * a^t + a
	addMatrix(x, arrayA, x, arraySize);

	// w* b
	scalar_multiplicationMatrix(arrayB, y, arraySize, w);

	// x = a * b + u * a^t + a - w *b
	addMatrix(x, y, x, arraySize);
}

__device__ void kernel_add(int* arrayA, int* arrayB, int* arrayX)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;


	if (col < maxSize && row < maxSize) {
		arrayX[row * maxSize + col] = arrayA[row * maxSize + col] + arrayB[row * maxSize + col];
	}


}



__device__ void kernel_transpose(int* arrayA, int* arrayX)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;


	if (col < maxSize && row < maxSize) {
		arrayX[row * maxSize + col] = 0;
		arrayX[row * maxSize + col] = arrayA[col * maxSize + row];
	}

}

__device__ void kernel_scalar(int* arrayA, int* arrayX,  int* x)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < maxSize && row < maxSize) {
		arrayX[row * maxSize + col] = *x * arrayA[row * maxSize + col];
	}
}

__device__ void kernel_multiply(int* arrayA, int* arrayB, int* arrayX, int* arraySize)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	int sum = 0;

	if (col < maxSize && row < maxSize) {
		for (int i = 0; i < *arraySize; i++) {
			sum += arrayA[row * maxSize + i] * arrayB[i *maxSize + col];
		}

		arrayX[row * maxSize + col] = sum;
	}


}