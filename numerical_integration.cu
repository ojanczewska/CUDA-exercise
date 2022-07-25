#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <chrono>

using namespace std;

// Liczba punktów =  NUM_VALS = THREADS * BLOCKS
#define THREADS 16
#define BLOCKS  64
#define NUM_VALS THREADS*BLOCKS

struct Wspolrzedne {
	float* x;
	float* y;
};

void pole(Wspolrzedne punkty, int size) {
	float sum = 0;
	float* tab;
	tab = (float*)malloc(sizeof(float)*size);

	for (int i = 0; i < size - 1; i++) {
		tab[i] = (punkty.y[i] + punkty.y[i + 1]) * (punkty.x[i + 1] - punkty.x[i]) * 0.5;
		sum += tab[i];
	}

	printf("\nCPU POLE = %f  \n", sum);

}

void quick_sort(Wspolrzedne punkty, int left, int right) {

	int i, j;
	float pivot, y, y2;
	i = left;
	j = right;

	pivot = punkty.x[(i + j) / 2];

	while (i <= j) {
		while (punkty.x[i] < pivot && i < right) {
			i++;
		}
		while (punkty.x[j] > pivot && j > left) {
			j--;
		}

		if (i <= j) {
			y = punkty.x[i];
			y2 = punkty.y[i];
			punkty.x[i] = punkty.x[j];
			punkty.y[i] = punkty.y[j];
			punkty.x[j] = y;
			punkty.y[j] = y2;
			i++;
			j--;
		}
	}
	if (j > left) {
		quick_sort(punkty, left, j);
	}
	if (i < right) {
		quick_sort(punkty, i, right);
	}
}

__global__ void bitonic_sort_step(float *pointsX, float *pointsY, int j, int k)
{
	float temp, temp2;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int ixj = i ^ j;

	if ((ixj) > i) {
		if (((i&k) == 0) && (pointsX[i] > pointsX[ixj]) || ((i&k) != 0) && (pointsX[i] < pointsX[ixj])) {
			temp = pointsX[i];
			temp2 = pointsY[i];
			pointsX[i] = pointsX[ixj];
			pointsY[i] = pointsY[ixj];
			pointsX[ixj] = temp;
			pointsY[ixj] = temp2;
		}
	}
}




__global__ void kernel(float *pointsX, float *pointsY, int size, float* d_pol) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < NUM_VALS - 1) {
		d_pol[i] = (pointsY[i] + pointsY[i + 1])*(pointsX[i + 1] - pointsX[i]) *0.5;
	}

}

__global__ void kernel_sum(float* d_pol, int summations_number, bool parity) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < summations_number) {
		d_pol[index] += d_pol[summations_number + index];


	}
	if (!parity) {
		if (index == 0) {
			d_pol[index] += d_pol[summations_number * 2];
		}
	}

}




void GPU_liczenie(Wspolrzedne punkty, int size) {

	Wspolrzedne d_punkty;
	float* d_pol;
	float* tab_pol;
	tab_pol = (float*)malloc(sizeof(float)*size);

	cudaMalloc(&d_punkty.x, sizeof(float)*size);
	cudaMalloc(&d_punkty.y, sizeof(float)*size);
	cudaMalloc(&d_pol, sizeof(float)*size);

	cudaMemcpy(d_punkty.x, punkty.x, sizeof(float)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_punkty.y, punkty.y, sizeof(float)*size, cudaMemcpyHostToDevice);

	dim3 blocks(BLOCKS);
	dim3 grid(THREADS);


	//---------------SORTOWANIE ----------------------------
	int j, k;
	for (k = 2; k <= NUM_VALS; k <<= 1) {
		for (j = k >> 1; j > 0; j = j >> 1) {
			bitonic_sort_step << < blocks, grid >> > (d_punkty.x, d_punkty.y, j, k);
		}
	}

	// ---------------CALKOWANIE METODA TRAPEZOW ----------------------------

	kernel << < blocks, grid >> > (d_punkty.x, d_punkty.y, size, d_pol);

	// ---------------SUMOWANIE ----------------------------

	int summation_threads = size - 1;
	bool parity = false;
	int threads_per_block = THREADS;


	while (summation_threads > 1) {

		if (summation_threads % 2 == 1) {
			parity = false;
		}
		else parity = true;

		summation_threads /= 2;
		int blocks_number = (summation_threads + threads_per_block - 1) / threads_per_block;

		kernel_sum << < blocks_number, threads_per_block >> > (d_pol, summation_threads, parity);

	}

	cudaMemcpy(tab_pol, d_pol, sizeof(float)*size, cudaMemcpyDeviceToHost);

	cudaFree(d_punkty.x);
	cudaFree(d_punkty.y);


	printf("\nGPU  POLE = %f  \n", tab_pol[0]);

}



int main() {

	Wspolrzedne punkty_cpu, punkty_gpu;
	int size = NUM_VALS;

	printf("\nLiczba punktow =  %d \n", size);

	punkty_cpu.x = (float*)malloc(sizeof(float)*size);
	punkty_cpu.y = (float*)malloc(sizeof(float)*size);
	punkty_gpu.x = (float*)malloc(sizeof(float)*size);
	punkty_gpu.y = (float*)malloc(sizeof(float)*size);


	//-----------GENEROWNIE PUNKTOW ------------------
	srand(time(0));
	//printf("TABELA :\n ");
	for (int i = 0; i < size; i++) {
		punkty_cpu.x[i] = (double)rand() / RAND_MAX * 100000;
		punkty_gpu.x[i] = punkty_cpu.x[i];

		punkty_cpu.y[i] = (double)rand() / RAND_MAX * 100000;
		punkty_gpu.y[i] = punkty_cpu.y[i];

		//printf("%f  ", punkty_cpu.x[i]);
		//printf("%f  \n", punkty_cpu.y[i]);
	}

	//-----------LICZENIE CPU ------------------
	auto CPU1start = chrono::steady_clock::now();

	quick_sort(punkty_cpu, 0, size - 1);
	pole(punkty_cpu, size);


	auto CPU1end = chrono::steady_clock::now();
	chrono::duration<double> elapsedCPU1 = CPU1end - CPU1start;
	printf("CPU czas =  %f [ms].\n", elapsedCPU1.count());

	//-----------LICZENIE GPU ------------------
	auto CPUstart = chrono::steady_clock::now();
	GPU_liczenie(punkty_gpu, size);
	auto CPUend = chrono::steady_clock::now();

	chrono::duration<double> elapsedCPU = CPUend - CPUstart;
	printf("GPU czas =  %f [ms].\n", elapsedCPU.count());


	return 0;
}
