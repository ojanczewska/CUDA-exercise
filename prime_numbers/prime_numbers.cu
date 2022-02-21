#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;


bool prime_CPU(unsigned long long int a);
bool prime_GPU(unsigned long long int a, bool gpu_prime);

__global__ void kernel(unsigned long long int* d_a, bool* d_c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
	i = 2 * i + 1;
	if (*d_a % i == 0) {
		*d_c = false;
	}
}
__global__ void  dummy() {

}

bool prime_OMP(unsigned long long int a) {
	bool prime = true;
	long long int pierwiastek = sqrt(a) + 2;

#pragma omp parallel for
	for (long long i = 3; i < pierwiastek; i = i + 2) {
		if (a % i == 0) {
			prime = false;
			break;
		}
	}



	return prime;


}


int main()
{
	//unsigned long long int a;
	//printf("Podaj liczbe do sprawdzenia:");
	//scanf("%llu", &a);
	bool gpu_prime = true;
	bool gpu_prime_wsp = true;
	dummy << <1, 1 >> > ();


	unsigned long long int liczby_testowe[6] = { 524287 ,2147483647  ,274876858369  ,1125897758834689 ,2305843009213693951 ,4611686014132420609 };


	for (int i = 0; i < 6; i++)
	{
		unsigned long long int proba = liczby_testowe[i];
		//unsigned long long int proba = a;
		printf("\nLiczba testowa: %llu \n", proba);
		if (proba == 1 || proba % 2 == 0) {
			printf("Liczba nie jest pierwsza");
			return (0);
		}
		if (proba == 2) {
			printf("Liczba jest pierwsza");
			return (0);
		}
		printf("\n-------------CPU ----------------\n");

		auto CPUstart = chrono::steady_clock::now();
		bool prime_cmp = prime_CPU(proba);
		auto CPUend = chrono::steady_clock::now();
		chrono::duration<double> elapsedCPU = CPUend - CPUstart;

		if (prime_cmp == true) {
			printf("Podana liczba jest liczba pierwsza.");
		}
		else
		{
			printf("Podana liczba jest zlozona.");
		}
		printf("\nCode executed in %f ms.\n", elapsedCPU.count());

		printf("\n-------------CPU OpenMP----------------\n");

		auto CMPstart = chrono::steady_clock::now();
		bool prime_cpu = prime_OMP(proba);
		auto CMPend = chrono::steady_clock::now();
		chrono::duration<double> elapsedCMP = CMPend - CMPstart;

		if (prime_cpu == true) {
			printf("Podana liczba jest liczba pierwsza.");
		}
		else
		{
			printf("Podana liczba jest zlozona.");
		}
		printf("\nCode executed in %f ms.\n", elapsedCMP.count());



		printf("\n-------------GPU ----------------\n");

		auto GPUstart = chrono::steady_clock::now();
		bool gpu3 = prime_GPU(proba, gpu_prime);
		auto GPUend = chrono::steady_clock::now();
		chrono::duration<double> elapsedGPU = GPUend - GPUstart;

		if (gpu3 == true) {
			printf("Podana liczba jest liczba pierwsza.");
		}
		else
		{
			printf("Podana liczba jest zlozona.");
		}

		printf("\nCode executed in %f ms.\n\n", elapsedGPU.count());


	}
	return (0);
}
bool prime_CPU(unsigned long long int a) {
	bool prime_cpu = true;
	for (long int i = 3; i <= sqrt(a); i += 2) {
		if (a % i == 0) {
			prime_cpu = false;
			break;
		}
	}
	return prime_cpu;
}


bool prime_GPU(unsigned long long int a, bool gpu_prime) {
	unsigned long long int lim = sqrt(a);
	unsigned long long int* d_a = 0;
	bool* d_c;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&d_a, sizeof(unsigned long long int));
	cudaStatus = cudaMalloc((void**)&d_c, sizeof(bool));

	cudaStatus = cudaMemcpy(d_a, &a, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_c, &gpu_prime, sizeof(bool), cudaMemcpyHostToDevice);
	long int x = (lim / 32 + 1) / 2;
	dim3 block(32);
	dim3 grid(x);
	kernel << <grid, block >> > (d_a, d_c);
	cudaStatus = cudaMemcpy(&a, d_a, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(&gpu_prime, d_c, sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
	return gpu_prime;
}



