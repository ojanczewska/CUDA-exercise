
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void prime_CPU(unsigned long long int a);
cudaError_t prime_GPU(unsigned long long int a);

__global__ void kernel(unsigned long long int *d_a , bool *d_c)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x ;

   if (*d_a % i == 0) {
	   *d_c = false;
   }
}

int main()
{
	unsigned long long int a; 
	printf("Podaj liczbę do sprawdzenia:");

	scanf("%llu", &a);

	if (a == 1) {
		printf("Liczba nie jest pierwsza");
		return (0);
	}
	if (a == 2) {
		printf("Liczba jest pierwsza");
		return (0);
	}
	if (a % 2 == 0) {
		printf("Liczba nie jest pierwsza");
		return (0);
	}
	prime_CPU(a);

	cudaError_t cudaStatus = prime_GPU(a);


	return (0);

}

void prime_CPU(unsigned long long int a) {

	bool prime_cpu = true;
	for (long int i = 3; i <= sqrt(a); i += 2) {
		if (a % i == 0) {
			prime_cpu = false;
			break;
		}
	}

	if (prime_cpu == true) {
		printf("Podana liczba jest liczba pierwsza.");
	}
	else
	{
		printf("Podana liczba nie jest liczba pierwsza.");
	}
}

cudaError_t prime_GPU(unsigned long long int a) {

	unsigned long long int lim = sqrt(a);
	unsigned long long int *d_a = 0;
	//unsigned long long int *d_s = 0;
	bool c = true;
	bool *d_c;

	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_a, sizeof(unsigned long long int));
	//cudaStatus = cudaMalloc((void**)&d_s, sizeof(long long int));
	cudaStatus = cudaMalloc((void**)&d_c, sizeof(bool));


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_a, &a, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(d_s, s, sizeof(long long int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_c, &c, sizeof(bool), cudaMemcpyHostToDevice);
	//unsigned long long int x = lim + 1 / 32;

	//dim3 block(32);
	//dim3 grid(x);

	//kernel <<< grid , block >>>(d_a,d_c);

	kernel << <64000, 1024 >> > (d_a, d_c);

	cudaStatus = cudaMemcpy(&a,d_a, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(&c, d_c, sizeof(bool), cudaMemcpyDeviceToHost);

Error:
	cudaFree(d_a);
	//cudaFree(d_s);
	cudaFree(d_c);

	return cudaStatus;

}



