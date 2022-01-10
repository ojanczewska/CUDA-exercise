#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

int main()
{

	const int arraySize = 5000;
	int a[arraySize][arraySize];
	int b[arraySize][arraySize];

	int size = 0;
	printf("Podaj rozmiar macierzy : ");
	scanf("%d", &size);


	return 0;
}
