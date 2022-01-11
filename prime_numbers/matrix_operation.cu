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


int main()
{
	const int arraySize = 50;
	srand(time(NULL));
	//int a[maxSize][maxSize];
	//int b[maxSize][maxSize];

	int size;
	printf("Podaj rozmiar macierzy : ");
	scanf("%d", &size);

	int a[maxSize][maxSize];
	int b[maxSize][maxSize];
	int x[maxSize][maxSize];
	int y[maxSize][maxSize];
	int u = 6;
	int w = -2;

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
	return 0;
}
