// 1. Napisz program z wykorzystaniem CUDA ktory będzie realizował operacje dodawania dla wektorów zallokowanych na ostatnich zajęciach
// 1a) jeden wątek przetwarza 1 element tablicy
// 1b) jeden wątek przetwarza n elementów tablicy (gdzie n = 128)

// 2. w wektorze (tablicy) o zadanym rozmiarze znajdzie najmniejszy element i jego indeks
// 2a) wykonaj w dowolny sposób na GPU
// 2b) wykorzystaj metoda redukcji rownolegloj
// podpowiedz nalezy wykorzystac __syncthreads()
// caly kod bedzie uruchamiany w 1 bloku !!
// *2c) wykorzystaj pamiec wspoldzielona do przechowywania wartosci najmniejszych

// TYLKO DLA TYPU FLOAT
// 3. oblicz wynik mnozenia macierzy kwadratowych o zadanym rozmiarze
// int -> sqrt(1 MB / sizeof(int)) = sqrt(262144) ~= 512
// 3a) można kożystać z pamieci globalnej
// *3b) wykorzystaj pamiec wspoldzielona

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 32

void checkCudaSuccess(cudaError_t error, std::string message)
{
    if (error != cudaSuccess) {
        std::cout << message;
    }
}

// Compute sum of 2 vectors with 1 Thread per Operation
__global__ void addWith1TpO(float* c, const float* a, const float* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// Compute sum of 2 vectors with 1 Thread per N Operations
__global__ void addWith1TpNO(float* c, const float* a, const float* b, int N)
{
    int start = blockIdx.x * N;
    for(int i = start; i < start + N; i++)
        c[i] = a[i] + b[i];
}

void addWithCuda(float* c, const float* a, const float* b, unsigned int size)
{
    float* devA, * devB, * devC1, * devC2;
    int vectorSize = size * sizeof(float);
    float time1 = 0, timeN = 0;
    cudaEvent_t start, stop;

    // Zainicjowanie timerów
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Alokacja pamięci dla wektorów A, B i C na GPU
    checkCudaSuccess(cudaMalloc((void**)&devA, vectorSize), "devA memory allocation failed!\n");
    checkCudaSuccess(cudaMalloc((void**)&devB, vectorSize), "devB memory allocation failed!\n");
    checkCudaSuccess(cudaMalloc((void**)&devC1, vectorSize), "devC memory allocation failed!\n");
    checkCudaSuccess(cudaMalloc((void**)&devC2, vectorSize), "devC memory allocation failed!\n");

    // Kopiowanie zawartości wektorów A oraz B do pamięci GPU
    checkCudaSuccess(cudaMemcpy(devA, a, vectorSize, cudaMemcpyHostToDevice), "Failed to copy devA from CPU to GPU!\n");
    checkCudaSuccess(cudaMemcpy(devB, b, vectorSize, cudaMemcpyHostToDevice), "Failed to copy devB from CPU to GPU!\n");


    // Wykonanie funkcji - 1 wątek, 1 operacja i zmierzenie czasu
    int numThreadsPerBlock1 = 1024;
    int numBlock1 = size / numThreadsPerBlock1;
    cudaEventRecord(start);
    addWith1TpO <<<numBlock1, numThreadsPerBlock1 >>> (devC1, devA, devB);
    cudaEventRecord(stop);

    // Kopiowanie wektora wynikowego do pamięci CPU i obliczenie czasu
    checkCudaSuccess(cudaMemcpy(c, devC1, vectorSize, cudaMemcpyDeviceToHost), "Failed to copy result from GPU to CPU!\n");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);


    // Wykonanie funkcji - 1 wątek, N operacji i zmierzenie czasu
    int N = 128;
    int numBlock2 = size / N;
    cudaEventRecord(start);
    addWith1TpNO <<<numBlock2, 1 >>> (devC2, devA, devB, N);
    cudaEventRecord(stop);

    // Kopiowanie wektora wynikowego do pamięci CPU i obliczenie czasu
    float* c2 = (float*)malloc(1024 * 1024);
    checkCudaSuccess(cudaMemcpy(c2, devC2, vectorSize, cudaMemcpyDeviceToHost), "Failed to copy result from GPU to CPU!\n");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeN, start, stop);

    // Sprawdzenie czy wektory się zgadzają
    bool isCorrect = true;
    for (int i = 0; i < size; i++)
    {
        if (c[i] != c2[i])
        {
            isCorrect = false;
            break;
        }
    }

    // Wypisanie wyników
    if (isCorrect)
    {
        std::cout << "Vector addition completed! Results are matching." << std::endl;
        std::cout << "Elapsed Time for one thread per operation: " << time1 << "ms" << std::endl;
        std::cout << "Elapsed Time for one thread per 128 operations: " << timeN << "ms" << std::endl;
    }
    else {
        std::cout << "Vector addition failed!" << std::endl;
    }

    // Zwolnienie pamięci
    cudaFree(devC1);
    cudaFree(devC2);
    cudaFree(devA);
    cudaFree(devB);
    free(c);
    free(c2);
}

__global__ void findMinWithCudaOneThread(int* v, int* result, int size)
{
    result[0] = v[0];
    for (int i = 1; i < size; i++) {
        if (result[0] > v[i])
        {
            result[0] = v[i];
            result[1] = i;
        }
    }
}

__global__ void findMinWithCudaParallel(int*v, int* result)
{
    const int threadId = threadIdx.x;
    auto step = 1;
    int numOfThreads = blockDim.x / 2;

    __shared__ int sdata[256], index[256];
    sdata[threadId] = v[threadId];
    index[threadId] = threadId;

    while (numOfThreads > 0)
    {
        __syncthreads();
        if (threadId < numOfThreads)
        {
            const auto fst = threadId * step * 2;
            const auto snd = fst + step;
            
            if (sdata[fst] > sdata[snd])
            {
                sdata[fst] = sdata[snd];
                index[fst] = index[snd];
            }
        }

        step *= 2;
        numOfThreads /= 2;
    }
    __syncthreads();

    if (threadId == 0)
    {
        result[0] = sdata[0];
        result[1]= index[0];
    }
}

int findMinWithCPU(int* v, int size)
{
    int minimum = 0;
    for (int i = 1; i < size; i++)
    {
        if(v[0] > v[i]) 
        {
            v[0] = v[i];
            minimum = i;
        }

    }
    return minimum;
}

__global__ void multiplyMatricesWithCuda(float* matrixA, float* matrixB, float* matrixC, int size)
{
    int i, j;
    float temp = 0;

    __shared__ float sMatrixA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sMatrixB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int t = 0; t < gridDim.x; t++)
    {
        i = t * BLOCK_SIZE + threadIdx.y;
        j = t * BLOCK_SIZE + threadIdx.x;

        sMatrixA[threadIdx.y][threadIdx.x] = matrixA[row * size + j];
        sMatrixB[threadIdx.y][threadIdx.x] = matrixB[i * size + col];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            temp += sMatrixA[threadIdx.y][k] * sMatrixB[k][threadIdx.x];
        }

        __syncthreads();
    }
    matrixC[row * size + col] = temp;
}

void multiplyMatricesWithCPU(float* matrixA, float* matrixB, float* matrixC, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            float tmp = 0.0;
            for (int k = 0; k < size; k++)
            {
                tmp += matrixA[i * size + k] * matrixB[k * size + j];
            }
            matrixC[i * size + j] = tmp;
        }
    }
}

void fillVectorRandom(float* v, int size)
{
    for (int i = 0; i < size; i++)
    {
        v[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 50.0));
    }
}

void fillVectorRandom(int* v, int size)
{
    for (int i = 0; i < size; i++)
    {
        v[i] = rand() / (RAND_MAX / 5555);
    }
}

void printVector(int* v, int size)
{
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("%d\t", v[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

void fillMatricesRandom(float** matrixA, float** matrixB, int size) {
    // Alokacja pamięci o rozmiarze size*size
    size_t matrixSize = size * size * sizeof(float);
    *matrixA = (float*)malloc(matrixSize);
    *matrixB = (float*)malloc(matrixSize);

    memset(*matrixA, 0, matrixSize);
    memset(*matrixB, 0, matrixSize);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            int index = size * i + j;
            (*matrixA)[index] = 0.5;
            (*matrixB)[index] = 1.25;
        }
    }

}

int main()
{
    /****************************** Zadanie 1 **********************************/
    // Definicja wielkości wektorów - 1MB danych typu float
    const int arraySize = (1024 * 1024) / sizeof(float);

    // Alokacja pamięci dla wektorów na CPU
    float* a = (float*)malloc(arraySize * sizeof(float));
    float* b = (float*)malloc(arraySize * sizeof(float));
    float* c = (float*)malloc(arraySize * sizeof(float));

    // Wypełnienie wektorów losowymi wartościami
    fillVectorRandom(a, arraySize);
    fillVectorRandom(b, arraySize);

    // Wykonanie funkcji dodającej dwa wektory na dwa sposoby
    addWithCuda(c, a, b, arraySize);

    /****************************** Zadanie 2 **********************************/
    std::cout << std::endl << std::endl;
    // Tworzenie wektora wejściowego na CPU o wielkości 256 elementów
    int elementCount = 256;
    int vectorSize = sizeof(int) * elementCount;
    int resultSize = sizeof(int) * 2;
    int* inVector = (int*)malloc(vectorSize);
    fillVectorRandom(inVector, elementCount);

    // Kopiowanie wektora wejściowego do GPU
    int* devInVector;
    checkCudaSuccess(cudaMalloc((void**)&devInVector, vectorSize), "Failed to allocate device vector!\n");
    checkCudaSuccess(cudaMemcpy(devInVector, inVector, vectorSize, cudaMemcpyHostToDevice), "Failed to copy vector from CPU to GPU!\n");
    
    // Tworzenie wektora wynikowego na CPU oraz GPU dla dowolnej metody
    int* simpleResult = (int*)malloc(resultSize);
    int* devSimpleResult;
    checkCudaSuccess(cudaMalloc((void**)&devSimpleResult, resultSize), "Failed to allocate device result vector!\n");

    // Wywołanie funkcji znajdującej minimum za pomocą 1 wątku na GPU i zmierzenie czasu
    cudaEvent_t start, stop;
    float findMinTimeSimple = 0, findMinTimeParallel = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    findMinWithCudaOneThread << <1, 1 >> > (devInVector, devSimpleResult, elementCount);
    cudaEventRecord(stop);

    // Kopiowanie wektora wynikowego z GPU do CPU i obliczenie czasu
    checkCudaSuccess(cudaMemcpy(simpleResult, devSimpleResult, resultSize, cudaMemcpyDeviceToHost), "Failed to copy output vector from GPU to CPU!\n");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findMinTimeSimple, start, stop);

    // Tworzenie wektora wynikowego na CPU oraz GPU dla redukcji równoległej
    int* parallelResult = (int*)malloc(resultSize);
    int* devParallelResult;
    checkCudaSuccess(cudaMalloc((void**)&devParallelResult, resultSize), "Failed to allocate device result vector!\n");
    
    // Wywołanie funkcji wykorzystującej redukcję równoległą i zmierzenie czasu
    cudaEventRecord(start);
    findMinWithCudaParallel <<<1, elementCount >>> (devInVector, devParallelResult);
    cudaEventRecord(stop);

    // Kopiowanie wektora wynikowego z GPU do CPU i obliczenie czasu
    checkCudaSuccess(cudaMemcpy(parallelResult, devParallelResult, resultSize, cudaMemcpyDeviceToHost), "Failed to copy output vector from GPU to CPU!\n");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findMinTimeParallel, start, stop);

    // Wywołanie funkcji znajdującej minimum na CPU i zmierzenie czasu
    clock_t startCPU = clock();
    int minIndexCPU = findMinWithCPU(inVector, elementCount);
    clock_t stopCPU = clock();
    double diffTimeCPU = ((double) (stopCPU - startCPU)) * 1000 / CLOCKS_PER_SEC;

    // Wypisanie wyników
    //std::cout << "Min found on CPU: " << inVector[0] << "\t\t\t\tIndex: " << minIndexCPU << "\tTime: " << diffTimeCPU << "ms" << std::endl;
    std::cout << "Min found using 1 thread on GPU: " << simpleResult[0] <<  "\t\tIndex: " << simpleResult[1] << "\tTime: " << findMinTimeSimple << "ms" << std::endl;
    std::cout << "Min found using parallel reduction on GPU: " << parallelResult[0] << "\tIndex: " << parallelResult[1] << "\tTime: " << findMinTimeParallel << "ms" << std::endl;

    /****************************** Zadanie 3 **********************************/
    std::cout << std::endl << std::endl;
    // Alokacja i inicjalizacja macierzy na CPU
    const int N = 512;
    size_t matrixSize = N * N * sizeof(float);
    float* matrixA, * matrixB, * matrixC, * matrixCPU;
    fillMatricesRandom(&matrixA, &matrixB, N);
    matrixC = (float*)malloc(matrixSize);
    matrixCPU = (float*)malloc(matrixSize);

    // Alokacja pamięci dla macierzy A, B i C na GPU
    float *devMatrixA, *devMatrixB, *devMatrixC;
    checkCudaSuccess(cudaMalloc((void**)&devMatrixA, matrixSize), "Failed to allocate matrixA!\n");
    checkCudaSuccess(cudaMalloc((void**)&devMatrixB, matrixSize), "Failed to allocate matrixB!\n");
    checkCudaSuccess(cudaMalloc((void**)&devMatrixC, matrixSize), "Failed to allocate matrixC!\n");

    // Określenie wielkości bloku oraz ilości wątków na block
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(N / BLOCK_SIZE, N / BLOCK_SIZE);
    float matrixMultTimeGPU = 0;
    
    // Inicjalizacja timerów
    cudaEvent_t startGPU3, stopGPU3;
    cudaEventCreate(&startGPU3);
    cudaEventCreate(&stopGPU3);

    // Kopiowanie macierzy z CPU do GPU
    checkCudaSuccess(cudaMemcpy(devMatrixA, matrixA, matrixSize, cudaMemcpyHostToDevice), "Failed to copy matrixA from CPU to GPU!\n");
    checkCudaSuccess(cudaMemcpy(devMatrixB, matrixB, matrixSize, cudaMemcpyHostToDevice), "Failed to copy matrixB from CPU to GPU!\n");

    // Wywołanie funkcji mnożącej macierze na GPU i zmierzenie czasu
    cudaEventRecord(startGPU3);
    multiplyMatricesWithCuda<<< gridDim, blockDim >>>(devMatrixA, devMatrixB, devMatrixC, N);
    cudaEventRecord(stopGPU3);

    // Kopiowanie macierzy wynikowej z GPU do CPU i obliczenie czasu
    checkCudaSuccess(cudaMemcpy(matrixC, devMatrixC, matrixSize, cudaMemcpyDeviceToHost), "Failed to copy matrixC from GPU to CPU!\n");;
    cudaEventSynchronize(stopGPU3);
    cudaEventElapsedTime(&matrixMultTimeGPU, startGPU3, stopGPU3);


    // Wywołanie funkcji mnożącej macierze na CPU
    clock_t startMultCPU = clock();
    multiplyMatricesWithCPU(matrixA, matrixB, matrixCPU, N);
    clock_t stopMultCPU = clock();
    double diffMultTimeCPU = ((double)(stopMultCPU - startMultCPU)) * 1000 / CLOCKS_PER_SEC;

    // Sprawdzenie wyników
    bool isCorrect = true;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (matrixC[i * N + j] != matrixCPU[i * N + j])
            {
                isCorrect = false;
                break;
            }
        }
    }
    std::cout << "Matrix multiplication finished! ";
    if (isCorrect)
    {
        std::cout << "GPU and CPU results are matching." << std::endl;
        std::cout << "Time on GPU: " << matrixMultTimeGPU << "ms" << std::endl;
        std::cout << "Time on CPU: " << diffMultTimeCPU << "ms" << std::endl;
    }
    else std::cout << "However GPU and CPU results are not matching!" << std::endl;

    return 0;
}