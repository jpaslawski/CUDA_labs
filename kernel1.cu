
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void printDeviceNames() 
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) 
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Indeks urzadzenia: %d\n", i);
        printf("Nazwa urzadzenia: %s\n\n", prop.name);
    }
}

void printDevicePropertiesById(int id) 
{
    int nDevices, nProcs;
    cudaGetDeviceCount(&nDevices);
    if (id < nDevices) 
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, id);
        cudaDeviceGetAttribute(&nProcs, cudaDevAttrMultiProcessorCount, id);

        printf("Nazwa urzadzenia: %s\n", prop.name);
        printf("Ilosc multiprocesorow: %d.\n", nProcs);
        printf("Kompatybilnosc obliczeniowa: %d.%d.\n\n", prop.major, prop.minor);
    }
    else 
    {
        printf("Urzadzenie o indeksie %d nie istnieje...\n", id);
    }
}

void reportGPUMemory()
{
    size_t free, total;
    int freeMem, totalMem;
    cudaMemGetInfo(&free, &total);

    freeMem = static_cast<int>(free / 1048576);
    totalMem = static_cast<int>(total / 1048576);
    printf("---------- STAN PAMIECI ----------\n");
    printf("Wolna: %d MB\nCalkowita: %d MB\nUzywana: %d MB\n\n", freeMem, totalMem, totalMem - freeMem);
}

void allocAndFreeMem()
{
    reportGPUMemory();

    printf("Alokowanie 16MB danych typu char...\n\n");
    char* charData;
    cudaMallocManaged((void**)&charData, 1 << 24);
    reportGPUMemory();

    printf("Alokowanie 128MB danych typu float...\n\n");
    float* floatData;
    cudaMallocManaged((void**)&floatData, 1 << 27);
    reportGPUMemory();

    printf("Zwolnienie pamieci danych typu char...\n");
    cudaFree(charData);
    reportGPUMemory();

    printf("Zwolnienie pamieci danych typu float...\n");
    cudaFree(floatData);
    reportGPUMemory();
}

void hostAndDeviceTransfer(float *cpuTimeGlobal, float *gpuTimeGlobal, int nElements) 
{
    cudaEvent_t startHostTransfer, stopHostTransfer;
    cudaEvent_t startDevTransfer, stopDevTransfer;

    float cpuTime = 0, gpuTime = 0;

    int *deviceArray;
    int *hostArray = (int*)malloc(nElements * sizeof(int));
    cudaMalloc((int**)&deviceArray, nElements * sizeof(int));

    cudaEventCreate(&startDevTransfer);
    cudaEventCreate(&stopDevTransfer);

    cudaEventRecord(startDevTransfer);
    cudaError_t deviceToHost = cudaMemcpy(deviceArray, hostArray, nElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stopDevTransfer);
    cudaEventSynchronize(stopDevTransfer);
    if (deviceToHost != cudaSuccess) {
        printf("Nie udalo sie przekopiowac danych z GPU do CPU!\n");
    }
    else {
        printf("Kopiowanie z GPU do CPU powiodlo sie!\n");
        cudaEventElapsedTime(&gpuTime, startDevTransfer, stopDevTransfer);
        printf("Czas kopiowania danych: %fs\n\n", gpuTime);
        *gpuTimeGlobal = gpuTime;
    }

    cudaEventCreate(&startHostTransfer);
    cudaEventCreate(&stopHostTransfer);

    cudaEventRecord(startHostTransfer);
    cudaError_t hostToDevice = cudaMemcpy(hostArray, deviceArray, nElements * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopHostTransfer);
    cudaEventSynchronize(stopHostTransfer);
    if (hostToDevice != cudaSuccess) {
        printf("Nie udalo sie przekopiowac danych z CPU do GPU!\n");
    }
    else {
        printf("Kopiowanie z CPU do GPU powiodlo sie!\n");
        cudaEventElapsedTime(&cpuTime, startHostTransfer, stopHostTransfer);
        printf("Czas kopiowania danych: %fs\n\n", cpuTime);
        *cpuTimeGlobal = cpuTime;
    }
}

float computeAverageValue(float *array)
{
    float sum = 0.0;
    for (int i = 0; i < sizeof(array); i++)
    {
        sum += array[i];
    }

    return sum / sizeof(array);
}

int main()
{
    // Zadanie 1
    printDeviceNames();

    // Zadanie 2
    int id;
    printf("Wprowadz indeks urzadzenia, aby wyswietlic jego parametry: ");
    scanf("%d", &id);
    printDevicePropertiesById(id);

    // Zadanie 3
    allocAndFreeMem();

    // Zadanie 4, 5, 6
    int nElements = 1024 * 1024;
    float cpuTime[10] = { 0.0 }, gpuTime[10] = { 0.0 };
    for (int i = 0; i < 10; i++) {
        hostAndDeviceTransfer(&cpuTime[i], &gpuTime[i], nElements);
    }

    unsigned int bytes = nElements * sizeof(int);
    float avgCpySpeedCPU = (bytes * 1e-6) / computeAverageValue(cpuTime);
    float avgCpySpeedGPU = (bytes * 1e-6) / computeAverageValue(gpuTime);

    printf("\n-----------------------------------------------\n");
    printf("Rozmiar kopiowanych danych: %d MB\n", bytes / (1024 * 1024));
    printf("Srednia predkosc kopiowania dla CPU [GB/s]: %f\n", avgCpySpeedCPU);
    printf("Srednia predkosc kopiowania dla GPU [GB/s]: %f\n", avgCpySpeedGPU);

    
    return 0;
}
