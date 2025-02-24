#include "lea_set.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		//if (abort) exit(code);
	}
}

void printLastCUDAError(){
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("-----\n");
		printf("ERROR: cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		printf("-----\n");
	}
}

__host__ u64* calculateRange() {
	u64* range;
	gpuErrorCheck(cudaMallocManaged(&range, 1 * sizeof(u64)));
	int threadCount = BLOCKS * THREADS;
	double keyRange = pow(2, TWO_POWER_RANGE);
	double threadRange = keyRange / threadCount;
	*range = ceil(threadRange);

	printf("Blocks                        : %d\n", BLOCKS);
	printf("Threads                       : %d\n", THREADS);
	printf("Total Thread count            : %d\n", threadCount);
	printf("Key Range (power)             : %d\n", TWO_POWER_RANGE);
	printf("Key Range (decimal)           : %.0f\n", keyRange);
	printf("Each Thread Key Range         : %.2f\n", threadRange);
	printf("Each Thread Key Range (kernel): %llu\n", range[0]);
	printf("Total encryptions             : %.0f\n", ceil(threadRange) * threadCount);
	printf("-------------------------------\n");
	
	return range;
}

void checkDeviceProperties() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads dim: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Shared memory per block: %lu\n", prop.sharedMemPerBlock);
}

// CUDA kernel function
