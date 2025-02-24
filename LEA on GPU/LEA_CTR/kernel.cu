#include <iostream>
#include <cuda_runtime.h>

#include "lea_ctr.cuh"

int main(){
	printf("\nCUDA CTR LEA\n");
	printf("LEA 128 CTR Shared Memory\n");
	LEA_128_CTR_ShaerdMemory_main();

	printf("\nLEA 192 CTR Shared Memory\n");
	LEA_192_CTR_ShaerdMemory_main();

	printf("\nLEA 256 CTR Shared Memory\n");
	LEA_256_CTR_ShaerdMemory_main();

	return 0;
}
