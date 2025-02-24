
// header files for LEA
#include "stdio.h"
#include "stdlib.h"
#include <cuda_runtime.h>

// type definition
typedef unsigned long long	    u64;
typedef unsigned int            u32;
typedef unsigned char           u8;

// rotation definition
#define ROR(W,i) (((W)>>(i)) | ((W)<<(32-(i))))
#define ROL(W,i) (((W)<<(i)) | ((W)>>(32-(i))))

// type conversion
#define u32_in(x)            (*(u32*)(x))
#define u32_out(x, v)        {*((u32*)(x)) = (v);}


#define BLOCKS				1024
#define THREADS				512
#define TWO_POWER_RANGE		35

#define MAX_U32							4294967295
#define MAX_U16							0x0000FFFF

#define LEA128_NUM_RNDS		        24
#define LEA128_KEY_BYTE_LEN	        16
#define LEA128_KEY_WORD_LEN	        4
#define LEA128_RND_KEY_BYTE_LEN     LEA128_NUM_RNDS*LEA128_KEY_BYTE_LEN

#define LEA192_NUM_RNDS		        28
#define LEA192_KEY_BYTE_LEN	        24
#define LEA192_KEY_WORD_LEN	        6
#define LEA192_RND_KEY_BYTE_LEN     LEA192_NUM_RNDS*LEA192_KEY_BYTE_LEN

#define LEA256_NUM_RNDS		        32
#define LEA256_KEY_BYTE_LEN	        32
#define LEA256_KEY_WORD_LEN	        8
#define LEA256_RND_KEY_BYTE_LEN     LEA256_NUM_RNDS*LEA256_KEY_BYTE_LEN

#define LEA_BLK_BYTE_LEN	    16
#define LEA_BLK_WORD_LEN	    4
#define LEA_RNDKEY_WORD_LEN	    6
#define LEA128_RNDKEY_WORD_LEN	4
#define LEA192_RNDKEY_WORD_LEN	6
#define LEA256_RNDKEY_WORD_LEN	8
#define LEA_DELTA_WORD_LEN	    8

#define INPUT_DATA 256
#define INPUT_DATA_BYTE_LEN    INPUT_DATA*16

void printLastCUDAError();

__host__ u64* calculateRange();

void checkDeviceProperties();

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
