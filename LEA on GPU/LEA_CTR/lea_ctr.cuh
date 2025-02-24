#ifndef LEA_CUDA_CUH
#define LEA_CUDA_CUH

#include "lea_set.cuh"

// CUDA kernel function declaration
__global__ void LEA_128_CTR_SharedMemory_TEST(u32* u_plain, u32* u_roundkey, u64* range);
__global__ void LEA_128_CTR_SharedMemory(u32* u_plain, u32* u_roundkey, u64* range);
__global__ void LEA_192_CTR_SharedMemory_TEST(u32* u_plain, u32* u_roundkey, u64* range);
__global__ void LEA_192_CTR_SharedMemory(u32* u_plain, u32* u_roundkey, u64* range);
__global__ void LEA_256_CTR_SharedMemory_TEST(u32* u_plain, u32* u_roundkey, u64* range);
__global__ void LEA_256_CTR_SharedMemory(u32* u_plain, u32* u_roundkey, u64* range);

// Host function to launch the CUDA kernel
void LEA128_EncryptBlk(u32 pbDst[LEA_BLK_WORD_LEN],
					   const u32 pbSrc[LEA_BLK_WORD_LEN],
					   const u32 pdRndKeys[LEA128_NUM_RNDS * LEA128_RNDKEY_WORD_LEN]);

void LEA128_Keyschedule(u32 pdRndKeys[LEA128_NUM_RNDS * LEA128_RNDKEY_WORD_LEN],
						const u32 pbKey[LEA128_KEY_WORD_LEN]);

void LEA192_EncryptBlk(u32 pbDst[LEA_BLK_WORD_LEN],
					   const u32 pbSrc[LEA_BLK_WORD_LEN],
					   const u32 pdRndKeys[LEA192_NUM_RNDS * LEA_RNDKEY_WORD_LEN]);

void LEA192_Keyschedule(u32 pdRndKeys[LEA192_NUM_RNDS * LEA_RNDKEY_WORD_LEN],
						const u32 pbKey[LEA192_KEY_WORD_LEN]);

void LEA256_EncryptBlk(u32 pbDst[LEA_BLK_WORD_LEN],
					   const u32 pbSrc[LEA_BLK_WORD_LEN],
					   const u32 pdRndKeys[LEA256_NUM_RNDS * LEA_RNDKEY_WORD_LEN]);

void LEA256_Keyschedule(u32 pdRndKeys[LEA256_NUM_RNDS * LEA_RNDKEY_WORD_LEN],
						const u32 pbKey[LEA256_KEY_WORD_LEN]);

void LEA_128_CTR_ShaerdMemory_main();
void LEA_192_CTR_ShaerdMemory_main();
void LEA_256_CTR_ShaerdMemory_main();

#endif // LEA_CUDA_CUH
