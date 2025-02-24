#include "lea_es.cuh"
#include <stdio.h>

/// @brief C version
void LEA128_Keyschedule(u32 pdRndKeys[LEA128_NUM_RNDS * LEA128_RNDKEY_WORD_LEN],
						const u32 pbKey[LEA128_KEY_WORD_LEN])
{
	u32 delta[4] = {0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec};
	u32 T[4] = {
		0x0,
	};

	T[0] = pbKey[0];
	T[1] = pbKey[1];
	T[2] = pbKey[2];
	T[3] = pbKey[3];

	for (int i = 0; i < LEA128_NUM_RNDS; i++)
	{
		T[0] = ROL(T[0] + ROL(delta[i & 3], i), 1);
		T[1] = ROL(T[1] + ROL(delta[i & 3], i + 1), 3);
		T[2] = ROL(T[2] + ROL(delta[i & 3], i + 2), 6);
		T[3] = ROL(T[3] + ROL(delta[i & 3], i + 3), 11);

		pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 0] = T[0];
		pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 1] = T[1];
		pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 2] = T[2];
		pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 3] = T[3];
		// pdRndKeys[i][3] = T[1];
		// pdRndKeys[i][4] = T[3];
		// pdRndKeys[i][5] = T[1];
	}
}

void LEA192_Keyschedule(u32 pdRndKeys[LEA192_NUM_RNDS * LEA_RNDKEY_WORD_LEN],
						const u32 pbKey[LEA192_KEY_WORD_LEN])
{
	u32 delta[6] = {0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 0x715ea49e, 0xc785da0a};
	u32 T[6] = {
		0x0,
	};

	T[0] = pbKey[0];
	T[1] = pbKey[1];
	T[2] = pbKey[2];
	T[3] = pbKey[3];
	T[4] = pbKey[4];
	T[5] = pbKey[5];

	for (int i = 0; i < LEA192_NUM_RNDS; i++)
	{
		T[0] = ROL(T[0] + ROL(delta[i % 6], i + 0), 1);
		T[1] = ROL(T[1] + ROL(delta[i % 6], i + 1), 3);
		T[2] = ROL(T[2] + ROL(delta[i % 6], i + 2), 6);
		T[3] = ROL(T[3] + ROL(delta[i % 6], i + 3), 11);
		T[4] = ROL(T[4] + ROL(delta[i % 6], i + 4), 13);
		T[5] = ROL(T[5] + ROL(delta[i % 6], i + 5), 17);

		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 0] = T[0];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 1] = T[1];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 2] = T[2];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 3] = T[3];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 4] = T[4];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 5] = T[5];
	}
}

void LEA256_Keyschedule(u32 pdRndKeys[LEA256_NUM_RNDS * LEA_RNDKEY_WORD_LEN],
						const u32 pbKey[LEA256_KEY_WORD_LEN])
{
	u32 delta[8] = {0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957};
	u32 T[8] = {
		0x0,
	};

	T[0] = pbKey[0];
	T[1] = pbKey[1];
	T[2] = pbKey[2];
	T[3] = pbKey[3];
	T[4] = pbKey[4];
	T[5] = pbKey[5];
	T[6] = pbKey[6];
	T[7] = pbKey[7];

	for (int i = 0; i < LEA256_NUM_RNDS; i++)
	{
		T[(6 * i) % 8] = ROL(T[(6 * i) % 8] + ROL(delta[i % 8], i), 1);
		T[(6 * i + 1) % 8] = ROL(T[(6 * i + 1) % 8] + ROL(delta[i % 8], i + 1), 3);
		T[(6 * i + 2) % 8] = ROL(T[(6 * i + 2) % 8] + ROL(delta[i % 8], i + 2), 6);
		T[(6 * i + 3) % 8] = ROL(T[(6 * i + 3) % 8] + ROL(delta[i % 8], i + 3), 11);
		T[(6 * i + 4) % 8] = ROL(T[(6 * i + 4) % 8] + ROL(delta[i % 8], i + 4), 13);
		T[(6 * i + 5) % 8] = ROL(T[(6 * i + 5) % 8] + ROL(delta[i % 8], i + 5), 17);

		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 0] = T[(6 * i) % 8];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 1] = T[(6 * i + 1) % 8];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 2] = T[(6 * i + 2) % 8];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 3] = T[(6 * i + 3) % 8];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 4] = T[(6 * i + 4) % 8];
		pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 5] = T[(6 * i + 5) % 8];
	}
}

void LEA128_EncryptBlk(u32 pbDst[LEA_BLK_WORD_LEN],
					   const u32 pbSrc[LEA_BLK_WORD_LEN],
					   const u32 pdRndKeys[LEA128_NUM_RNDS * LEA128_RNDKEY_WORD_LEN])
{
	u32 X0, X1, X2, X3;
	u32 temp;

	X0 = pbSrc[0];
	X1 = pbSrc[1];
	X2 = pbSrc[2];
	X3 = pbSrc[3];

	for (int i = 0; i < LEA128_NUM_RNDS; i++)
	{
		X3 = ROR((X2 ^ pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 3]) + (X3 ^ pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 1]), 3);
		X2 = ROR((X1 ^ pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 2]) + (X2 ^ pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 1]), 5);
		X1 = ROL((X0 ^ pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 0]) + (X1 ^ pdRndKeys[i * LEA128_RNDKEY_WORD_LEN + 1]), 9);
		temp = X0;
		X0 = X1;
		X1 = X2;
		X2 = X3;
		X3 = temp;
	}

	pbDst[0] = X0;
	pbDst[1] = X1;
	pbDst[2] = X2;
	pbDst[3] = X3;
}

void LEA192_EncryptBlk(u32 pbDst[LEA_BLK_WORD_LEN],
					   const u32 pbSrc[LEA_BLK_WORD_LEN],
					   const u32 pdRndKeys[LEA192_NUM_RNDS * LEA_RNDKEY_WORD_LEN])
{
	u32 X0, X1, X2, X3;
	u32 temp;

	X0 = pbSrc[0];
	X1 = pbSrc[1];
	X2 = pbSrc[2];
	X3 = pbSrc[3];

	for (int i = 0; i < LEA192_NUM_RNDS; i++)
	{
		X3 = ROR((X2 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 4]) + (X3 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 5]), 3);
		X2 = ROR((X1 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 2]) + (X2 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 3]), 5);
		X1 = ROL((X0 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 0]) + (X1 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 1]), 9);
		temp = X0;
		X0 = X1;
		X1 = X2;
		X2 = X3;
		X3 = temp;
	}

	pbDst[0] = X0;
	pbDst[1] = X1;
	pbDst[2] = X2;
	pbDst[3] = X3;
}

void LEA256_EncryptBlk(u32 pbDst[LEA_BLK_WORD_LEN],
					   const u32 pbSrc[LEA_BLK_WORD_LEN],
					   const u32 pdRndKeys[LEA256_NUM_RNDS * LEA_RNDKEY_WORD_LEN])
{
	u32 X0, X1, X2, X3;
	u32 temp;

	X0 = pbSrc[0];
	X1 = pbSrc[1];
	X2 = pbSrc[2];
	X3 = pbSrc[3];

	for (int i = 0; i < LEA256_NUM_RNDS; i++)
	{
		X3 = ROR((X2 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 4]) + (X3 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 5]), 3);
		X2 = ROR((X1 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 2]) + (X2 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 3]), 5);
		X1 = ROL((X0 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 0]) + (X1 ^ pdRndKeys[i * LEA_RNDKEY_WORD_LEN + 1]), 9);
		temp = X0;
		X0 = X1;
		X1 = X2;
		X2 = X3;
		X3 = temp;
	}

	pbDst[0] = X0;
	pbDst[1] = X1;
	pbDst[2] = X2;
	pbDst[3] = X3;
}

/////////////////////////////////////////

__device__ __forceinline__
	u32
	ROR_device(u32 input, u32 shift)
{
	u32 r;
	asm("{                  \n\t"
		"shf.r.wrap.b32 %0, %1, %2, %3; \n\t"
		"}"
		: "=r"(r)							   // 출력: 가상 레지스터 할당
		: "r"(input), "r"(input), "r"(shift)); // 입력: 가상 레지스터 할당
	return r;
}

__device__ __forceinline__
	u32
	ROL_device(u32 input, u32 shift)
{
	u32 r;
	asm("{                  \n\t"
		"shf.l.wrap.b32 %0, %1, %2, %3; \n\t"
		"}"
		: "=r"(r)							   // 출력: 가상 레지스터 할당
		: "r"(input), "r"(input), "r"(shift)); // 입력: 가상 레지스터 할당
	return r;
}

__global__ void LEA_128_ES_SharedMemory_TEST(u32 *u_plain, u32 *u_masterkey, u32 *u_delta, u64 *range)
{
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ u32 dtS[LEA_DELTA_WORD_LEN];
	if (threadIdx.x < LEA_DELTA_WORD_LEN)
	{
		dtS[threadIdx.x] = u_delta[threadIdx.x];
	}
	__syncthreads();

	u32 X0, X1, X2, X3;
	u32 T[4];
	u32 temp;

	X0 = u_plain[0];
	X1 = u_plain[1];
	X2 = u_plain[2];
	X3 = u_plain[3];

	if (threadIndex == 0)
	{
		printf("threadIndex : %llu\n", threadIndex);
		printf("Plaintext   : %08X %08X %08X %08X\n", X0, X1, X2, X3);
		printf("-------------------------------\n");
	}

	T[0] = u_masterkey[0];
	T[1] = u_masterkey[1];
	T[2] = u_masterkey[2];
	T[3] = u_masterkey[3];

	for (int i = 0; i < LEA128_NUM_RNDS; i++)
	{
		// key gen
		T[0] = ROL_device(T[0] + ROL_device(dtS[i & 3], i), 1);
		T[1] = ROL_device(T[1] + ROL_device(dtS[i & 3], i + 1), 3);
		T[2] = ROL_device(T[2] + ROL_device(dtS[i & 3], i + 2), 6);
		T[3] = ROL_device(T[3] + ROL_device(dtS[i & 3], i + 3), 11);

		// encryption
		X3 = ROR_device((X2 ^ T[3]) + (X3 ^ T[1]), 3);
		X2 = ROR_device((X1 ^ T[2]) + (X2 ^ T[1]), 5);
		X1 = ROL_device((X0 ^ T[0]) + (X1 ^ T[1]), 9);
		temp = X0;
		X0 = X1;
		X1 = X2;
		X2 = X3;
		X3 = temp;
		if (threadIndex == 0)
		{
			printf("Ciphertext  %d : %08X %08X %08X %08X\n", i, X0, X1, X2, X3);
		}
	}

	if (threadIndex == 0)
	{
		printf("threadIndex : %llu\n", threadIndex);
		printf("Ciphertext   : %08X %08X %08X %08X\n", X0, X1, X2, X3);
		printf("-------------------------------\n");
	}
}

__global__ void LEA_128_ES_SharedMemory(u32 *u_plain, u32 *u_cipher, u32 *u_masterkey, u32 *u_delta, u64 *range)
{
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ u32 dtS[LEA_DELTA_WORD_LEN];
	__shared__ u32 ctS[LEA_BLK_WORD_LEN];

	if (threadIdx.x < LEA_DELTA_WORD_LEN)
	{
		dtS[threadIdx.x] = u_delta[threadIdx.x];
		ctS[threadIdx.x] = u_cipher[threadIdx.x];
	}
	__syncthreads();

	u32 X0, X1, X2, X3;
	u32 T[4];
	u32 temp;

	// X0 = u_plain[0];
	// X1 = u_plain[1];
	// X2 = u_plain[2];
	// X3 = u_plain[3];

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = u_masterkey[0];
	rk1Init = u_masterkey[1];
	rk2Init = u_masterkey[2];
	rk3Init = u_masterkey[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = u_plain[0];
	pt1Init = u_plain[1];
	pt2Init = u_plain[2];
	pt3Init = u_plain[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	// if (threadIndex == 0) {
	// 	printf("threadIndex : %llu\n", threadIndex);
	// 	printf("Plaintext   : %08X %08X %08X %08X\n", X0, X1, X2, X3);
	// 	printf("-------------------------------\n");
	// }

	// T[0] = u_masterkey[0];
	// T[1] = u_masterkey[1];
	// T[2] = u_masterkey[2];
	// T[3] = u_masterkey[3];
	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++)
	{
		// u32 rk0, rk1, rk2, rk3;
		T[0] = rk0Init;
		T[1] = rk1Init;
		T[2] = rk2Init;
		T[3] = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		// u32 s0, s1, s2, s3;
		X0 = pt0Init;
		X1 = pt1Init;
		X2 = pt2Init;
		X3 = pt3Init;

		for (int i = 0; i < LEA128_NUM_RNDS; i++)
		{
			// key gen
			T[0] = ROL_device(T[0] + ROL_device(dtS[i & 3], i), 1);
			T[1] = ROL_device(T[1] + ROL_device(dtS[i & 3], i + 1), 3);
			T[2] = ROL_device(T[2] + ROL_device(dtS[i & 3], i + 2), 6);
			T[3] = ROL_device(T[3] + ROL_device(dtS[i & 3], i + 3), 11);

			// encryption
			X3 = ROR_device((X2 ^ T[3]) + (X3 ^ T[1]), 3);
			X2 = ROR_device((X1 ^ T[2]) + (X2 ^ T[1]), 5);
			X1 = ROL_device((X0 ^ T[0]) + (X1 ^ T[1]), 9);
			temp = X0;
			X0 = X1;
			X1 = X2;
			X2 = X3;
			X3 = temp;
			// if (threadIndex == 0) {
			// 	printf("Ciphertext  %d : %08X %08X %08X %08X\n", i, X0, X1, X2, X3);
			// }
		}

		if (X0 == ctS[0])
		{
			if (X1 == ctS[1])
			{
				if (X2 == ctS[2])
				{
					if (X3 == ctS[3])
					{
						printf("threadIndex : %llu\n", threadIndex);
						printf("threadRange : %llu\n", threadRange);
						printf("Ciphertext   : %08X %08X %08X %08X\n", X0, X1, X2, X3);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32)
		{
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
	// if (threadIndex == 0) {
	// 	printf("threadIndex : %llu\n", threadIndex);
	// 	printf("Ciphertext   : %08X %08X %08X %08X\n", X0, X1, X2, X3);
	// 	printf("-------------------------------\n");
	// }
}

void LEA_128_ES_ShaerdMemory_main()
{
	// master key
	// 0x3C2D1E0F, 0x78695A4B, 0xB4A59687, 0xF0E1D2C3
	// plaintext
	// 0x13121110, 0x17161514, 0x1B1A1918, 0x1F1E1D1C
	// ciphertext
	// 0x354EC89F, 0x18C6C628, 0xA7C73255, 0xFD8B6404

	u32 pdRndKeys[LEA128_NUM_RNDS * LEA128_RNDKEY_WORD_LEN];
	u32 masterKeys[LEA128_KEY_WORD_LEN] = {0x3C2D1E0F, 0x78695A4B, 0xB4A59687, 0xF0E1D2C3};
	u32 plaintext[LEA_BLK_WORD_LEN] = {0x13121110, 0x17161514, 0x1B1A1918, 0x1F1E1D1C};
	u32 delta[LEA_DELTA_WORD_LEN] = {0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec};
	u32 ciphertext[LEA_BLK_WORD_LEN] = {
		0,
	};

	// checkDeviceProperties();
	// C test
	LEA128_Keyschedule(pdRndKeys, masterKeys);
	LEA128_EncryptBlk(ciphertext, plaintext, pdRndKeys);
	printf("LEA128 ciphertext: 0x%08X, 0x%08X, 0x%08X, 0x%08X\n", ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]);

	// LEA192_Keyschedule(pdRndKeys,masterKeys);
	// LEA192_EncryptBlk(ciphertext,plaintext,pdRndKeys);
	// printf("LEA192 ciphertext: 0x%08X, 0x%08X, 0x%08X, 0x%08X\n", ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]);

	// LEA256_Keyschedule(pdRndKeys,masterKeys);
	// LEA256_EncryptBlk(ciphertext,plaintext,pdRndKeys);
	// printf("LEA256 ciphertext: 0x%08X, 0x%08X, 0x%08X, 0x%08X\n", ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]);

	// 데이터 크기:     (temporal) 16 bytes * 256 units = 4096 bytes
	// round key 크기:             16 bytes * 24 rounds= 384 bytes

	// size_t size = INPUT_DATA_BYTE_LEN;

	u32 *u_plain, *u_masterkey, *u_delta, *u_cipher;

	gpuErrorCheck(cudaMallocManaged(&u_plain, LEA_BLK_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_masterkey, LEA128_KEY_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_delta, LEA_DELTA_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_cipher, LEA_BLK_WORD_LEN * sizeof(u32)));

	for (int i = 0; i < LEA_BLK_WORD_LEN; i++)
	{
		u_plain[i] = plaintext[i];
	}
	for (int i = 0; i < (LEA128_KEY_WORD_LEN); i++)
	{
		u_masterkey[i] = masterKeys[i];
	}
	for (int i = 0; i < (LEA_DELTA_WORD_LEN); i++)
	{
		u_delta[i] = delta[i];
	}
	for (int i = 0; i < LEA_BLK_WORD_LEN; i++)
	{
		u_cipher[i] = ciphertext[i];
	}

	printf("-------------------------------\n");
	u64 *range = calculateRange();

	clock_t beginTime = clock();
	// LEA_128_CTR_SharedMemory_TEST<<<BLOCKS,THREADS>>>(u_plain, u_roundkey, range);
	// LEA_128_CTR_SharedMemory<<<BLOCKS,THREADS>>>(u_plain, u_roundkey, range);
	LEA_128_ES_SharedMemory_TEST<<<4, 4>>>(u_plain, u_masterkey, u_delta, range);
	// LEA_128_ES_SharedMemory<<<BLOCKS, THREADS>>>(u_plain, u_cipher, u_masterkey, u_delta, range);
	gpuErrorCheck(cudaDeviceSynchronize());
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();

	cudaFree(u_plain);
	cudaFree(u_masterkey);
	cudaFree(u_delta);
}

__global__ void LEA_192_ES_SharedMemory(u32 *u_plain, u32 *u_cipher, u32 *u_masterkey, u32 *u_delta, u64 *range)
{
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ u32 dtS[LEA_DELTA_WORD_LEN];
	__shared__ u32 ctS[LEA_BLK_WORD_LEN];

	if (threadIdx.x < LEA_DELTA_WORD_LEN)
	{
		if (threadIdx.x < LEA_BLK_WORD_LEN)
			ctS[threadIdx.x] = u_cipher[threadIdx.x];
		dtS[threadIdx.x] = u_delta[threadIdx.x];
	}
	__syncthreads();

	u32 X0, X1, X2, X3;
	u32 T[6];
	u32 temp;

	u32 rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init;
	rk0Init = u_masterkey[0];
	rk1Init = u_masterkey[1];
	rk2Init = u_masterkey[2];
	rk3Init = u_masterkey[3];
	rk4Init = u_masterkey[4];
	rk5Init = u_masterkey[5];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = u_plain[0];
	pt1Init = u_plain[1];
	pt2Init = u_plain[2];
	pt3Init = u_plain[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	// rk2Init = rk2Init + threadRangeStart / MAX_U32;
	// rk3Init = rk3Init + threadRangeStart % MAX_U32;
	rk4Init = rk4Init + threadRangeStart / MAX_U32;
	rk5Init = rk5Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++)
	{
		// u32 rk0, rk1, rk2, rk3;
		T[0] = rk0Init;
		T[1] = rk1Init;
		T[2] = rk2Init;
		T[3] = rk3Init;
		T[4] = rk4Init;
		T[5] = rk5Init;

		// Create plaintext as 32 bit unsigned integers
		// u32 s0, s1, s2, s3;
		X0 = pt0Init;
		X1 = pt1Init;
		X2 = pt2Init;
		X3 = pt3Init;

		for (int i = 0; i < LEA192_NUM_RNDS; i++)
		{
			// key gen
			T[0] = ROL_device(T[0] + ROL_device(dtS[i % 6], i + 0), 1);
			T[1] = ROL_device(T[1] + ROL_device(dtS[i % 6], i + 1), 3);
			T[2] = ROL_device(T[2] + ROL_device(dtS[i % 6], i + 2), 6);
			T[3] = ROL_device(T[3] + ROL_device(dtS[i % 6], i + 3), 11);
			T[4] = ROL_device(T[4] + ROL_device(dtS[i % 6], i + 4), 13);
			T[5] = ROL_device(T[5] + ROL_device(dtS[i % 6], i + 5), 17);

			// encryption
			X3 = ROR_device((X2 ^ T[4]) + (X3 ^ T[5]), 3);
			X2 = ROR_device((X1 ^ T[2]) + (X2 ^ T[3]), 5);
			X1 = ROL_device((X0 ^ T[0]) + (X1 ^ T[1]), 9);
			temp = X0;
			X0 = X1;
			X1 = X2;
			X2 = X3;
			X3 = temp;
		}

		if (X0 == ctS[0])
		{
			if (X1 == ctS[1])
			{
				if (X2 == ctS[2])
				{
					if (X3 == ctS[3])
					{
						printf("threadIndex : %llu\n", threadIndex);
						printf("threadRange : %llu\n", threadRange);
						printf("Ciphertext   : %08X %08X %08X %08X\n", X0, X1, X2, X3);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk5Init == MAX_U32)
		{
			rk4Init++;
		}
		// Create key as 32 bit unsigned integers
		rk5Init++;
	}
}

void LEA_192_ES_ShaerdMemory_main()
{
	// master key
	// 0x3C2D1E0F, 0x78695A4B, 0xB4A59687, 0xF0E1D2C3, 0xc3d2e1f0, 0x8796a5b4
	// plaintext
	// 0x23222120, 0x27262524, 0x2B2A2928, 0x2F2E2D2C
	// ciphertext
	// 0x325eb96f, 0x871bad5a, 0x35f5dc8c, 0xf2c67476

	u32 pdRndKeys[LEA192_NUM_RNDS * LEA_RNDKEY_WORD_LEN];
	u32 masterKeys[LEA192_KEY_WORD_LEN] = {0x3C2D1E0F, 0x78695A4B, 0xB4A59687, 0xF0E1D2C3, 0xc3d2e1f0, 0x8796a5b4};
	u32 plaintext[LEA_BLK_WORD_LEN] = {0x23222120, 0x27262524, 0x2B2A2928, 0x2F2E2D2C};
	u32 delta[LEA_DELTA_WORD_LEN] = {0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957};
	u32 ciphertext[LEA_BLK_WORD_LEN] = {
		0,
	};

	LEA192_Keyschedule(pdRndKeys, masterKeys);
	LEA192_EncryptBlk(ciphertext, plaintext, pdRndKeys);
	printf("LEA192 ciphertext: 0x%08X, 0x%08X, 0x%08X, 0x%08X\n", ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]);

	u32 *u_plain, *u_masterkey, *u_delta, *u_cipher;

	gpuErrorCheck(cudaMallocManaged(&u_plain, LEA_BLK_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_masterkey, LEA192_KEY_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_delta, LEA_DELTA_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_cipher, LEA_BLK_WORD_LEN * sizeof(u32)));

	for (int i = 0; i < LEA_BLK_WORD_LEN; i++)
	{
		u_plain[i] = plaintext[i];
	}
	for (int i = 0; i < (LEA192_KEY_WORD_LEN); i++)
	{
		u_masterkey[i] = masterKeys[i];
	}
	for (int i = 0; i < (LEA_DELTA_WORD_LEN); i++)
	{
		u_delta[i] = delta[i];
	}
	for (int i = 0; i < LEA_BLK_WORD_LEN; i++)
	{
		u_cipher[i] = ciphertext[i];
	}

	printf("-------------------------------\n");
	u64 *range = calculateRange();

	clock_t beginTime = clock();
	LEA_192_ES_SharedMemory<<<BLOCKS, THREADS>>>(u_plain, u_cipher, u_masterkey, u_delta, range);
	gpuErrorCheck(cudaDeviceSynchronize());
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();

	cudaFree(u_plain);
	cudaFree(u_masterkey);
	cudaFree(u_delta);
}


__global__ void LEA_256_ES_SharedMemory(u32 *u_plain, u32 *u_cipher, u32 *u_masterkey, u32 *u_delta, u64 *range)
{
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ u32 dtS[LEA_DELTA_WORD_LEN];
	__shared__ u32 ctS[LEA_BLK_WORD_LEN];

	if (threadIdx.x < LEA_DELTA_WORD_LEN)
	{
		if (threadIdx.x < LEA_BLK_WORD_LEN)
			ctS[threadIdx.x] = u_cipher[threadIdx.x];
		dtS[threadIdx.x] = u_delta[threadIdx.x];
	}
	__syncthreads();

	u32 X0, X1, X2, X3;
	u32 T[8];
	u32 temp;

	// X0 = u_plain[0];
	// X1 = u_plain[1];
	// X2 = u_plain[2];
	// X3 = u_plain[3];

	u32 rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init, rk6Init, rk7Init;
	rk0Init = u_masterkey[0];
	rk1Init = u_masterkey[1];
	rk2Init = u_masterkey[2];
	rk3Init = u_masterkey[3];
	rk4Init = u_masterkey[4];
	rk5Init = u_masterkey[5];
	rk6Init = u_masterkey[6];
	rk7Init = u_masterkey[7];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = u_plain[0];
	pt1Init = u_plain[1];
	pt2Init = u_plain[2];
	pt3Init = u_plain[3];

	u64 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	// rk2Init = rk2Init + threadRangeStart / MAX_U32;
	// rk3Init = rk3Init + threadRangeStart % MAX_U32;
	rk6Init = rk6Init + threadRangeStart / MAX_U32;
	rk7Init = rk7Init + threadRangeStart % MAX_U32;

	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++)
	{
		// u32 rk0, rk1, rk2, rk3;
		T[0] = rk0Init;
		T[1] = rk1Init;
		T[2] = rk2Init;
		T[3] = rk3Init;
		T[4] = rk4Init;
		T[5] = rk5Init;
		T[6] = rk6Init;
		T[7] = rk7Init;

		// Create plaintext as 32 bit unsigned integers
		// u32 s0, s1, s2, s3;
		X0 = pt0Init;
		X1 = pt1Init;
		X2 = pt2Init;
		X3 = pt3Init;

		for (int i = 0; i < LEA256_NUM_RNDS; i++)
		{
			// key gen
			T[(6 * i + 0) % 8] = ROL_device(T[(6 * i + 0) % 8] + ROL_device(dtS[i % 8], i + 0), 1);
			T[(6 * i + 1) % 8] = ROL_device(T[(6 * i + 1) % 8] + ROL_device(dtS[i % 8], i + 1), 3);
			T[(6 * i + 2) % 8] = ROL_device(T[(6 * i + 2) % 8] + ROL_device(dtS[i % 8], i + 2), 6);
			T[(6 * i + 3) % 8] = ROL_device(T[(6 * i + 3) % 8] + ROL_device(dtS[i % 8], i + 3), 11);
			T[(6 * i + 4) % 8] = ROL_device(T[(6 * i + 4) % 8] + ROL_device(dtS[i % 8], i + 4), 13);
			T[(6 * i + 5) % 8] = ROL_device(T[(6 * i + 5) % 8] + ROL_device(dtS[i % 8], i + 5), 17);

			// encryption
			X3 = ROR_device((X2 ^ T[(6 * i + 4) % 8]) + (X3 ^ T[(6 * i + 5) % 8]), 3);
			X2 = ROR_device((X1 ^ T[(6 * i + 2) % 8]) + (X2 ^ T[(6 * i + 3) % 8]), 5);
			X1 = ROL_device((X0 ^ T[(6 * i + 0) % 8]) + (X1 ^ T[(6 * i + 1) % 8]), 9);
			temp = X0;
			X0 = X1;
			X1 = X2;
			X2 = X3;
			X3 = temp;
		}

		if (X0 == ctS[0])
		{
			if (X1 == ctS[1])
			{
				if (X2 == ctS[2])
				{
					if (X3 == ctS[3])
					{
						printf("threadIndex : %llu\n", threadIndex);
						printf("threadRange : %llu\n", threadRange);
						printf("Ciphertext   : %08X %08X %08X %08X\n", X0, X1, X2, X3);
						printf("-------------------------------\n");
					}
				}
			}
		}

		// Overflow
		if (rk7Init == MAX_U32)
		{
			rk6Init++;
		}

		// Create key as 32 bit unsigned integers
		rk7Init++;
	}
}


void LEA_256_ES_ShaerdMemory_main()
{
	// master key
	// 0x3C2D1E0F, 0x78695A4B, 0xB4A59687, 0xF0E1D2C3, 0xc3d2e1f0, 0x8796a5b4, 0x4b5a6978, 0x0f1e2d3c
	// plaintext
	// 0x33323130, 0x37363534, 0x3B3A3938, 0x3F3E3D3C
	// ciphertext
	// 0xf6af51d6, 0xc189b147, 0xca00893a, 0x97e1f927

	u32 pdRndKeys[LEA256_NUM_RNDS * LEA_RNDKEY_WORD_LEN];
	u32 masterKeys[LEA256_KEY_WORD_LEN] = {0x3C2D1E0F, 0x78695A4B, 0xB4A59687, 0xF0E1D2C3, 0xc3d2e1f0, 0x8796a5b4, 0x4b5a6978, 0x0f1e2d3c};
	u32 plaintext[LEA_BLK_WORD_LEN] = {0x33323130, 0x37363534, 0x3B3A3938, 0x3F3E3D3C};
	u32 delta[LEA_DELTA_WORD_LEN] = {0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957};
	u32 ciphertext[LEA_BLK_WORD_LEN] = {
		0,
	};

	LEA256_Keyschedule(pdRndKeys, masterKeys);

	LEA256_EncryptBlk(ciphertext, plaintext, pdRndKeys);
	printf("LEA256 ciphertext: 0x%08X, 0x%08X, 0x%08X, 0x%08X\n", ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]);

	u32 *u_plain, *u_masterkey, *u_delta, *u_cipher;

	gpuErrorCheck(cudaMallocManaged(&u_plain, LEA_BLK_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_masterkey, LEA256_KEY_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_delta, LEA_DELTA_WORD_LEN * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&u_cipher, LEA_BLK_WORD_LEN * sizeof(u32)));

	for (int i = 0; i < LEA_BLK_WORD_LEN; i++)
	{
		u_plain[i] = plaintext[i];
	}
	for (int i = 0; i < (LEA256_KEY_WORD_LEN); i++)
	{
		u_masterkey[i] = masterKeys[i];
	}
	for (int i = 0; i < (LEA_DELTA_WORD_LEN); i++)
	{
		u_delta[i] = delta[i];
	}
	for (int i = 0; i < LEA_BLK_WORD_LEN; i++)
	{
		u_cipher[i] = ciphertext[i];
	}

	printf("-------------------------------\n");
	u64 *range = calculateRange();

	clock_t beginTime = clock();
	LEA_256_ES_SharedMemory<<<BLOCKS, THREADS>>>(u_plain, u_cipher, u_masterkey, u_delta, range);
	gpuErrorCheck(cudaDeviceSynchronize());
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();

	cudaFree(u_plain);
	cudaFree(u_masterkey);
	cudaFree(u_delta);
}
