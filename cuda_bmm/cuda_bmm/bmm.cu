//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block 
#define TILEX 32
#define TILEY 16
// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX, TILEY);
	return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {
#if (TILEX < TILEY) 
	const int  p = TILEY / TILEX;;
	__shared__ float as[TILEY][TILEY];
	__shared__ float bs[TILEY][TILEX];
	int row = ty + TILEY * by;
	int col = tx + TILEX * bx;

	float temp = 0;

	for (int m = 0; m < n / TILEY; m++)
	{
		for (int i = 0; i < p; i++) {
		as[ty][tx+ i * TILEX] = ad[(row)*n + (m * TILEY + tx + (i * TILEX))];
		}
			bs[ty][tx] = bd[(m * TILEY + ty) * n + (col)];
		
		__syncthreads();
		for (int k = 0; k < TILEY; k++)
		{
			temp += as[ty][k] * bs[k][tx];
		}
		__syncthreads();
		
	}
	cd[(row)*n + (col)] = temp;



#endif

#if (TILEX >= TILEY) 
const int  p = TILEX / TILEY;;
	__shared__ float as[TILEY][TILEX];
	__shared__ float bs[TILEX][TILEX];
	int row = ty + TILEY * by;
	int col = tx + TILEX * bx;

	float temp = 0;

	for (int m = 0; m < n / TILEX; m++)
	{
		as[ty][tx] = ad[(row)*n + (m * TILEX + tx)];
		for (int i = 0; i < p; i++) {
			bs[ty + (i * TILEY)][tx] = bd[(m * TILEX + ty + (i * TILEY)) * n + (col)];
		}
		__syncthreads();
		for (int k = 0; k < TILEX; k++)
		{
			temp += as[ty][k] * bs[k][tx];
		}
		__syncthreads();
	}
	cd[(row)*n + (col)] = temp;

#endif
}

