#ifndef CUDACC
#define CUDACC
#endif
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>
#include <math.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
#define mem2d(data,q,y,x)   data[((y)<<(q))+(x)]
#define TileY 32
#define TileX 32



double calc_mse(float* data1, float* data2, int size);
void cpuKernel_yx(const float* const a, const float* const b, float* c, const int m, const int n, const int y, const int x);
void cpuKernel(const float* const a, const float* const b, float* c, const int m, const int n);
__global__ void mat_mul(float* a, float* b, float* c, int n) {
	__shared__ float as[TileY][TileY];
	__shared__ float bs[TileY][TileY];
	int row = ty + TileY * by;
	int col = tx + TileX * bx ;
	float temp = 0;
	
		
	for (int p = 0; p < TileX / TileY; p++)
		{
		for (int m = 0; m < n / TileY; m++)
			{
			as[ty][tx] = a[(row)*n + (m * TileY + threadIdx.x)];
			bs[ty][tx] = b[(m * TileY + ty) * n + (col+p*TileY)];
			__syncthreads();
				for (int k = 0; k < TileY; k++) 
					{
					temp += as[ty][k] * bs[k][tx];
					}
			__syncthreads();
			}
		c[(row)*n + (col + p * TileY)] = temp;
		temp = 0;
		}	
}

void GPU_conf(float* a, float* b, float* c, int n) {
	float* a_gpu;
	float* b_gpu;
	float* c_gpu;

	//allocate memory on GPU
	
		cudaMalloc((void**)&a_gpu, n* n * sizeof(float));
		cudaMalloc((void**)&b_gpu, n* n * sizeof(float));
		cudaMalloc((void**)&c_gpu, n* n * sizeof(float));
	

	//transfer data on allocated memory
	
		cudaMemcpy(a_gpu, a, n*n * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_gpu, b, n*n * sizeof(float), cudaMemcpyHostToDevice);
	

	//grid and block setting
	dim3 grid(n/ TileX,n/ TileY, 1);
	dim3 block(TileY, TileY,1);
	mat_mul <<< grid, block >>> (a_gpu, b_gpu, c_gpu, n);
	
	cudaMemcpy(c, c_gpu, n * n* sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(c_gpu);
}
void fill(float* a, int size) {
	for (int i = 0; i < size*size; i++)
			a[i] = rand() % 10-5;
		
}
void disp(float* a,int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			printf("%f\t", a[i*n+j]);
		printf("\n");
	}
	printf("\n \n");
}
int main() {
	const int m = 6;
	const int N = pow(2,m);
	double mse;
	float* a = new float [N * N];
	float* b = new float [N * N];
	float* c = new float [N * N];
	float* c_serial = new float[N * N];
	fill(a, N);
	fill(b, N);
	GPU_conf(a, b, c, N);
	cpuKernel(a, b, c_serial,m , N);
	mse=calc_mse(c, c_serial, N);
	printf("MSE value =%f", mse);
//	disp(a, N);
//  disp(b, N);
//	disp(c, N);

	return 0;
}

double calc_mse(float* data1, float* data2, int size) {
	double mse = 0.0;
	int i; for (i = 0; i < size; i++) {
		double e = data1[i] - data2[i];
		e = e * e;
		mse += e;
	}
	return mse;
}
void cpuKernel_yx(const float* const a, const float* const b, float* c, const int m, const int n,
	const int y, const int x) { // one element: y,x
	mem2d(c, m, y, x) = 0.0f;
	for (int k = 0; k < n; k++) {
		mem2d(c, m, y, x) += mem2d(a, m, y, k) * mem2d(b, m, k, x);
	}
}
void cpuKernel(const float* const a, const float* const b, float* c, const int m, const int n) { // entire matrix
	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++) {
			cpuKernel_yx(a, b, c, m, n, y, x);
		}
}