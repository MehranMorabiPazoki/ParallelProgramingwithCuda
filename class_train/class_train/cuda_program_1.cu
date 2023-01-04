#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>
#include <math.h>
#define N	14 
__global__ 
void add_vector(float* dist, float* sa, float* sb,int thread) {
	int i = blockIdx.x + blockIdx.y * blockDim.x + blockDim.x * blockDim.y * blockIdx.z;
	i= i*thread + threadIdx.x;
	dist[i] = sa[i] + sb[i];
}
void fill(float* a,int size) {
	for (int i = 0; i < size; i++)
		a[i] = rand() % 100;
}
void GPU_conf(float* a,float* b,float* c,int n,int thread) {
	float* a_gpu;
	float* b_gpu;
	float* c_gpu;
	//allocate memory on GPU
	cudaMalloc((void**)&a_gpu, n * sizeof(float));
	cudaMalloc((void**)&b_gpu, n * sizeof(float));
	cudaMalloc((void**)&c_gpu, n * sizeof(float));
	//transfer data on allocated memory
	cudaMemcpy(a_gpu, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, n * sizeof(float), cudaMemcpyHostToDevice);
	//grid and block setting
	dim3 grid(N/thread, 1, 1);
	dim3 block(thread, 1, 1);
	add_vector <<< grid, block >>> (c_gpu, a_gpu, b_gpu, thread);
	cudaMemcpy(c, c_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(c_gpu);
}
int  calc_thread(int n) {
	int i = 1;
	int r = pow(2, i);
	while (n % r != 0) {
		i = i + 1;
		r = pow(2, i);
	}
	return pow(2, i - 1);
}
int main(){
	float* a;
	float* b;
	float* c;
	b = new float[N];
	a = new float[N];
	c = new float[N];
	fill(a, N);
	fill(b, N);
	clock_t	t1 = clock();
	GPU_conf(a,b,c, N, calc_thread(N));
	clock_t t2 = clock();
	printf("done! calculation time is:%06lu ms\n", (t1 - t2)/1000 );
	for(int i=0;i<N;i++)
	printf("%f + %f =%f\n", a[i], b[i], c[i]);
}