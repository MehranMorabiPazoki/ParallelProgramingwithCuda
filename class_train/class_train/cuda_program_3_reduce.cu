#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include <math.h>



#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
#define N   1>>25
//#define m 1024
//__global__ void reduce(float* a, float* result,const int N) {
//	__shared__ float as[m];
//	as[ tx] = a[bx*1024+ tx];
//	as[tx + N / 4] = a[bx * 1024 +tx + N / 4];
//	as[tx + 2 * N / 4] = a[bx * 1024 +tx + 2 * N / 4];
//	as[tx + 3 * N / 4] = a[bx * 1024 + tx + 3*N/4];
//	__syncthreads();
//	for (int j = N / 4; j >= 1; j = j / 4) {
//		
//		if(tx<j){
//			as[tx] = as[tx] + as[tx + j]+ as[tx+ 2 * j]+ as[tx + 3 * j];
//		}
//		__syncthreads();
//	}
//	
//	if (tx == 0)
//		result[bx] = as[0];
//}
__global__ void scan(float* x,float* y,float* z){
	__shared__ float xs[2048];

	xs[tx] = x[tx+bx*2048];
	xs[tx + 1024] = x[tx + 1024+ bx * 2048];
	__syncthreads();
	for (int d = 0; d <= 11 - 1; d++) {
		if (tx < (2048 / (1 << d + 1))) {

			xs[(1 << d + 1) * (tx + 1) - 1] = xs[(1 << d + 1) * tx + (1 << d) - 1] + xs[(1 << d + 1) * (tx + 1) - 1];
		}
		__syncthreads();
	}
	if (tx == 0) {
		y[2047+ bx * 2048] = xs[2047];
		z[bx] = xs[2047];
		xs[2047] = 0;
	}
	__syncthreads();
	for (int d = 11 - 1; d >= 0; d--) {

		if (tx < (2048 / (1 << d + 1))) {
			float ai = xs[(1 << d + 1) * (tx + 1) - 1];
			float bi = xs[(1 << d + 1) * tx + (1 << d) - 1] + xs[(1 << d + 1) * (tx + 1) - 1];
			xs[(1 << d + 1) * tx + (1 << d) - 1] = ai;
			xs[(1 << d + 1) * (tx + 1) - 1] = bi;
		}
		__syncthreads();
	}

	y[tx+ bx * 2048 ] = xs[tx + 1];
	y[tx+ bx * 2048 + 1024 - 1] = xs[tx + 1024];
	
	


	/*xs[ tx] = x[ tx];
	xs[tx + 1024] = x[tx + 1024];
	__syncthreads();
	for (int d = 0; d <= 11-1; d++) {
		if(tx<(N/( 1 << d + 1))){

			xs[(1<<d+1) *(tx+1) - 1] = xs[(1 << d + 1) *tx+ (1 << d) -1] + xs[(1 << d + 1) * (tx + 1) - 1];
		}
		__syncthreads();
	}
	if (tx == 0) {
		y[2047] = xs[2047];
		xs[2047] = 0;
	}
	__syncthreads();
	for (int d = 11 - 1; d >= 0; d--) {
		
		if (tx < (N / (1 << d + 1))) {
			float ai= xs[(1 << d + 1) * (tx + 1) - 1];
			float bi = xs[(1 << d + 1) * tx + (1 << d) - 1] + xs[(1 << d + 1) * (tx + 1) - 1];
			xs[(1 << d + 1) * tx + (1 << d) - 1] = ai;
			xs[(1 << d + 1) * (tx + 1) - 1] = bi;
		}
		__syncthreads();
	}
	
	y[ tx] = xs[ tx+1];
	y[ tx + 1024-1] = xs[tx + 1024];*/

}
__global__ void scan_2(float* x) {
	for (int d = 0; d <= 9 - 1; d++) {
		if (tx < (512 / (1 << d + 1))) {

			x[(1 << d + 1) * (tx + 1) - 1] = x[(1 << d + 1) * tx + (1 << d) - 1] + x[(1 << d + 1) * (tx + 1) - 1];
		}
		__syncthreads();
	}
	if (tx == 0) {
		x[511] = 0;
	}
	__syncthreads();
	for (int d = 9 - 1; d >= 0; d--) {

		if (tx < (512 / (1 << d + 1))) {
			float ai = x[(1 << d + 1) * (tx + 1) - 1];
			float bi = x[(1 << d + 1) * tx + (1 << d) - 1] + x[(1 << d + 1) * (tx + 1) - 1];
			x[(1 << d + 1) * tx + (1 << d) - 1] = ai;
			x[(1 << d + 1) * (tx + 1) - 1] = bi;
		}
		__syncthreads();
	}

}
__global__ void add_batch(float* x, float* y,float offset) {
	y[tx + bx * 2048] += x[bx]+ offset;
	y[tx + 1024 + bx * 2048] += x[bx]+ offset;
}
void GPU_conf_scan(float* x, float* y, int size, double* gpu_kernel_time) {

	#if size<1<<24
	long long batch_size = 1 << 20;
	float offset = 0;
	float* x_gpu;
	float* y_gpu;
	float* z_gpu;
	cudaMalloc((void**)&x_gpu, batch_size * sizeof(float));
	cudaMalloc((void**)&y_gpu, batch_size * sizeof(float));
	cudaMalloc((void**)&z_gpu, 512 * sizeof(float));
	for (int m = 0; m < size / batch_size; m++) {
		
		cudaMemcpy(x_gpu, x+m* batch_size, batch_size * sizeof(float), cudaMemcpyHostToDevice);
		dim3 blocks(1024, 1, 1);
		dim3 grids(512, 1, 1);

		//gputimer timer1;
		//timer1.start();
		scan << < grids, blocks >> > (x_gpu, y_gpu, z_gpu);
		scan_2 << <1, 512 / 2 >> > (z_gpu);
		add_batch << < grids, blocks >> > (z_gpu, y_gpu, offset);
		//timer1.stop();
		//gpu_kernel_time = timer1.elapsed();
		cudaMemcpy(y+ m * batch_size, y_gpu, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
		offset = y[(m + 1) * batch_size - 1];

	}

	#endif
#if size >= 1<<24
	long long batch_size = 1 << 24;
	float offset = 0;
	float* x_gpu;
	float* y_gpu;
	float* z_gpu;
	cudaMalloc((void**)&x_gpu, batch_size * sizeof(float));
	cudaMalloc((void**)&y_gpu, batch_size * sizeof(float));
	cudaMalloc((void**)&z_gpu, 8192 * sizeof(float));
	for (int m = 0; m < size / batch_size; m++) {

		cudaMemcpy(x_gpu, x + m * batch_size, batch_size * sizeof(float), cudaMemcpyHostToDevice);
		dim3 blocks(1024, 1, 1);
		dim3 grids(8192, 1, 1);

		//gputimer timer1;
		//timer1.start();
		scan << < grids, blocks >> > (x_gpu, y_gpu, z_gpu);
		scan_2 << <1, 8192 / 2 >> > (z_gpu);
		add_batch << < grids, blocks >> > (z_gpu, y_gpu, offset);
		//timer1.stop();
		//gpu_kernel_time = timer1.elapsed();
		cudaMemcpy(y + m * batch_size, y_gpu, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
		offset = y[(m + 1) * batch_size - 1];
#endif // size >= 1<<24




	cudaFree(z_gpu);
	cudaFree(y_gpu);
	cudaFree(x_gpu);
}
/*void GPU_conf_reduce(float* a,float* result, int size,double* gpu_kernel_time) {
	float* a_gpu;
	float* result_gpu;
	float* fresult_gpu;
	cudaMalloc((void**)&a_gpu, sizeof(float) * 1024*1024);
	cudaMalloc((void**)&result_gpu, 1024*sizeof(float));
	cudaMalloc((void**)&fresult_gpu, sizeof(float));
	cudaMemcpy(a_gpu, a, sizeof(float) * 1024*1024,cudaMemcpyHostToDevice);
	
	dim3 block(1024/4, 1, 1);
	dim3 grid1(1024, 1, 1);
	dim3 grid2(1, 1, 1);
	GpuTimer timer;
	timer.Start();
	reduce <<<grid1, block >>> (a_gpu, result_gpu, 1024);
	reduce <<<grid2, block >>> (result_gpu, fresult_gpu,1024);
	timer.Stop();
	cudaMemcpy(result, fresult_gpu, sizeof(float), cudaMemcpyDeviceToHost);
	*gpu_kernel_time = timer.Elapsed();

	cudaFree(a_gpu);
	cudaFree(result_gpu);
	cudaFree(fresult_gpu);
}*/

void fill(float* a, int size) {
	for (int i = 0; i < size ; i++)
		a[i] = rand() % 4 - 2;

}
float cpu_kernel_reduce(float* a, int size) {
	float sum = 0;
	for (int i = 0; i < size; i++)
		sum += a[i];
	return sum;
}
void cpu_kernel_scan(float* x,float* y,int size) {
	/*y[0] =0;
	for (int i = 1; i < size; i++) {
		y[i]=y[i-1]+x[i-1];
	}*/
	y[0] = x[0];
	for (int i = 1; i < size; i++) {
		y[i] = y[i - 1] + x[i];
	}

	//for (int j = 0; j < size/2048; j++) {
	//	y[j * 2048] = x[j*2048];
	//	for (int i = 1; i < 2048; i++) {
	//		y[i+ j * 2048] = y[i+ j * 2048 - 1] + x[i+ j * 2048];
	//	}
	//}


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
int main() {
	struct cudaDeviceProp p;
	cudaGetDeviceProperties(&p, 0);
	printf("Device Name: %s\n", p.name);
//******************************** Reduce ************************************
	/*float* a;
a = (float*)malloc(1024*1024* sizeof(float));
fill(a, 1024*1024);

float Gpu_result = 0.0;
double gpu_kernel_time = 0.0;

clock_t t1 = clock();
GPU_conf_reduce(a, &Gpu_result, 1024, &gpu_kernel_time);
clock_t t2 = clock();

float cpu_result = cpu_kernel_reduce(a,1024*1024);

printf("n=%d GPU=%g ms GPU-Kernel=%g ms diff=%g\n",
	1024*1024, (t2 - t1) / 1000.0, gpu_kernel_time, cpu_result-Gpu_result);
free(a);*/
//******************************** Scan Code **************************************
	float* x;
	float* y_cpu;
	float* y_gpu;
	x = (float*)malloc(sizeof(float) * N);
	y_cpu = (float*)malloc(sizeof(float)* N);
	y_gpu = (float*)malloc(sizeof(float)* N);
	fill(x, N);
	cpu_kernel_scan(x, y_cpu, N);

	double gpu_kernel_time = 0.0;
	clock_t t1 = clock();
	GPU_conf_scan(x, y_gpu, N, &gpu_kernel_time);
	clock_t t2 = clock();
	double mse = calc_mse(y_cpu, y_gpu, N);

	printf("n=%d GPU=%g ms GPU-Kernel=%g ms Mse=%g\n",
	N, (t2 - t1) / 1000.0, gpu_kernel_time, mse);

	free(y_cpu);
	free(y_gpu);
	free(x);
	return 0;
}