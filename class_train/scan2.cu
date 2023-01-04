// ONLY MODIFY THIS FILE

#include "scan2.h"
#include "gpuerrors.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!
__global__ void scan(float* x, float* y, float* z,int thread_num,int thread_len) {
	extern __shared__ float xs[];
	xs[tx] = x[tx + bx * thread_num*2];
	xs[tx + thread_num] = x[tx + thread_num + bx * thread_num*2];
	__syncthreads();


	for (int d = 0; d <= thread_len; d++) {
		if (tx < ((thread_num * 2)  / (1 << d + 1))) {

			xs[(1 << d + 1) * (tx + 1) - 1] = xs[(1 << d + 1) * tx + (1 << d) - 1] + xs[(1 << d + 1) * (tx + 1) - 1];
		}
		__syncthreads();
	}

	if (tx == 0) {
		y[thread_num * 2-1 + bx * thread_num * 2] = xs[thread_num * 2 - 1];
		z[bx] = xs[thread_num * 2 - 1];
		xs[thread_num * 2 - 1] = 0;
	}

	
	for (int d = thread_len; d >= 0; d--) {
		__syncthreads();
		if (tx < ((thread_num * 2) / (1 << d + 1))) {
			float ai = xs[(1 << d + 1) * (tx + 1) - 1];
			float bi = xs[(1 << d + 1) * tx + (1 << d) - 1] + xs[(1 << d + 1) * (tx + 1) - 1];
			xs[(1 << d + 1) * tx + (1 << d) - 1] = ai;
			xs[(1 << d + 1) * (tx + 1) - 1] = bi;
		}
	
	}
	__syncthreads();
	y[tx + bx * thread_num*2] = xs[tx + 1];
	y[tx + bx * thread_num*2 + thread_num - 1] = xs[tx + thread_num];
	

}
__global__ void scan_2(float* z,int block_num,int block_len) {
	extern __shared__ float zs[];

	zs[tx] = z[tx];
	zs[tx + block_num/2] = z[tx + block_num/2];
	__syncthreads();
	for (int d = 0; d <= block_len - 1; d++) {
		
		if (tx < (block_num / (1 << d + 1))) {

			zs[(1 << d + 1) * (tx + 1) - 1] = zs[(1 << d + 1) * tx + (1 << d) - 1] + zs[(1 << d + 1) * (tx + 1) - 1];
		}
		__syncthreads();
	}
	
	if (tx == 0) {
		zs[block_num -1] = 0;
	}
	__syncthreads();
	for (int d = block_len - 1; d >= 0; d--) {
		
		if (tx < (block_num / (1 << d + 1))) {
			float ai = zs[(1 << d + 1) * (tx + 1) - 1];
			float bi = zs[(1 << d + 1) * tx + (1 << d) - 1] + zs[(1 << d + 1) * (tx + 1) - 1];
			zs[(1 << d + 1) * tx + (1 << d) - 1] = ai;
			zs[(1 << d + 1) * (tx + 1) - 1] = bi;
		}
		__syncthreads();
	}
	
	z[tx] = zs[tx];
	z[tx + block_num/2] = zs[tx + block_num/2];
}
__global__ void add_batch(float* x, float* y, float offset,int thread_num) {
	y[tx + bx * thread_num*2] += x[bx] + offset;
	y[tx + thread_num + bx * thread_num*2] += x[bx] + offset;
}
void gpuKernel(float* x, float* y,int n) {
	int batch_size = 1024 * 1024;
		int	thread_num = 256;
		int	thread_len = 8;
		int block_num = 2048;
		int block_len = 11;
	
	float offset = 0;
	float* x_gpu;
	float* y_gpu;
	float* z_gpu;
	cudaMalloc((void**)&x_gpu, batch_size * sizeof(float));
	cudaMalloc((void**)&y_gpu, batch_size * sizeof(float));
	cudaMalloc((void**)&z_gpu, block_num * sizeof(float));
	
	for (int m = 0; m < n / batch_size; m++) {
	
		cudaMemcpy(x_gpu, x + m * batch_size, batch_size * sizeof(float), cudaMemcpyHostToDevice);
		
		dim3 blocks(thread_num, 1, 1);
		dim3 grids(block_num,1, 1);

	
		scan <<< grids, blocks, thread_num * 2*sizeof(float) >>> (x_gpu, y_gpu, z_gpu , thread_num, thread_len);
		scan_2 <<< 1, block_num /2, block_num*sizeof(float) >>> (z_gpu, block_num, block_len);
		add_batch <<< grids, blocks >>> (z_gpu, y_gpu, offset, thread_num);
		
		cudaMemcpy(y + m * batch_size, y_gpu, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
		offset = y[(m + 1) * batch_size - 1];

	}


		cudaFree(z_gpu);
		cudaFree(y_gpu);
		cudaFree(x_gpu);

}