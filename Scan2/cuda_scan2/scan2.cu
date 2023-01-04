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
__global__ void scan(float* x, float* y, float* z) {
	extern __shared__ float xs[];
	xs[tx] = x[tx + bx *512];
	xs[tx + 256] = x[tx + 256 + bx * 512];
	__syncthreads();


	for (int d = 0; d <= 8; d++) {
		if (tx < ((512)  / (1 << d + 1))) {

			xs[(1 << d + 1) * (tx + 1) - 1] = xs[(1 << d + 1) * tx + (1 << d) - 1] + xs[(1 << d + 1) * (tx + 1) - 1];
		}
		__syncthreads();
	}

	if (tx == 0) {
		y[512-1 + bx * 512] = xs[512 - 1];
		z[bx] = xs[512 - 1];
		xs[512 - 1] = 0;
	}

	
	for (int d = 8; d >= 0; d--) {
		__syncthreads();
		if (tx < ((512) / (1 << d + 1))) {
			float ai = xs[(1 << d + 1) * (tx + 1) - 1];
			float bi = xs[(1 << d + 1) * tx + (1 << d) - 1] + xs[(1 << d + 1) * (tx + 1) - 1];
			xs[(1 << d + 1) * tx + (1 << d) - 1] = ai;
			xs[(1 << d + 1) * (tx + 1) - 1] = bi;
		}
	
	}
	__syncthreads();
	y[tx + bx * 512] = xs[tx + 1];
	y[tx + bx * 512 + 256 - 1] = xs[tx + 256];
	

}
__global__ void scan_2(float* z) {
	extern __shared__ float zs[];

	zs[tx] = z[tx];
	zs[tx + 1024] = z[tx + 1024];
	__syncthreads();
	for (int d = 0; d <= 11 - 1; d++) {
		
		if (tx < (2048 / (1 << d + 1))) {

			zs[(1 << d + 1) * (tx + 1) - 1] = zs[(1 << d + 1) * tx + (1 << d) - 1] + zs[(1 << d + 1) * (tx + 1) - 1];
		}
		__syncthreads();
	}
	
	if (tx == 0) {
		zs[2048 -1] = 0;
	}
	__syncthreads();
	for (int d = 11 - 1; d >= 0; d--) {
		
		if (tx < (2048 / (1 << d + 1))) {
			float ai = zs[(1 << d + 1) * (tx + 1) - 1];
			float bi = zs[(1 << d + 1) * tx + (1 << d) - 1] + zs[(1 << d + 1) * (tx + 1) - 1];
			zs[(1 << d + 1) * tx + (1 << d) - 1] = ai;
			zs[(1 << d + 1) * (tx + 1) - 1] = bi;
		}
		__syncthreads();
	}
	
	z[tx] = zs[tx];
	z[tx + 1024] = zs[tx + 1024];
}
__global__ void add_batch(float* x, float* y, float offset) {
	y[tx + bx * 512] += x[bx] + offset;
	y[tx + 256 + bx * 512] += x[bx] + offset;
}

void gpuKernel(float* x, float* y,int n) {
	const int batch_size = 1024 * 1024;
	const 	int	thread_num = 256;
//	const 	int	thread_len = 8;
	const 	int block_num = 2048;
//	const 	int block_len = 11;
	
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

	
		scan <<< grids, blocks, thread_num * 2*sizeof(float) >>> (x_gpu, y_gpu, z_gpu );
		scan_2 <<< 1, block_num /2, block_num*sizeof(float) >>> (z_gpu);
		add_batch <<< grids, blocks >>> (z_gpu, y_gpu, offset);
		
		cudaMemcpy(y + m * batch_size, y_gpu, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
		offset = y[(m + 1) * batch_size - 1];

	}


		cudaFree(z_gpu);
		cudaFree(y_gpu);
		cudaFree(x_gpu);

}