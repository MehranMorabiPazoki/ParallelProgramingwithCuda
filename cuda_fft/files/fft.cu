//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z


int N1;
 //you may define other parameters here!

 __constant__	float W[1024];
// __constant__	float Wi_1024[512];




const int TILE_DIM = 8;
const int BLOCK_ROWS = 2;
// you may define other macros here!
// you may define other functions here!
__device__ int reverse(int N,int M, int n) {
	int j, p = 0;
	for (j = 1; j <= M; j++) {
		if (n & (1 << (M - j)))
			p |= 1 << (j - 1);
	}
	return p;
}
__global__ void bit_reverse(float* x, float* temp, int N,int M) {
	int i;
	i = bx*1024+tx;
	temp [i]= x[reverse(N,M,i)];
}
__global__ void transfer(float* x, float* temp) {

	__shared__ float s[TILE_DIM];
	int i =bz*gridDim.y*gridDim.x*blockDim.x+ by * gridDim.x * blockDim.x + blockDim.x * bx + tx;
	s[tx] = temp[i];
	x[i] = s[tx];
	
}
__global__ void gpu_weight(float* w_r, float* w_i, int N) {
	int i = bx * 512 + tx;
	w_r[i] = cos(2 * PI * i / N);
	w_i[i] = -sin(2 * PI * i / N);
}
__global__ void transposeNoBankConflicts(float* odata, const float* idata)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}
__global__ void copySharedMem(float* odata, const float* idata)
{
	__shared__ float tile[TILE_DIM * TILE_DIM];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];

	__syncthreads();

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}
__global__ void copy(float* odata, const float* idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y + j) * width + x] = idata[(y + j) * width + x];
}
__global__ void transposeNaive(float* odata, const float* idata)
{
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[x * width + (y + j)] = idata[(y + j) * width + x];
}
__global__ void nonSquredTranspose(float* in, float* out) {
	__shared__ float s[TILE_DIM];
	int i = bz * gridDim.y * gridDim.x * blockDim.x+by * gridDim.x* blockDim.x + blockDim.x*bx+tx;
	int j = bz * gridDim.y * gridDim.x * blockDim.x+ by + (blockDim.x * bx + tx) * gridDim.y ;

	s[tx] = in[i];
	out[j] = s[tx];
}
__global__ void fft_rad2_v1(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N, int a, int n) {
	int i = (bx) * 1024 + tx;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	if (!(n & i)) {
		temp_xs1_r = x_r_d[i];
		temp_xs1_i = x_i_d[i];
		__syncthreads();
		temp_xs2_r = x_r_d[i + n] * w_r_d[(i * a) % (n * a)] - x_i_d[i + n] * w_i_d[(i * a) % (n * a)];
		temp_xs2_i = x_i_d[i + n] * w_r_d[(i * a) % (n * a)] + x_r_d[i + n] * w_i_d[(i * a) % (n * a)];

		x_r_d[i] = temp_xs2_r + temp_xs1_r;
		x_i_d[i] = temp_xs2_i + temp_xs1_i;
		x_r_d[i + n] = temp_xs1_r - temp_xs2_r;
		x_i_d[i + n] = temp_xs1_i - temp_xs2_i;
	}
}
__global__ void fft_rad2_v2(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N, int m, int k) {
	__shared__ float x_r[1024];
	__shared__ float x_i[1024];
	__shared__ float w_r[512];
	__shared__ float w_i[512];
	float Wr;
	float Wi;
	int i = bx * 1024 + tx, j;
	x_r[i] = x_r_d[i];
	x_i[i] = x_i_d[i];
	if (i < 512) {
		w_r[i] = w_r_d[k * i];
		w_i[i] = w_i_d[k * i];
	}
	__syncthreads();
	int a = N / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * w_r[(tx * a) % (n * a)] - x_i[tx + n] * w_i[(tx * a) % (n * a)];
			temp_xs2_i = x_i[tx + n] * w_r[(tx * a) % (n * a)] + x_r[tx + n] * w_i[(tx * a) % (n * a)];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	x_r_d[i] = x_r[tx];
	x_i_d[i] = x_i[tx];
}
/********************************  second version   *******************************************/
__global__ void fft_rad2_v3(float* x_r_d, float* x_i_d,float* w_r_d,float* w_i_d, int N1,int m1,int N,int m) {
	__shared__ float x_r[1024];
	__shared__ float x_i[1024];
	__shared__ float w_r[512];
	__shared__ float w_i[512];
	float Wr;
	float Wi;
	
	/*int i = tx*1024 + bx, j;*/
	int i = tx  + bx * 1024, j;
	x_r[reverse(N1, m1, tx)] = x_r_d[i];
	x_i[reverse(N1, m1, tx)] = x_i_d[i];
	if (tx < 512) {
		w_r[tx] = w_r_d[tx];
		w_i[tx] = w_i_d[tx];
	}
	__syncthreads();
	int a = N1 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m1; j++) {
		if (!(n & tx)) {
			
			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];
			
			temp_xs2_r = x_r[tx + n] * w_r[(tx * a) % (n * a)] - x_i[tx + n] * w_i[(tx * a) % (n * a)];
			temp_xs2_i = x_i[tx + n] * w_r[(tx * a) % (n * a)] + x_r[tx + n] * w_i[(tx * a) % (n * a)];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	Wr = cos(2 * PI * tx * bx /N);
	Wi = -sin(2 * PI * tx * bx / N);
	  x_r_d[i]= x_r[tx]*Wr- x_i[tx]*Wi;
	  x_i_d[i] =x_i[tx]*Wr+ x_r[tx]*Wi;
}
__global__ void fft_rad2_v4(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N2, int m2) {
	 __shared__ float x_r[1024];
	 __shared__ float x_i[1024];
	 __shared__ float w_r[512];
	 __shared__ float w_i[512];
	

//	int i = tx + bx* 1024, j;
	 int i = tx * 1024 + bx, j;
	x_r[reverse(N2, m2, tx)] = x_r_d[i];
	x_i[reverse(N2, m2, tx)] = x_i_d[i];
	if (tx < 512) {
		w_r[tx] = w_r_d[tx];
		w_i[tx] = w_i_d[tx];
	}
	__syncthreads();
	int a = N2 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m2; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * w_r[(tx * a) % (n * a)] - x_i[tx + n] * w_i[(tx * a) % (n * a)];
			temp_xs2_i = x_i[tx + n] * w_r[(tx * a) % (n * a)] + x_r[tx + n] * w_i[(tx * a) % (n * a)];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	
	x_r_d[i] = x_r[tx] ;
	x_i_d[i] = x_i[tx] ;
}
__global__ void fft_rad2_v5(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N1, int m1, int N, int m) {
	__shared__ float x_r[1024];
	__shared__ float x_i[1024];
	__shared__ float w_r[512];
	__shared__ float w_i[512];

	int i = tx * gridDim.x + bx, j;

	x_r[reverse(N1, m1, tx)] = x_r_d[i];
	x_i[reverse(N1, m1, tx)] = x_i_d[i];
	if (tx < 512) {
		w_r[tx] = w_r_d[tx];
		w_i[tx] = w_i_d[tx];
	}
	__syncthreads();
	int a = N1 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m1; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * w_r[(tx * a) % (n * a)] - x_i[tx + n] * w_i[(tx * a) % (n * a)];
			temp_xs2_i = x_i[tx + n] * w_r[(tx * a) % (n * a)] + x_r[tx + n] * w_i[(tx * a) % (n * a)];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	//Wr = cos(2 * PI * tx * bx / N);
	//Wi = -sin(2 * PI * tx * bx / N);
	//x_r_d[i] = x_r[tx] * Wr - x_i[tx] * Wi;
	//x_i_d[i] = x_i[tx] * Wr + x_r[tx] * Wi;
	x_r_d[i] = x_r[tx];
	x_i_d[i] = x_i[tx];
}
__global__ void fft_rad2_v8(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N1, int m1, int N, int m) {
	__shared__ float x_r[64];
	__shared__ float x_i[64];
	__shared__ float w_r[32];
	__shared__ float w_i[32];
	float Wr;
	float Wi;


	int i = tx + blockDim.x * bx, j;

	x_r[reverse(N1, m1, tx)] = x_r_d[i];
	x_i[reverse(N1, m1, tx)] = x_i_d[i];
	if (tx < N1 / 2) {
		w_r[tx] = w_r_d[tx];
		w_i[tx] = w_i_d[tx];
	}
	__syncthreads();
	int a = N1 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m1; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * w_r[(tx * a) % (n * a)] - x_i[tx + n] * w_i[(tx * a) % (n * a)];
			temp_xs2_i = x_i[tx + n] * w_r[(tx * a) % (n * a)] + x_r[tx + n] * w_i[(tx * a) % (n * a)];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	Wr = cos(2 * PI * tx * bx / N);
	Wi = -sin(2 * PI * tx * bx / N);
	x_r_d[i] = x_r[tx] * Wr - x_i[tx] * Wi;
	x_i_d[i] = x_i[tx] * Wr + x_r[tx] * Wi;

}
/******************************last version***************************************************/
__global__ void fft_rad2_v9(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N1, int m1, int N, int m) {
	__shared__ float x_r[1024];
	__shared__ float x_i[1024];
	//__shared__ float w_r[512];
	//__shared__ float w_i[512];
	float Wr, Wi;
	int i = tx * gridDim.x + bx, j;

	x_r[reverse(N1, m1, tx)] = x_r_d[i];
	x_i[reverse(N1, m1, tx)] = x_i_d[i];
	/*if (tx < N1/2) {
		w_r[tx] = w_r_d[tx];
		w_i[tx] = w_i_d[tx];
	}*/
	__syncthreads();
	int a = N1 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m1; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * W[(tx * a) % (n * a)] - x_i[tx + n] * W[(tx * a) % (n * a)+512];
			temp_xs2_i = x_i[tx + n] * W[(tx * a) % (n * a)] + x_r[tx + n] * W[(tx * a) % (n * a)+512];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	Wr = cos(2 * PI * tx * bx / N);
	Wi = -sin(2 * PI * tx * bx / N);
	x_r_d[i] = x_r[tx] * Wr - x_i[tx] * Wi;
	x_i_d[i] = x_i[tx] * Wr + x_r[tx] * Wi;
}
__global__ void fft_rad2_v6(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N2, int m2, int N, int m) {
	__shared__ float x_r[1024];
	__shared__ float x_i[1024];
	//__shared__ float w_r[512];
	//__shared__ float w_i[512];

	int i = tx * gridDim.x + bx+by * gridDim.x * blockDim.x, j;

	x_r[reverse(N2, m2, tx)] = x_r_d[i];
	x_i[reverse(N2, m2, tx)] = x_i_d[i];
	//if (tx < 512) {
	//	w_r[tx] = w_r_d[tx];
	//	w_i[tx] = w_i_d[tx];
	//}
	__syncthreads();
	int a = N2 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m2; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * W[(tx * a) % (n * a)] - x_i[tx + n] * W[(tx * a) % (n * a)+512];
			temp_xs2_i = x_i[tx + n] * W[(tx * a) % (n * a)] + x_r[tx + n] * W[(tx * a) % (n * a)+512];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}

	x_r_d[i] = x_r[tx];
	x_i_d[i] = x_i[tx];
}
__global__ void fft_rad2_v7(float* x_r_d, float* x_i_d, float* w_r_d, float* w_i_d, int N3, int m3,int N,int m) {


	extern __shared__ float x_r[];
	float* x_i;
	float* w_r;
	float* w_i;


	float* X = x_r;
	x_i = (float*)&X[N3];
	w_r = (float*)&x_i[N3];
	w_i = (float*)&w_r[N3 / 2];
	

	float Wr;
	float Wi;


	int i = tx + blockDim.x * bx+by*gridDim.x*blockDim.x, j;

	x_r[reverse(N3, m3, tx)] = x_r_d[i];
	x_i[reverse(N3, m3, tx)] = x_i_d[i];
	if (tx < N3 / 2) {
		w_r[tx] = w_r_d[tx];
		w_i[tx] = w_i_d[tx];
	}
	__syncthreads();
	int a = N3 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m3; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * w_r[(tx * a) % (n * a)] - x_i[tx + n] * w_i[(tx * a) % (n * a)];
			temp_xs2_i = x_i[tx + n] * w_r[(tx * a) % (n * a)] + x_r[tx + n] * w_i[(tx * a) % (n * a)];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	Wr = cos(2 * PI * tx * bx / N);
	Wi = -sin(2 * PI * tx * bx / N);
	x_r_d[i] = x_r[tx] * Wr - x_i[tx] * Wi;
	x_i_d[i] = x_i[tx] * Wr + x_r[tx] * Wi;
}

//------------------------------functions-----------------------------------------------
//void gpu_bitreverse(float* x, const unsigned int N, const unsigned int M) {
//	float* temp;
//	cudaMalloc((void**)&temp, sizeof(float) * N);
//	bit_reverse << <N / 1024, 1024 >> > (x, temp, N, M);
//	transfer << < N / 1024, 1024 >> > (x, temp, N);
//	cudaFree(temp);
//}
void transpose(float* x, int N,int nx, int ny,int nz) {
	float* temp;
	cudaMalloc((void**)&temp, sizeof(float) * N);
	dim3 dimGrid(nx/ TILE_DIM, ny, nz);
	dim3 dimBlock(TILE_DIM, 1, 1);
	nonSquredTranspose << <dimGrid, dimBlock >> > (x, temp);
	transfer << <dimGrid, dimBlock >> > (x, temp);
	//dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
	//dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
	////transposeNoBankConflicts << <dimGrid, dimBlock >> > (temp, x);
	//transposeNaive << <dimGrid, dimBlock >> > (temp, x);
	////copySharedMem << <dimGrid, dimBlock >> > (temp, x);
	//copy << <dimGrid, dimBlock >> > (x, temp);
	cudaFree(temp);
}
void gpuKernel(float* x_r_d, float* x_i_d, /*float* X_r_d, float* X_i_d,*/ const unsigned int N, const unsigned int M)
{	
	float* w_r_1024;
	float* w_i_1024;
	float* w_r;
	float* w_i;
	//int N1;
	N1 = N / (1 << 20);
	/**************************************************/
	cudaMalloc((void**)&w_r_1024, sizeof(float) * 512);
	cudaMalloc((void**)&w_i_1024, sizeof(float) * 512);
	cudaMalloc((void**)&w_r, sizeof(float) * N1/2);
	cudaMalloc((void**)&w_i, sizeof(float) * N1 / 2);
	
	gpu_weight << <1, 512 >> > (w_r_1024, w_i_1024, 1024);
	gpu_weight <<<1, N1/2 >>> (w_r, w_i, N1);
	cudaMemcpyToSymbol(W, w_r_1024, sizeof(float) * 512,0, cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(W, w_i_1024, sizeof(float) * 512, sizeof(float) * 512, cudaMemcpyDeviceToDevice);

	/**************************************************/
	//


	dim3 Block(N/1024,1, 1);
	dim3 Thread(1024, 1, 1);
	fft_rad2_v9 <<<Block, Thread >> > (x_r_d, x_i_d, w_r_1024, w_i_1024, 1024, 10, N, M);
	
	
	/**************************************************/
	transpose(x_r_d,N, 1024, N1, 1024);
	transpose(x_i_d,N, 1024, N1, 1024);
	dim3 Block3(1024, 1024, 1);
	dim3 Thread3(N1, 1, 1);
	fft_rad2_v7 << <Block3, Thread3, N1*sizeof(float)+ N1 * sizeof(float)+ (N1 / 2) * sizeof(float)+ (N1/2) * sizeof(float) >> > (x_r_d, x_i_d, w_r, w_i, N1, M - 20, N / 1024, M - 10);
	
	
	
	/**************************************************/
	dim3 Block1(N1, 1024, 1);
	dim3 Thread1(1024, 1, 1);
	fft_rad2_v6 << <Block1, Thread1 >> > (x_r_d, x_i_d, w_r_1024, w_i_1024, 1024, 10, N / 1024, M - 10);
	transpose(x_r_d, N, N / 1024,1024, 1);
	transpose(x_i_d, N, N / 1024, 1024, 1);

	cudaFree(w_r);
	cudaFree(w_i);
	cudaFree(w_r_1024);
	cudaFree(w_i_1024);
/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 	other code	   $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*	cudaMalloc((void**)&w_r_1024, sizeof(float) * 512);
	cudaMalloc((void**)&w_i_1024, sizeof(float) * 512);
	cudaMalloc((void**)&w_r, sizeof(float) * 32);
	cudaMalloc((void**)&w_i, sizeof(float) * 32);
	gpu_weight << <1, 512 >> > (w_r_1024, w_i_1024, 1024);
	gpu_weight << <1, 32 >> > (w_r, w_i, 64);

	transpose(x_r_d, N, 1024, N / 1024, 1);
	transpose(x_i_d, N, 1024, N / 1024, 1);

	dim3 Block1(1024, 1, 1);
	dim3 Thread1(N / 1024, 1, 1);
	fft_rad2_v8 << <Block1, Thread1 >> > (x_r_d, x_i_d, w_r, w_i, N / 1024, M - 10, N, M);

	dim3 Block(N / 1024, 1, 1);
	dim3 Thread(1024, 1, 1);
	fft_rad2_v5 << <Block, Thread >> > (x_r_d, x_i_d, w_r_1024, w_i_1024, 1024, 10, N, M);
	cudaFree(w_r);
	cudaFree(w_i);
	cudaFree(w_r_1024);
	cudaFree(w_i_1024)*/;



	/*cudaMalloc((void**)&w_r_1024, sizeof(float) * 512);
	cudaMalloc((void**)&w_i_1024, sizeof(float) * 512);
	cudaMalloc((void**)&w_r, sizeof(float) * 32);
	cudaMalloc((void**)&w_i, sizeof(float) * 32);
	gpu_weight << <1, 512 >> > (w_r_1024, w_i_1024, 1024);
	gpu_weight << <1, 32 >> > (w_r, w_i, 64);
	
	transpose(x_r_d, N, 1024, N / 1024);
	transpose(x_i_d, N, 1024, N / 1024);

	dim3 Block1(1024, 1, 1);
	dim3 Thread1(N / 1024, 1, 1);
	fft_rad2_v8 << <Block1, Thread1 >> > (x_r_d, x_i_d, w_r, w_i, N / 1024, M - 10, N, M);

	dim3 Block(N / 1024, 1, 1);
	dim3 Thread(1024, 1, 1);
	fft_rad2_v5 << <Block, Thread >> > (x_r_d, x_i_d, w_r_1024, w_i_1024, 1024, 10, N, M);*/

	
	


	
	/*cudaMalloc((void**)&w_r, sizeof(float) *512);
	cudaMalloc((void**)&w_i, sizeof(float) *512);
	gpu_weight << <1, 512 >> > (w_r, w_i, 1024);
	dim3 Block(1024, 1, 1);
	dim3 Thread(1024, 1, 1);
	transpose(x_r_d, N, 1024, N / 1024);
	transpose(x_i_d, N, 1024, N / 1024);
	fft_rad2_v3 <<<Block, Thread >> > (x_r_d, x_i_d, w_r, w_i, 1024, 10,N,M);
	fft_rad2_v4 <<<Block, Thread >> > (x_r_d, x_i_d, w_r, w_i, 1024, 10);*/

	





	/*
	gpu_bitreverse(x_r_d, N, M);
	gpu_bitreverse(x_i_d, N, M);
	cudaMalloc((void**)&w_r, sizeof(float) * N/2);
	cudaMalloc((void**)&w_i, sizeof(float) * N/2);
	gpu_weight <<<N / 1024, 512 >> > (w_r, w_i,N);
	
	int j ;
	dim3 Block(N / 1024,1, 1);
	dim3 Thread(1024, 1, 1);
int a = N / 2, n = 1;
	for (j = 0; j < M; j++) {

		fft_rad2_v1 <<< Block, Thread >>> (x_r_d, x_i_d, w_r, w_i,N,a,n);
		n *= 2;
		a /= 2;
	}
	*/
//	fft_rad2 <<<N/1024,1024 >> > (x_r_d, x_i_d, w_r, w_i, N, M,N/1024);
//	fft_rad2 <<<N/1024,1024 >> > (x_r_d, x_i_d, w_r, w_i, N, M,1);
	// In this function, both inputs and outputs are on GPU.
	// No need for cudaMalloc, cudaMemcpy or cudaFree.
	
}
