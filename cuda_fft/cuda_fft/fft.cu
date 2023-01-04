//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z



 //you may define other parameters here!

 __constant__	float W[512];

const int TILE_DIM = 128;
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
__global__ void transfer(float* x, float* temp) {

	__shared__ float s[512];
	//int i =bz*gridDim.y*gridDim.x*blockDim.x+ by * gridDim.x * blockDim.x + blockDim.x * bx + tx;
	int i = by * gridDim.x * blockDim.x + blockDim.x * bx + tx;
	s[tx] = temp[i];
	x[i] = s[tx];
	
}
__global__ void transfer1(float* x, float* temp) {

	__shared__ float s[TILE_DIM];
	int i =bz*gridDim.y*gridDim.x*blockDim.x+ by * gridDim.x * blockDim.x + blockDim.x * bx + tx;
	
	s[tx] = temp[i];
	x[i] = s[tx];

}
__global__ void gpu_weight(float* w_r, float* w_i, int N) {
	int i = bx * 256 + tx;
	w_r[i] = cos(2 * PI * i / N);
	w_i[i] = -sin(2 * PI * i / N);
}


__global__ void nonSquredTranspose(float* in, float* out) {
	__shared__ float s[512];
	/*int i = bz * gridDim.y * gridDim.x * blockDim.x+by * gridDim.x* blockDim.x + blockDim.x*bx+tx;
	int j = bz * gridDim.y * gridDim.x * blockDim.x+ by + (blockDim.x * bx + tx) * gridDim.y ;*/

	int i =by * gridDim.x* blockDim.x + blockDim.x*bx+tx;
	int j = by * gridDim.x * blockDim.x + tx*gridDim.x+bx ;

	s[tx] = in[i];
	out[j] = s[tx];
}
__global__ void nonSquredTranspose1(float* in, float* out) {
	__shared__ float s[TILE_DIM];
	int i = bz * gridDim.y * gridDim.x * blockDim.x+by * gridDim.x* blockDim.x + blockDim.x*bx+tx;
	int j = bz * gridDim.y * gridDim.x * blockDim.x+ by + (blockDim.x * bx + tx) * gridDim.y ;


	s[tx] = in[i];
	out[j] = s[tx];
}
/******************************last version***************************************************/
__global__ void fft_rad2_v9(float* x_r_d, float* x_i_d, int N1, int m1, int N, int m) {
	__shared__ float x_r[512];
	__shared__ float x_i[512];

	float Wr, Wi;
	int i = tx * (gridDim.x*gridDim.y) + bx+by*gridDim.x, j;
	x_r[reverse(N1, m1, tx)] = x_r_d[i];
	x_i[reverse(N1, m1, tx)] = x_i_d[i];

	__syncthreads();
	int a = N1 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m1; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * W[(tx * a) % (n * a)] - x_i[tx + n] * W[(tx * a) % (n * a)+256];
			temp_xs2_i = x_i[tx + n] * W[(tx * a) % (n * a)] + x_r[tx + n] * W[(tx * a) % (n * a)+256];

			x_r[tx] = temp_xs2_r + temp_xs1_r;
			x_i[tx] = temp_xs2_i + temp_xs1_i;
			x_r[tx + n] = temp_xs1_r - temp_xs2_r;
			x_i[tx + n] = temp_xs1_i - temp_xs2_i;
		}
		n *= 2;
		a /= 2;
		__syncthreads();
	}
	Wr = cos(2 * PI * tx * (bx + by * gridDim.x) / N);
	Wi = -sin(2 * PI * tx * (bx + by * gridDim.x) / N);
	x_r_d[i] = x_r[tx] * Wr - x_i[tx] * Wi;
	x_i_d[i] = x_i[tx] * Wr + x_r[tx] * Wi;
}
//__global__ void fft_rad2_v9(float* x_r, float* x_i, int N1, int m1, int N, int m) {
//	//__shared__ float x_r[512];
//	//__shared__ float x_i[512];
//
//	float Wr, Wi;
//	int i = tx * gridDim.x + bx, j;
//	int o = reverse(N1, m1, tx);
//	int k = (o)*gridDim.x + bx;
//		//int i = tx * (gridDim.x*gridDim.y) + bx+by*gridDim.x, j;
//		//x_r[reverse(N1, m1, tx)] = x_r_d[i];
//		//x_i[reverse(N1, m1, tx)] = x_i_d[i];
//
//		__syncthreads();
//	int a = N1 / 2, n = 1;
//	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
//	for (j = 0; j < m1; j++) {
//		if (!(n & o)) {
//
//			temp_xs1_r = x_r[k];
//			temp_xs1_i = x_i[k];
//
//			temp_xs2_r = x_r[(o + n) * gridDim.x + bx] * W[(o * a) % (n * a)] - x_i[(o + n) * gridDim.x + bx] * W[(o * a) % (n * a) + 256];
//			temp_xs2_i = x_i[(o + n) * gridDim.x + bx] * W[(o * a) % (n * a)] + x_r[(o + n) * gridDim.x + bx] * W[(o * a) % (n * a) + 256];
//
//			x_r[k] = temp_xs2_r + temp_xs1_r;
//			x_i[k] = temp_xs2_i + temp_xs1_i;
//			x_r[(o + n) * gridDim.x + bx] = temp_xs1_r - temp_xs2_r;
//			x_i[(o + n) * gridDim.x + bx] = temp_xs1_i - temp_xs2_i;
//		}
//		n *= 2;
//		a /= 2;
//		__syncthreads();
//	}
//	Wr = cos(2 * PI * o * bx / N);
//	Wi = -sin(2 * PI * o * bx / N);
//	x_r[k] = x_r[k] * Wr - x_i[k] * Wi;
//	x_i[k] = x_i[k] * Wr + x_r[k] * Wi;
//}
__global__ void fft_rad2_v6(float* x_r_d, float* x_i_d, int N2, int m2, int N, int m) {
	__shared__ float x_r[512];
	__shared__ float x_i[512];


	int i = tx * gridDim.x + bx+by * gridDim.x * blockDim.x, j;

	x_r[reverse(N2, m2, tx)] = x_r_d[i];
	x_i[reverse(N2, m2, tx)] = x_i_d[i];

	__syncthreads();
	int a = N2 / 2, n = 1;
	float temp_xs1_r, temp_xs2_r, temp_xs1_i, temp_xs2_i;
	for (j = 0; j < m2; j++) {
		if (!(n & tx)) {

			temp_xs1_r = x_r[tx];
			temp_xs1_i = x_i[tx];

			temp_xs2_r = x_r[tx + n] * W[(tx * a) % (n * a)] - x_i[tx + n] * W[(tx * a) % (n * a)+256];
			temp_xs2_i = x_i[tx + n] * W[(tx * a) % (n * a)] + x_r[tx + n] * W[(tx * a) % (n * a)+256];

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

void transpose(float* x, int N,int nx, int ny) {
	float* temp;
	cudaMalloc((void**)&temp, sizeof(float) * N);
	dim3 dimGrid(nx/ 512, ny);
	dim3 dimBlock(512, 1, 1);
	nonSquredTranspose << <dimGrid, dimBlock >> > (x, temp);
	transfer << <dimGrid, dimBlock >> > (x, temp);
	cudaFree(temp);
}
void transpose1(float* x, int N, int nx, int ny) {
	float* temp;
	cudaMalloc((void**)&temp, sizeof(float) * N);
	dim3 dimGrid(nx / TILE_DIM, ny,1);
	dim3 dimBlock(TILE_DIM, 1, 1);
	nonSquredTranspose1 << <dimGrid, dimBlock >> > (x, temp);
	transfer1 << <dimGrid, dimBlock >> > (x, temp);
	cudaFree(temp);
}
void gpuKernel(float* x_r_d, float* x_i_d, /*float* X_r_d, float* X_i_d,*/ const unsigned int N, const unsigned int M)
{	
	float* w_r_1024;
	float* w_i_1024;
	float* w_r;
	float* w_i;
	int N1;
	N1 = N / (1 << 18);
	/**************************************************/
	cudaMalloc((void**)&w_r_1024, sizeof(float) * 256);
	cudaMalloc((void**)&w_i_1024, sizeof(float) * 256);
	cudaMalloc((void**)&w_r, sizeof(float) * N1 / 2);
	cudaMalloc((void**)&w_i, sizeof(float) * N1 / 2);
	
	gpu_weight << <1, 256 >> > (w_r_1024, w_i_1024, 512);
	gpu_weight <<<1, N1/2 >>> (w_r, w_i, N1);
	cudaMemcpyToSymbol(W, w_r_1024, sizeof(float) * 256,0, cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(W, w_i_1024, sizeof(float) * 256, sizeof(float) * 256, cudaMemcpyDeviceToDevice);
	cudaFree(w_r_1024);
	cudaFree(w_i_1024);
	/**************************************************/
	//
	int i, j;
	if (M <=24) {
		i = N / (512);
		j = 1;
		
	}
	else if (M == 25) {
		i = N / (1024);
		j = 2;
	}
	else if(M>25) {
		i = N / (2048);
		j = 4;
	}
	dim3 Block(i, j, 1);
	dim3 Thread(512, 1, 1);
	fft_rad2_v9 <<<Block, Thread >> > (x_r_d, x_i_d, 512, 9, N, M);
	
	
	/**************************************************/
	transpose(x_r_d,N, 512* N1, 512);
	transpose(x_i_d,N, 512* N1, 512);
	dim3 Block3(512, 512, 1);
	dim3 Thread3(N1, 1, 1);
	fft_rad2_v7 << <Block3, Thread3, N1*sizeof(float)+ N1 * sizeof(float)+ (N1 / 2) * sizeof(float)+ (N1/2) * sizeof(float) >> > (x_r_d, x_i_d, w_r, w_i, N1, M - 18, N / 512, M - 9);
	
	
	
	/**************************************************/
	dim3 Block1(N1, 512, 1);
	dim3 Thread1(512, 1, 1);
	fft_rad2_v6 << <Block1, Thread1 >> > (x_r_d, x_i_d, 512, 9, N / 512, M - 9);
	//transpose(x_r_d, N, N / 1024,1024, 1);
	//transpose(x_i_d, N, N / 1024, 1024, 1);
	transpose1(x_r_d, N, N / 512, 512);
	transpose1(x_i_d, N, N / 512, 512);

	cudaFree(w_r);
	cudaFree(w_i);

}
