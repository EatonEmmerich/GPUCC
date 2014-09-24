#include <iostream>
#include <string>
void random_ints(int* a, int N)
{
	int i;
	for (i = 0; i < N; ++i)
		a[i] = rand();
}

__global__ void add(int * a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

void printc(int* c, int N) {
	for(int i = 0; i < N; i++){
		std::cout << c[i];
	}
}

int main(void) {
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int N = 2048*2048;
	int NUM_THREADS = 512;
	int size = N*sizeof(int);
	
	cudaMalloc((void **) &d_a,size);
	cudaMalloc((void **) &d_b,size);
	cudaMalloc((void **) &d_c,size);
	
	a = (int *)malloc(size);
	random_ints(a,N);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
	random_ints(b,N);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	add<<<N/NUM_THREADS,NUM_THREADS>>>(d_a,d_b,d_c);
	cudaMemcpy(c,d_c, size, cudaMemcpyDeviceToHost);
	printc(c, N);

	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
