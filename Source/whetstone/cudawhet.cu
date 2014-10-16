
__global__ void mypan2(float in[4],float t,float t2){
//	float t2 = 2.0;
//	float t = 1.0 - (threadIdx.x + blockIdx.x*blockDim.x)c -c -arch=sm_20 cudacode.cu
//	int x = blockIdx.x;
	for(int y = 0; y < 6; y ++){
		in[0] = (in[0]+in[1]+in[2]-in[3])*t;
		in[1] = (in[0]+in[1]-in[2]+in[3])*t;
		in[2] = (in[0]-in[1]+in[2]+in[3])*t;
		in[3] = (-in[0]+in[1]+in[2]-in[3])/t2;
	}
}

__global__ void mypan1(float in[4],float t){
//	float t2 = 2.0;
//	float t = 1.0 - (threadIdx.x + blockIdx.x*blockDim.x)c -c -arch=sm_20 cudacode.cu
//	int x = blockIdx.x;
	for(int y = 0; y < 6; y ++){
		in[0] = (in[0]+in[1]+in[2]-in[3])*t;
		in[1] = (in[0]+in[1]-in[2]+in[3])*t;
		in[2] = (in[0]-in[1]+in[2]+in[3])*t;
		in[3] = (-in[0]+in[1]+in[2]-in[3])*t;
	}
}

__global__ void myp3(float *x, float *y, float *z, float t, float t1, float t2){
	*x = *y;
	*y = *z;
	*x = t* (*x + *y);
	*y = t1 * (*x + *y);
	*z = (*x + *y)/t2;
}

void wrapN2(float in[4],float t,float t2,long n2){
	mypan2<<<n2, 1>>>(in,t,t2);
	cudaDeviceSynchronize();
}

void wrapN1(float in[4],float t,long n1){
	mypan1<<<n1,1>>>(in,t);
	cudaDeviceSynchronize();
}

void wrapN6(float *x,float *y, float *z, float t, float t1, float t2, long int n6,long int xtra){
	float *x_d;
	cudaMalloc((void **)&x_d,sizeof(float));
	cudaMemcpy(x_d,x,sizeof(float),cudaMemcpyHostToDevice);
	float *y_d;
	cudaMalloc((void **)&y_d,sizeof(float));
	cudaMemcpy(y_d,y,sizeof(float),cudaMemcpyHostToDevice);
	float *z_d;
	cudaMalloc((void **)&z_d,sizeof(float));
	cudaMemcpy(z_d,z,sizeof(float),cudaMemcpyHostToDevice);
	myp3<<<n6,xtra>>>(x_d,y_d,z_d,t,t1,t2);
	cudaDeviceSynchronize();
	cudaMemcpy(x,x_d,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(y,y_d,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(z,z_d,sizeof(float),cudaMemcpyDeviceToHost);
}

void mycudaInit(float *in_d,float *in){
	cudaMalloc((void **)&in_d,4*sizeof(float));
        cudaMemcpy(in_d,in,4*sizeof(float),cudaMemcpyHostToDevice);
}

void mycudaFree(float *in_d, float *in){
	cudaMemcpy(in,in_d,4*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(in_d);
}
