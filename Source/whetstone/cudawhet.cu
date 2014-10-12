
__global__ void mypa(float in[4],float t,float t2){
//	float t2 = 2.0;
//	float t = 1.0 - (threadIdx.x + blockIdx.x*blockDim.x)c -c -arch=sm_20 cudacode.cu
	for(int x = 0; x < 6; x ++){
		in[0] = (in[0]+in[1]+in[2]-in[3])*t;
		in[1] = (in[0]+in[1]-in[2]+in[3])*t;
		in[2] = (in[0]-in[1]+in[2]+in[3])*t;
		in[3] = (-in[0]+in[1]+in[2]-in[3])/t2;
	}
}

void wrap(float in[4],float t,float t2){
	mypa<<<1, 1>>>(in,t,t2);
}

void mycudaInit(float *in_d,float *in){
	cudaMalloc((void **)&in_d,4*sizeof(float));
        cudaMemcpy(in_d,in,4*sizeof(float),cudaMemcpyHostToDevice);
}

void mycudaFree(float *in_d, float *in){
	cudaMemcpy(in_d,in,4*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(in_d);
}
