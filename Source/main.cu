/* 
 * File:   main.cu
 * Author: eaton
 * TO MAKE RUN: nvcc main.cu -L$CUDA_TOOLKIT_ROOT_DIR/lib64 -lcuda -lcufft
 * Created on September 23, 2014, 11:08 PM
 */
typedef long long int64;
typedef unsigned long long uint64;
#include "main.h"
#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif

using namespace std;
/*
 * Polyphase filter prefilter as kernel
 */

__global__ void createFilter(float * out){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int M = blockDim.x*gridDim.x;
    int N = blockDim.x;
    float temp1 = (2.0 * M_PI * (x*1.00 / M));
    float temp2 = (0.5-0.5*__cosf(temp1));
    float temp3 = (x-M/2.00)/N;
    if(temp3 != 0){
		out[x] = __sinf(temp3)/temp3;
    }
    else{
		out[x] = 1;
    }
}

/*
 * Apply prefilter as kernel
 * Sum outputs to one N size vector
 */

__global__ void appliedPolyphasePhysics(float * in, float * filter, float * ppf_out){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    in [x] *= filter[x];
    ppf_out [threadIdx.x] += in[x];
}

/*
 * Cojugate Function.
 */
__device__ __host__ inline cufftComplex cufftConj(cufftComplex in) {
	in.y = -in.y;
	return in;
}

__device__ __host__ inline cufftComplex cufftMult(cufftComplex a, cufftComplex b){
	cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

/*
 * Cross correlate two vectors.
 */
__global__ void correlate(cufftComplex *in1, cufftComplex *in2, cufftComplex *out){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	out[x] = cufftMult(in1[x],cufftConj(in2[x]));
}

/*
 * Host main function
 */
int main(int argc, char** argv){
    vector<vector<float> > inputs;
	float * in1;
	float * in2;
	float * in1_d, *in2_d;
	float * prefilter_d;
	unsigned int M = 0;
	unsigned int N = 4096;
	unsigned int threads;
	int convtoSec = 1000000;
	float * ppf_out1;
	float * ppf_out1_d;
	float * ppf_out2;
	float * ppf_out2_d;
	int64 t_start;
	int64 t_total;
	int64 ppfPref_start;
	int64 ppfPref_stop;
	int64 ppf_start;
	int64 ppf_stop;
	int64 fft_start;
	int64 fft_stop;
	int64 correlate_start;
	int64 correlate_stop;
	int64 startcpyinputs;
	int64 stopcpyinputs;
	int64 startcpyinputs2;
	int64 stopcpyinputs2;

    Read_data(inputs,"sampleinputs.csv");
	M = inputs[0].size();
	in1 = &inputs[0][0];
	in2 = &inputs[1][0];
	cout<< "starting the simulation. please be patient. (hopefully not too much so.)";
	t_start = GetTimeMs64();
	startcpyinputs = GetTimeMs64();
	cudaMalloc((void **) &in1_d, M*sizeof(float));
	cudaMalloc((void **) &in2_d, M*sizeof(float));
	cudaMemcpy(in1_d, in1, M*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(in2_d, in2, M*sizeof(float),cudaMemcpyHostToDevice);
	stopcpyinputs = GetTimeMs64();
//	cout << "not segfault. [1]\n";
	cudaMalloc((void **) &prefilter_d, M*sizeof(float));
	threads = M/N;
	ppfPref_start = GetTimeMs64();
	createFilter<<<N,threads>>>(prefilter_d);
	//possible sync neccesary here.
	ppfPref_stop = GetTimeMs64();

	ppf_out1 = new float[N];
	memset(ppf_out1, 0.00, N*sizeof(float));
	ppf_out2 = new float[N];
	memset(ppf_out1, 0.00, N*sizeof(float));
//	cout << "not segfault. [2]\n";
	cudaMalloc((void **) &ppf_out1_d,N*sizeof(float));
	cudaMalloc((void **) &ppf_out2_d,N*sizeof(float));
// Put this section inside kernel for less useless data traffic.
//	cout << "not segfault. [3]\n";
	cudaMemcpy(ppf_out1_d,ppf_out1,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(ppf_out2_d,ppf_out2,N*sizeof(float),cudaMemcpyHostToDevice);
	
	ppf_start = GetTimeMs64();
	appliedPolyphasePhysics<<<N,threads>>>(in1_d,prefilter_d,ppf_out1_d); //mabe do a sync after every call?
	appliedPolyphasePhysics<<<N,threads>>>(in2_d,prefilter_d,ppf_out2_d);
	ppf_stop = GetTimeMs64();
	//prepare the fft
	cufftHandle plan;
	cufftComplex *output;
	cudaMalloc((void **) &output, ((N/2)+1)*sizeof(cufftComplex));
	cufftPlan1d(&plan,N, CUFFT_R2C, 1);

	cufftHandle plan2;
	cufftComplex *output2;
	cudaMalloc((void **) &output2, ((N/2)+1)*sizeof(cufftComplex));
	cufftPlan1d(&plan2,N, CUFFT_R2C, 1);

	//do the fft
	fft_start = GetTimeMs64();
	cufftExecR2C(plan,(cufftReal *) ppf_out1_d,output);
	cufftExecR2C(plan,(cufftReal *) ppf_out2_d,output2);
	//synchronize
	cudaDeviceSynchronize();
	fft_stop = GetTimeMs64();

	//prepare cross correlation
	cufftComplex *ccout1;
	cudaMalloc((void **) &ccout1, ((N/2)+1)*sizeof(cufftComplex));
	cufftComplex *ccout2;
	cudaMalloc((void **) &ccout2, ((N/2)+1)*sizeof(cufftComplex));
	//do correlation
	correlate_start = GetTimeMs64();
	correlate<<<N/2+1,1>>>(output,output2,ccout1);
	correlate<<<N/2+1,1>>>(output2,output,ccout2);
	correlate_stop = GetTimeMs64();
	//copy back to HOST
	startcpyinputs2 = GetTimeMs64();
	cufftComplex *final = (cufftComplex*) malloc((N/2+1)*sizeof(cufftComplex));
	cudaMemcpy(ccout1,final,(N/2+1)*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	cudaMemcpy(ccout2,final,(N/2+1)*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	stopcpyinputs2 = GetTimeMs64();
	Save_data("output1.csv",final,N);
	Save_data("output2.csv",final,N);
	t_total = GetTimeMs64();
	//free the data again.

	//print timing results.
	double timet = 0.00;
	int64 totalflop = 0.00;
	timet = ((double)(t_total-t_start)/convtoSec);
	cout << "\ntotal time to execute:                   " << NumberToString<double>(timet);
	timet = ((double)(ppfPref_stop-ppfPref_start)/convtoSec);
	cout << "\ntotal time to calculate prefilter        " << NumberToString<double>(timet);
	timet = ((double)(ppf_stop-ppf_start)/convtoSec);
	cout << "\ntotal time to apply Polyphasefilter      " << NumberToString<double>(timet);
	timet = ((double)(fft_stop-fft_start)/convtoSec);
	cout << "\ntotal time to apply FFT                  " << NumberToString<double>(timet);
	timet = ((double)(correlate_stop-correlate_start)/convtoSec);
	cout << "\ntotal time to apply Correlation Process  " << NumberToString<double>(timet);
	timet = ((double)(stopcpyinputs-startcpyinputs)/convtoSec);
	cout << "\ntotal time to copy in                    " << NumberToString<double>(timet);
	timet = ((double)(stopcpyinputs2-startcpyinputs2)/convtoSec);
	cout << "\ntotal time to copy out                   " << NumberToString<double>(timet);

	cout << "\ntotal clicks                             " << NumberToString<double>(t_total-t_start);
	cout << "\nclocks per second                        " << NumberToString<double>(CLOCKS_PER_SEC
);
cout << "\n";

	cudaFree(in1_d);	cudaFree(in2_d);
	cudaFree(prefilter_d);	cudaFree(ppf_out1_d);
	cudaFree(output); cudaFree(output2);
	cudaFree(ccout1); cudaFree(ccout2);
	delete[](ppf_out1); delete[](ppf_out2);
	delete[](final);

    return 0;
}


//time code added
int64 GetTimeMs64(){
#ifdef WIN32
 /* Windows */
 FILETIME ft;
 LARGE_INTEGER li;

 /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
  * to a LARGE_INTEGER structure. */
 GetSystemTimeAsFileTime(&ft);
 li.LowPart = ft.dwLowDateTime;
 li.HighPart = ft.dwHighDateTime;

 uint64 ret = li.QuadPart;
 ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
 ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

 return ret;
#else
 /* Linux */
 struct timeval tv;

 gettimeofday(&tv, NULL);

 uint64 ret = tv.tv_usec;
 /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
// NOT: ret /= 1000;

 /* Adds the seconds (10^0) after converting them to microseconds (10^-6) */
 ret += (tv.tv_sec * 1000000);

 return ret;
#endif
}
//


void getdata(vector<vector<float> >& Data, ifstream &myfile, unsigned int axis1, unsigned int axis2) {
    string line;
    int i = 0;
    int j = 0;
	float temp;
    stringstream lineStream;
    Data.resize(axis1,vector<float>(axis2, 0.00));
    while (getline(myfile, line)) {
        lineStream << line;
        string ex2;
        while (getline(lineStream, ex2, ',')) {
            temp = StringToNumber<float>(ex2);
            Data[i][j] = temp;
            j++;
        }
        j = 0;
        i++;
        lineStream.str("");
        lineStream.clear();
    }
}

bool checkaxis2(stringstream &lineStream, unsigned int * axis2) {
    string line;
    vector<string> result;
    string cell;
    while (getline(lineStream, cell, ',')) {
        result.push_back(cell);
    }
    if ((*axis2) != result.size()) {
        return false;
    }
    return true;
}

void checkformat(ifstream &file, unsigned int * axis1, unsigned int * axis2) {
    // first line axis2 should be equal throughout the file.
    // comma separated file.
    vector<string> result;
    string line;
    getline(file, line);
    stringstream lineStream(line);
    string cell;
    while (getline(lineStream, cell, ',')) {
        result.push_back(cell);
    }
    *axis2 = result.size();
    (*axis1)++;
    while (getline(file, line)) {
        stringstream lineStream(line);
        if (checkaxis2(lineStream, axis2)) {
            (*axis1)++;
        } else {
            //not same sizes.
            cout << "Error at line number:" << ((*axis1) + 1) << "\n";
            throw NotSameLengthException();
        }
    }

}

void Read_data(vector<vector<float> >& Data,const string filename) {
    unsigned int axis1 = 0;
    unsigned int axis2 = 0;
    std::ifstream myfile;
    try {
        myfile.open(filename.c_str(), ios::in);
        if (myfile.is_open()) {
            checkformat(myfile, &axis1, &axis2);
            myfile.close();
            myfile.open(filename.c_str(), ios::in);
            getdata(Data, myfile, axis1, axis2);
            myfile.close();
        } else {
            throw FileNotFoundException();
        }
    } catch (exception& e) {
        cout << e.what() << "\nPlease contact the author.";
    }
}

string toString(cufftComplex in){
	return NumberToString<float>(in.x) + string(" ") + NumberToString<float>(in.y) + string("i ");
}

void Save_data(const string filename, cufftComplex *data, unsigned int N){
    std::fstream myfile;
    try {
        myfile.open(filename.c_str(),ios::out);
        if (myfile.is_open()){
            for(unsigned int x = 0; x < N-1; x++){
                myfile << toString(data[x]) + ",";
            }
            myfile << toString(data[N-1]);
            myfile.close();
        }
    } catch (exception& e) {
        cout << e.what() << "\nPlease contact the author.";
    }
}
