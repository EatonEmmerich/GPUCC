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

vector<vector<float> > inputs;
float * in1;
float * in2;
float * in1_d, *in2_d;
float * prefilter_d;
unsigned int M = 0;
int N = 4096*16;
unsigned int threads;
int convtoSec = 1000000;
float * ppf_out1_d;
float * ppf_out2_d;
float * testingvalues;
int64 t_start;
int64 t_total;
double t_tot = 0.00;
int64 ppfPref_start;
int64 ppfPref_stop;
double t_ppfpre = 0.00;
int64 ppf_start;
int64 ppf_stop;
double t_ppf = 0.00;
int64 fft_start;
int64 fft_stop;
double t_fft = 0.00;
int64 correlate_start;
int64 correlate_stop;
double t_cc = 0.00;
int64 startcpyinputs;
int64 stopcpyinputs;
double t_cpy1 = 0.00;
int64 startcpyinputs2;
int64 stopcpyinputs2;
double t_cpy2 = 0.00;



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
	
	out[x] = out[x]*temp2;
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

__global__ void initializePPFOut(float * in){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	in[x] = 0.00;
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

void runCross(){

	t_start = GetTimeMs64();
	startcpyinputs = GetTimeMs64();
	cudaMalloc((void **) &in1_d, M*sizeof(float));

	cudaMalloc((void **) &in2_d, M*sizeof(float));
	cudaMemcpy(in1_d, in1, M*sizeof(float),cudaMemcpyHostToDevice);

	cudaMemcpy(in2_d, in2, M*sizeof(float),cudaMemcpyHostToDevice);
	stopcpyinputs = GetTimeMs64();
	cudaMalloc((void **) &prefilter_d, M*sizeof(float));
	threads = M/N;
	ppfPref_start = GetTimeMs64();
	createFilter<<<N,threads>>>(prefilter_d);
	//possible sync neccesary here.
	cudaDeviceSynchronize();
	ppfPref_stop = GetTimeMs64();
	
	//<<<testing>>> save to output file.
//	cudaDeviceSynchronize();
	testingvalues = (float*) malloc(M*sizeof(float));

//	cudaMemcpy
	cudaMemcpy(testingvalues, prefilter_d, M*sizeof(float),cudaMemcpyDeviceToHost);

	Save_data("PPFPrefilter.csv",testingvalues, M);

	cudaMalloc((void **) &ppf_out1_d,N*sizeof(float));
	cudaDeviceSynchronize();

	cudaMalloc((void **) &ppf_out2_d,N*sizeof(float));
// Put this section inside kernel for less useless data traffic.

	initializePPFOut<<<N/32,32>>>(ppf_out1_d);

	initializePPFOut<<<N/32,32>>>(ppf_out2_d);
	cudaDeviceSynchronize();
	ppf_start = GetTimeMs64();
	appliedPolyphasePhysics<<<N,threads>>>(in1_d,prefilter_d,ppf_out1_d); 
//mabe do a sync after every call?
	appliedPolyphasePhysics<<<N,threads>>>(in2_d,prefilter_d,ppf_out2_d);
	cudaDeviceSynchronize();
	ppf_stop = GetTimeMs64();

	//prepare the fft
	cufftHandle plan;
	cufftComplex *output;
	cudaMalloc((void **) &output, ((N/2)+1)*sizeof(cufftComplex));
	cudaDeviceSynchronize();
	cufftPlan1d(&plan, N, CUFFT_R2C, 1);

	cufftHandle plan2;
	cufftComplex *output2;
	cudaMalloc((void **) &output2, ((N/2)+1)*sizeof(cufftComplex));
	cufftPlan1d(&plan2,N, CUFFT_R2C, 1);

	//do the fft
	fft_start = GetTimeMs64();
	cufftExecR2C(plan,(cufftReal *) ppf_out1_d,(cufftComplex *) output);
	cufftExecR2C(plan,(cufftReal *) ppf_out2_d,output2);
	//synchronize
	cudaDeviceSynchronize();
	fft_stop = GetTimeMs64();
	cufftDestroy(plan);

	//prepare cross correlation
	cufftComplex *ccout1;
	cudaMalloc((void **) &ccout1, ((N/2)+1)*sizeof(cufftComplex));
	cufftComplex *ccout2;
	cudaMalloc((void **) &ccout2, ((N/2)+1)*sizeof(cufftComplex));
	cudaDeviceSynchronize();

	//do correlation
	correlate_start = GetTimeMs64();
	correlate<<<N/2+1,1>>>(output,output2,ccout1);
	correlate<<<N/2+1,1>>>(output2,output,ccout2);
	cudaDeviceSynchronize();
	correlate_stop = GetTimeMs64();
	//copy back to HOST
	startcpyinputs2 = GetTimeMs64();
	cufftComplex *final = (cufftComplex*) malloc((N/2+1)*sizeof(cufftComplex));
	cufftComplex *final2 = (cufftComplex*) malloc((N/2+1)*sizeof(cufftComplex));
	cudaMemcpy(final, ccout1,(N/2+1)*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	cudaMemcpy(final2, ccout2,(N/2+1)*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	stopcpyinputs2 = GetTimeMs64();
	Save_data("output1.csv",final,N);
	Save_data("output2.csv",final2,N);
	t_total = GetTimeMs64();
	//t_tot += ((double)(t_total-t_start)/convtoSec);
	t_ppfpre += ((double)(ppfPref_stop-ppfPref_start)/convtoSec);
	t_ppf += ((double)(ppf_stop-ppf_start)/convtoSec);
	t_fft += ((double)(fft_stop-fft_start)/convtoSec);
	t_cc += ((double)(correlate_stop-correlate_start)/convtoSec);
	t_cpy1 += ((double)(stopcpyinputs-startcpyinputs)/convtoSec);
	t_cpy2 += ((double)(stopcpyinputs2-startcpyinputs2)/convtoSec);




	//free the data again.
	cudaFree(in1_d);	cudaFree(in2_d);
	cudaFree(prefilter_d);	cudaFree(ppf_out1_d);
	cudaFree(ppf_out2_d);
	cudaFree(output); cudaFree(output2);
	cudaFree(ccout1); cudaFree(ccout2);
	delete[](final); delete[](final2);

}

/*
 * Host main function
 */
int main(int argc, char** argv){
	Read_data(inputs,"sampleinputs.csv");
	M = inputs[0].size();
	in1 = &inputs[0][0];
	in2 = &inputs[1][0];
	cout<< "\nstarting the simulation. please be patient. (hopefully not too much so.)";
	int runs = 100;
	for(int x = 0; x < runs; x++){
		runCross();
	}
	t_ppfpre = t_ppfpre/runs;
	t_ppf = t_ppf/runs;
	t_fft = t_fft/runs;
	t_cc = t_cc/runs;
	t_cpy1 = t_cpy1/runs;
	t_cpy2 = t_cpy2/runs;


	//print timing results.
//	double timet = 0.00;
//	int64 totalflop = 0.00;
//	cout << "\ntotal time to execute:                   " << NumberToString<double>(t_tot);
	cout << "\ntotal time to calculate prefilter        " << NumberToString<double>(t_ppfpre);
	cout << "\ntotal time to apply Polyphasefilter      " << NumberToString<double>(t_ppf);
	cout << "\ntotal time to apply FFT                  " << NumberToString<double>(t_fft);
	cout << "\ntotal time to apply Correlation Process  " << NumberToString<double>(t_cc);
	cout << "\ntotal time to copy in                    " << NumberToString<double>(t_cpy1);
	cout << "\ntotal time to copy out                   " << NumberToString<double>(t_cpy2);
	cout << "\ntotal clicks                             " << NumberToString<double>(t_total-t_start);
	cout << "\nclocks per second                        " << NumberToString<double>(CLOCKS_PER_SEC);
	cout << "\nwindowsize:                              " << N;
	cout << "\nsamples:                                 " << M;
	cout << "\n";



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


void Save_data(const string filename, float *data, unsigned int N){
    std::fstream myfile;
    try {
        myfile.open(filename.c_str(),ios::out);
        if (myfile.is_open()){
            for(unsigned int x = 0; x < N-1; x++){
                myfile << NumberToString<float>(data[x]) + ",";
            }
            myfile << NumberToString<float>(data[N-1]);
            myfile.close();
        }
    } catch (exception& e) {
        cout << e.what() << "\nPlease contact the author.";
    }
}
