/* 
 * File:   main.cu
 * Author: eaton
 *
 * Created on September 23, 2014, 11:08 PM
 */
#include "main.h"
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
 * Host main function
 */
int main(int argc, char** argv){
    vector<vector<float> > inputs;
    Read_data(inputs,"sampleinputs.csv");
	float * in1 = &(inputs[0][0]);
	float * in2 = &(inputs[1][0]);
	float * in1_d, *in2_d;
	int M = inputs[0].size();
	int N = 512;
	cudaMalloc((void **) &in1_d, M*sizeof(float));
	cudaMalloc((void **) &in2_d, M*sizeof(float));
	cudaMemcpy(in1_d, in1, M*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(in2_d, in2, M*sizeof(float),cudaMemcpyDeviceToHost);
	float * prefilter_d;
	cudaMalloc((void **) &prefilter_d, M*sizeof(float));
	int threads = M/N;
	createFilter<<<N,threads>>>(prefilter_d);
	float * ppf_out1 = new float[N];
	memset(ppf_out1, 0.00, N*sizeof(float));
	float * ppf_out1_d;
	cudaMalloc((void **) &ppf_out1_d,N*sizeof(float));
	cudaMemcpy(ppf_out1_d,ppf_out1_d,N*sizeof(float),cudaMemcpyDeviceToHost);
	appliedPolyphasePhysics<<<N,threads>>>(in1_d,prefilter_d,ppf_out1_d);
	//free the data again
	cudaFree(in1_d);	cudaFree(in2_d);	cudaFree(prefilter_d);	cudaFree(ppf_out1_d);
	delete[](ppf_out1);
	
    return 0;
}

void getdata(vector<vector<float> >& Data, ifstream &myfile, unsigned int axis1, unsigned int axis2) {
    string line;
    Data.resize(axis1,vector<float>(axis2, 0.00));   //maybe make this rather a double vector? YES!
    int i = 0;
    int j = 0;
    stringstream lineStream;
    while (getline(myfile, line)) {
        lineStream << line;
        string ex2;
        while (getline(lineStream, ex2, ',')) {
            float temp = StringToNumber<float>(ex2);
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
