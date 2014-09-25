/* 
 * File:   main.cu
 * Author: eaton
 *
 * Created on September 23, 2014, 11:08 PM
 */
#include "main.h"
using namespace std;
   
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
 * 
 */
int main(int argc, char** argv){
    vector<vector<double> > inputs;
    Read_data(inputs,"sampleinputs.csv");
    return 0;
}

void getdata(vector<vector<double> >& Data, ifstream &myfile, unsigned int axis1, unsigned int axis2) {
    string line;
    Data.resize(axis1,vector<double>(axis2, 0.00));   //maybe make this rather a double vector? YES!
    int i = 0;
    int j = 0;
    stringstream lineStream;
    while (getline(myfile, line)) {
        lineStream << line;
        string ex2;
        while (getline(lineStream, ex2, ',')) {
            double temp = StringToNumber<double>(ex2);
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

void Read_data(vector<vector<double> >& Data,const string filename) {
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
