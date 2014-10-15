#ifndef MAIN_H
#define MAIN_H
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <exception>
#include <cufft.h>

class NotSameLengthException : public std::exception
{
        virtual const char* what() const throw()
        {
                return "The file vectors are not all the same length.";
        }
};

class FileNotFoundException : public std::exception
{
        virtual const char* what() const throw()
        {
                return "The file could not be read";
        }
};

template <typename T>
T StringToNumber(const std::string &in) {
    std::stringstream ss(in);
    T result;
    return ss >> result ? result : 0;
}

template <typename T>
std::string NumberToString(const T in) {
    std::ostringstream convert;
    convert << in;
    std::string result;
    result = convert.str();
    return result;
}

void Read_data(std::vector<std::vector<float> >& Data,const std::string filename);
void checkformat(std::ifstream &file, unsigned int * axis1, unsigned int * axis2);
bool checkaxis2(std::stringstream &lineStream, unsigned int * axis2);
void getdata(std::vector<std::vector<float> >& Data, std::ifstream &myfile, unsigned int axis1, unsigned int axis2);
void Save_data(const std::string filename, cufftComplex *data, unsigned int N);
void Save_data(const std::string filename, float *data, unsigned int N);
long long GetTimeMs64();
#endif  /* MAIN_H */
