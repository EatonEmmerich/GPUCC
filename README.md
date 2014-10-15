GPUCC
=====

Cross correlator implementation on a GPU using Nvidea CUDA

Compile with $ nvcc main.cu -lcufft
then run $ a.out

if run throws error, try adding to the LD_LIBRARY_PATH with:
$ $ export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH
