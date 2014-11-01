#define Arraysize 65535
void wrapN2(float in[4], float t, float t2, long int n2);
void wrapN1(float in[4], float t, long int n2);
void wrapN2p1(float *in, float t, float t2, long int n2);

void wrapN6(float *x, float *y, float *z, float t, float t1, float t2, long int n6, long int xtra);
void mycudaInit(float *in_d,float *in);
void mycudaFree(float *in_d, float *in);
void mycudaInit2(float *in_d,float *in);
void mycudaFree2(float *in_d, float *in);


