/*  gcc whets.c cpuidc64.o cpuida64.o -m64 -lrt -lc -lm -o whet
 *
 *  Document:         Whets.c 
 *  File Group:       Classic Benchmarks
 *  Creation Date:    6 November 1996
 *  Revision Date:    6 November 2010 Ubuntu Version for PCs
 *
 *  Title:            Whetstone Benchmark in C/C++
 *  Keywords:         WHETSTONE BENCHMARK PERFORMANCE MIPS
 *                    MWIPS MFLOPS
 *
 *  Abstract:         C or C++ version of Whetstone one of the
 *                    Classic Numeric Benchmarks with example
 *                    results on P3 to P6 based PCs.        
 *
 *  Contributor:      roy@roylongbottom.org.uk
 *
 ************************************************************
 *
 *     C/C++ Whetstone Benchmark Single or Double Precision
 *
 *     Original concept        Brian Wichmann NPL      1960's
 *     Original author         Harold Curnow  CCTA     1972
 *     Self timing versions    Roy Longbottom CCTA     1978/87
 *     Optimisation control    Bangor University       1987/90
 *     C/C++ Version           Roy Longbottom          1996
 *     Compatibility & timers  Al Aburto               1996
 *
 ************************************************************
 *
 *              Official version approved by:
 *
 *         Harold Curnow  100421.1615@compuserve.com
 *
 *      Happy 25th birthday Whetstone, 21 November 1997
 *
 ************************************************************
 *
 *     The program normally runs for about 100 seconds
 *     (adjustable in main - variable duration). This time
 *     is necessary because of poor PC clock resolution.
 *     The original concept included such things as a given
 *     number of subroutine calls and divides which may be
 *     changed by optimisation. For comparison purposes the
 *     compiler and level of optimisation should be identified.
 *
 *     This version is set to run for 10 seconds using high
 *     resolution timer.
 *       
 ************************************************************
 *
 *     The original benchmark had a single variable I which
 *     controlled the running time. Constants with values up
 *     to 899 were multiplied by I to control the number
 *     passes for each loop. It was found that large values
 *     of I could overflow index registers so an extra outer
 *     loop with a second variable J was added.
 *
 *     Self timing versions were produced during the early
 *     days. The 1978 changes supplied timings of individual
 *     loops and these were used later to produce MFLOPS and
 *     MOPS ratings.
 *
 *     1987 changes converted the benchmark to Fortran 77
 *     standards and removed redundant IF statements and
 *     loops to leave the 8 active loops N1 to N8. Procedure
 *     P3 was changed to use global variables to avoid over-
 *     optimisation with the first two statements changed from
 *     X1=X and Y1=Y to X=Y and Y=Z. A self time calibrating
 *     version for PCs was also produced, the facility being
 *     incorporated in this version.
 *
 *     This version has changes to avoid worse than expected
 *     speed ratings, due to underflow, and facilities to show
 *     that consistent numeric output is produced with varying
 *     optimisation levels or versions in different languages.
 *
 *     Some of the procedures produce ever decreasing numbers.
 *     To avoid problems, variables T and T1 have been changed
 *     from 0.499975 and 0.50025 to 0.49999975 and 0.50000025.
 *
 *     Each section now has its own double loop. Inner loops
 *     are run 100 times the loop constants. Calibration
 *     determines the number of outer loop passes. The
 *     numeric results produced in the main output are for
 *     one pass on the outer loop. As underflow problems were
 *     still likely on a processor 100 times faster than a 100
 *     MHz Pentium, three sections have T=1.0-T inserted in the
 *     outer loop to avoid the problem. The two loops avoid
 *     index register overflows.
 *
 *     The first section is run ten times longer than required
 *     for accuracy in calculating MFLOPS. This time is divided
 *     by ten for inclusion in the MWIPS calculations.
 *
 *     Early version has facilities for typing in details of
 *     the particular run, appended to file whets.txt along
 *     with the results. This version attemps to obtain these
 *     automatically.
 *
 *     2010 Section 4 modified slightly to avoid over optimisation
 *     by GCC compiler
 *
 *     Roy Longbottom  roy@roylongbottom.org.uk
 *
 ************************************************************
 *
 *     Whetstone benchmark results, further details of the
 *     benchmarks and history are available from:
 *
 *     http://www.roylongbottom.org.uk/whetstone%20results.htm
 *     http://www.roylongbottom.org.uk/whetstone.htm
 *
 ************************************************************
 *
 *     Source code is available in C/C++, Fortran, Basic and
 *     Visual Basic in the same format as this version. Pre-
 *     compiled versions for PCs are also available via C++.
 *     These comprise optimised and non-optimised versions
 *     for DOS, Windows and NT. See:
 *
 *     http://www.roylongbottom.org.uk/whetstone%20results.htm
 *
 ************************************************************
 *
 * Example of initial calibration display (Pentium 100 MHz)
 *
 * Single Precision C/C++ Whetstone Benchmark
 *
 * Calibrate
 *      0.17 Seconds          1   Passes (x 100)
 *      0.77 Seconds          5   Passes (x 100)
 *      3.70 Seconds         25   Passes (x 100)
 *
 * Use 676  passes (x 100)
 *
 * 676 passes are used for an approximate duration of 100
 * seconds, providing an initial estimate of a speed rating
 * of 67.6 MWIPS.
 *
 * This is followed by the table of results as below.

 * Whetstone Single  Precision Benchmark in C/C++
 *
 * Loop content                 Result            MFLOPS     MOPS   Seconds
 *
 * N1 floating point    -1.12475025653839100      19.971              0.274
 * N2 floating point    -1.12274754047393800      11.822              3.240
 * N3 if then else       1.00000000000000000               11.659     2.530
 * N4 fixed point       12.00000000000000000               13.962     6.430
 * N5 sin,cos etc.       0.49904659390449520                2.097    11.310
 * N6 floating point     0.99999988079071040       3.360             45.750
 * N7 assignments        3.00000000000000000                2.415    21.810
 * N8 exp,sqrt etc.      0.75110864639282230                1.206     8.790
 *
 * MWIPS                                          28.462            100.134
 *
 *  Note different numeric results to single precision. Slight variations
 *  are normal with different compilers and sometimes optimisation levels. 
 *
 ***************************************************************************/
  
#include <math.h>       /* for sin, exp etc.           */
#include <stdio.h>      /* standard I/O                */ 
#include <string.h>     /* for strcpy - 3 occurrences  */
#include <stdlib.h>     /* for exit   - 1 occurrence   */
#include "cpuidh.h"
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "cudawhet.h"
#include <sys/sysinfo.h> 
#include <sys/utsname.h> 
#include <unistd.h>

 /*PRECISION PRECISION PRECISION PRECISION PRECISION PRECISION PRECISION*/

 /* #define DP */
 
#define SPDP float
#define Precision "Single"
#define opt "Opt 3 64 Bit"
#define int64 long long
#define uint64 unsigned long long


void whetstones(long xtra, long x100, int calibrate);  
void pa(SPDP e[4], SPDP t, SPDP t2);
void po(SPDP e1[4], long j, long k, long l);
void p3(SPDP *x, SPDP *y, SPDP *z, SPDP t, SPDP t1, SPDP t2);
void pout(char title[22], float ops, int type, SPDP checknum,
                  SPDP time, int calibrate, int section);

//extern "C" {void _cpuida();}
extern "C" {void _calculateMHz();}  

static SPDP loop_time[9];
static SPDP loop_mops[9];
static SPDP loop_mflops[9];
static SPDP TimeUsed;
static SPDP mwips;
static char headings[9][18];
static SPDP Check;
static SPDP results[9];
////////
char    configdata[10][200];
char    timeday[30];
char    idString1[100] = " ";
char    idString2[100] = " ";
double  theseSecs = 0.0;
double  startSecs = 0.0;
double  secs;
int     megaHz;

int     pagesize;
double  ramGB;

int CPUconf;
int CPUavail;
long int pages;


unsigned int eaxCode1   = 0;
unsigned int ebxCode1   = 0;
unsigned int ecxCode1   = 0;
unsigned int edxCode1   = 0;
unsigned int ext81edx   = 0;
unsigned int intel1amd2 = 0;
unsigned int startCount = 0;
unsigned int endCount   = 0;
unsigned int cycleCount = 0;
unsigned int millisecs  = 0;
unsigned int looptime   = 10;


int     hasMMX = 0;
int     hasSSE = 0;
int     hasSSE2 = 0;
int     hasSSE3 = 0;
int     has3DNow = 0;
//////// 
  void local_time(){
     time_t t;
     t = time(NULL);
     sprintf(timeday, "%s", asctime(localtime(&t)));
     return;

  }

  struct timespec tp1;

  void getSecs(){
     clock_gettime(CLOCK_REALTIME, &tp1);
     theseSecs =  tp1.tv_sec + tp1.tv_nsec / 1e9;           
     return;
  }

  void start_time(){
      getSecs();
      startSecs = theseSecs;
      return;
  }

  void end_time(){
      getSecs();
      secs = theseSecs - startSecs;
      millisecs = (int)(1000.0 * secs);
      return;

  }    
  int getDetails(){
     int max;
     int min = 1000000;
     int i;
     unsigned int MM_EXTENSION   = 0x00800000;
     unsigned int SSE_EXTENSION  = 0x02000000;
     unsigned int SSE2_EXTENSION = 0x04000000;
     unsigned int SSE3_EXTENSION = 0x00000001;
     unsigned int _3DNOW_FEATURE = 0x80000000;

     printf("  ####################################################\n  getDetails and MHz\n\n");

     struct sysinfo sysdata;
     struct utsname udetails; 
 
     _cpuida();
     sprintf(configdata[1], "  Assembler CPUID and RDTSC      ");  
     sprintf(configdata[2], "  CPU %s, Features Code %8.8X, Model Code %8.8X",
                           idString1, edxCode1, eaxCode1);
     sprintf(configdata[3], "  %s", idString2);

     max = 0;
     for (i=0; i<10; i++){
        startCount = 0;
        endCount   = 0;
        start_time();
        _calculateMHz();
        end_time();      
        megaHz = (int)((double)cycleCount / 1000000.0 / secs + 0.5);
        if (megaHz > max) max = megaHz;
        if (megaHz < min) min = megaHz;
     }
     sprintf(configdata[4], "  Measured - Minimum %d MHz, Maximum %d MHz", min, max);
     megaHz = max;

     sprintf(configdata[5], "  Linux Functions"); 
     CPUconf  =  get_nprocs_conf();
     CPUavail =  get_nprocs();
     sprintf(configdata[6], "  get_nprocs() - CPUs %d, Configured CPUs %d", CPUconf, CPUavail);
     pages = get_phys_pages();
     pagesize = getpagesize();

     ramGB = ((double)pages * (double)pagesize / 1024 / 1024 / 1024);

     sprintf(configdata[7], "  get_phys_pages() and size - RAM Size %5.2f GB, Page Size %d Bytes", ramGB, pagesize);

     if (uname(&udetails) > -1){
        sprintf(configdata[8], "  uname() - %s, %s, %s", udetails.sysname,
                                  udetails.nodename, udetails.release);  
        sprintf(configdata[9], "  %s, %s", udetails.version, udetails.machine);
     }

     else{
         sprintf(configdata[8], "  uname() - No details"); 
         sprintf(configdata[9], "  ");
     }

     if (edxCode1 & MM_EXTENSION){
           hasMMX = 1;
     }
     if (edxCode1 & SSE_EXTENSION){
           hasSSE = 1;
     }
     if (edxCode1 & SSE2_EXTENSION){
           hasSSE2 = 1;
     }
     if (ecxCode1 & SSE3_EXTENSION){
           hasSSE3 = 1;
     }
     if (intel1amd2 == 2){
           if (ext81edx & _3DNOW_FEATURE){
               has3DNow = 1;
           }
     }

     return 0; 
  }

void populatee1(float e1 [Arraysize]){
	for(int x = 0; x < Arraysize; x++){
		e1[x] = -1*(-x);
		if(x == 0){
			e1[x] = 1;
		}
	}
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



int main(int argc, char *argv[])
{
    int count = 10, calibrate = 1;
    long xtra = 1;
    int section;
    long x100 = 100;
    int duration = 10;
    FILE *outfile;
    char compiler[80], options[256], general[10][80] = {" "};
    char endit[80];
    int i;
    int nopause = 1;

    if (argc > 1)
     {
        switch (argv[1][0])
         {
             case 'N':
                nopause = 0;
                break;
             case 'n':
                nopause = 0;
                break;
         }
      }

    getDetails();
    for (i=1; i<10; i++)
    {
        printf("%s\n", configdata[i]);
    }

    local_time();
    printf("\n");
    printf("##########################################\n"); 
    printf("%s Precision C Whetstone Benchmark %s, %s\n", Precision, opt, timeday);

    outfile = fopen("whets.txt","a+");
    if (outfile == NULL)
      {
       printf ("Cannot open results file \n\n");
       printf("Press Enter to exit\n");
       i = getchar();
 
       exit (0);
      }
            
  printf("Calibrate\n");
  do
   {
    TimeUsed=0;
            
    whetstones(xtra,x100,calibrate);
            
    printf("%11.2f Seconds %10.0lf   Passes (x 100)\n",
                                     TimeUsed,(SPDP)(xtra));
    calibrate++;
    count--;

    if (TimeUsed > 2.0)
      {
       count = 0;
      }
       else
      {
       xtra = xtra * 5;
      }
   }
   
   while (count > 0);
       
   if (TimeUsed > 0) xtra = (long)((SPDP)(duration * xtra) / TimeUsed);
   if (xtra < 1) xtra = 1;
       
   calibrate = 0;
  
   printf("\nUse %d  passes (x 100)\n", (int)xtra);

   printf("\n          %s Precision C/C++ Whetstone Benchmark",Precision);
   
   #ifdef PRECOMP
      printf("\n          Compiler  %s", precompiler);
      printf("\n          Options   %s\n", preoptions);
   #else
      printf("\n");
   #endif
   
   printf("\nLoop content                  Result              MFLOPS "
                                "     MOPS   Seconds\n\n");

   TimeUsed=0;
   whetstones(xtra,x100,calibrate);

   printf("\nMWIPS            ");
   if (TimeUsed>0)
     {
      mwips=(float)(xtra) * (float)(x100) / (10 * TimeUsed);
     }
      else
     {
      mwips = 0;
     }  
   
   printf("%39.3f%19.3f\n\n",mwips,TimeUsed);
     
   if (Check == 0) printf("Wrong answer  ");
              

 /************************************************************************/
 /*               Add results to output file whets.txt                   */
 /************************************************************************/
 fprintf (outfile, "\n"); 
 fprintf (outfile, "##############################################\n\n");
 for (i=1; i<10; i++)
 {
     fprintf(outfile, "%s \n", configdata[i]);
 }
 fprintf (outfile, "\n");
 fprintf (outfile, "##############################################\n\n");
 fprintf (outfile, "Whetstone %s Precision C Benchmark  %s, %s\n",Precision, opt, timeday);
 fprintf (outfile, "\n");

 fprintf (outfile,"Loop content                   Result"
            "              MFLOPS      MOPS   Seconds\n\n"); 

 for (section=1; section<9; section++)
    {
     fprintf (outfile, "%s  %24.17f   ", headings[section],
                                              results[section]);
     if (loop_mops[section] == 99999)
       {          
        fprintf (outfile,"  %9.3f           %9.3f\n",
                 loop_mflops[section], loop_time[section]);
       }
       else
       {       
        fprintf (outfile, "            %9.3f %9.3f\n",
             loop_mops[section], loop_time[section], results[section]);
       }
    }
 fflush(outfile);
 fprintf (outfile, "\nMWIPS             ");
 fprintf (outfile, "%39.3f%20.3f\n\n",mwips,TimeUsed);
 fprintf (outfile, "Results  to  load  to  spreadsheet   ");
 fprintf (outfile, "     MWIPS   Mflops1   Mflops2   Mflops3   Cosmops"
                      "   Expmops  Fixpmops    Ifmops    Eqmops\n");
 fprintf (outfile, "Results  to  load  to  spreadsheet   ");   
                
 fprintf (outfile, " %9.3f %9.3f %9.3f", mwips, loop_mflops[1],
                                                         loop_mflops[2]);

 fprintf (outfile, " %9.3f %9.3f %9.3f", loop_mflops[6],
                                             loop_mops[5], loop_mops[8]);
 fprintf (outfile, " %9.3f %9.3f %9.3f\n\n", loop_mops[4],
                                              loop_mops[3], loop_mops[7]);
 fflush(outfile);

    
 fclose (outfile);
 
 printf ("\n");
 printf ("A new results file, whets.txt,  will have been created in the same\n");
 printf ("directory as the .EXE files, if one did not already exist.\n\n");
 
 if (nopause)
 {
    printf(" Press Enter\n\n");
    i = getchar();
 }  
 return 0;             
}

    void whetstones(long xtra, long x100, int calibrate)
      {

        long n1,n2,n3,n4,n5,n6,n7,n8,i,ix,n1mult;
        SPDP x,y,z;              
        long j,k,l;
        SPDP * e1 = new float[Arraysize];
	SPDP * e1_d;
                        
        SPDP t =  0.49999975;
        SPDP t0 = t;        
        SPDP t1 = 0.50000025;
        SPDP t2 = 2.0;
                
        Check=0.0;
       
        n1 = 12*x100;
        n2 = 14*x100;
        n3 = 345*x100;
        n4 = 210*x100;
        n5 = 32*x100;
        n6 = 899*x100;
        n7 = 616*x100;
        n8 = 93*x100;
        n1mult = 10;

        /* Section 1, Array elements */

        e1[0] = 1.0;
        e1[1] = -1.0;
        e1[2] = -1.0;
        e1[3] = -1.0;
	mycudaInit(e1_d,e1);
       start_time();
         {
            for (ix=0; ix<xtra; ix++)
              {
       //         for(i=0; i<n1*n1mult; i++)
        //          {
        //              e1[0] = (e1[0] + e1[1] + e1[2] - e1[3]) * t;
        //              e1[1] = (e1[0] + e1[1] - e1[2] + e1[3]) * t;
        //              e1[2] = (e1[0] - e1[1] + e1[2] + e1[3]) * t;
        //              e1[3] = (-e1[0] + e1[1] + e1[2] + e1[3]) * t;
       //           }
		wrapN1(e1_d,t,n1*n1mult);
                t = 1.0 - t;
              }
            t =  t0;                    
         }
        end_time();
	mycudaFree(e1_d,e1);
        secs = secs/(SPDP)(n1mult);
        pout("N1 floating point\0",(float)(n1*16)*(float)(xtra),
                             1,e1[3],secs,calibrate,1);

        /* Section 2, Array as parameter */
/*EDIT: get the FLOPS of device with call.*/
	start_time();
	mycudaInit(e1_d,e1);
//	cudaMalloc((void **)&e1_d,4*sizeof(SPDP));
//	cudaMemcpy(e1_d,e1,4*sizeof(SPDP),cudaMemcpyHostToDevice);
         {
            for (ix=0; ix<xtra; ix++)
              { 
  //              for(i=0; i<n2; i++)
//                  {
                     wrapN2(e1_d,t,t2,n2);
    //              }
                t = 1.0 - t;
              }
            t =  t0;
//mypa<<<xtra,n2>>>(e1_d);
         }
	mycudaFree(e1_d,e1);
        end_time();
        pout("N2 floating point\0",(float)(n2*96)*(float)(xtra),
                             1,e1[3],secs,calibrate,2);
	populatee1(e1);
//	start_time();
	int64 startSecs2 = GetTimeMs64();
	mycudaInit2(e1_d,e1);
//	cudaMalloc((void **)&e1_d,4*sizeof(SPDP));
//	cudaMemcpy(e1_d,e1,4*sizeof(SPDP),cudaMemcpyHostToDevice);
         {
  //          for (ix=0; ix<xtra; ix++)
//              { 
  //              for(i=0; i<n2; i++)
//                  {
                     wrapN2p1(e1_d,t,t2,1);
    //              }
//                t = 1.0 - t;
//              }
            t =  t0;
//mypa<<<xtra,n2>>>(e1_d);
         }
	mycudaFree2(e1_d,e1);
	int temp = e1[2];
	temp = temp*temp;
//	printf("currtime: %f",startSecs);
//	printf("currtime: %f",theseSecs);
	int64 theseSecs2 = GetTimeMs64();
	int64 secs2 = theseSecs2 - startSecs2;
        end_time();
        pout("N2.1 floatingCopy\0",(float)(Arraysize),
                             1,e1[3],(float)(secs2)/1000000,calibrate,2);


        /* Section 3, Conditional jumps */
        j = 1;
       start_time();
         {
            for (ix=0; ix<xtra; ix++)
              {
                for(i=0; i<n3; i++)
                  {
                     if(j==1)       j = 2;
                     else           j = 3;
                     if(j>2)        j = 0;
                     else           j = 1;
                     if(j<1)        j = 1;
                     else           j = 0;
                  }
              }
         }
        end_time();
        pout("N3 if then else  \0",(float)(n3*3)*(float)(xtra),
                        2,(SPDP)(j),secs,calibrate,3);

        /* Section 4, Integer arithmetic */
        j = 1;
        k = 2;
        l = 3;
        e1[0] = 0.0;
        e1[1] = 0.0;
       start_time();
         {
            for (ix=0; ix<xtra; ix++)
              {
                for(i=0; i<n4; i++)
                  {
                     j = j *(k-j)*(l-k);
                     k = l * k - (l-j) * k;
                     l = (l-k) * (k+j);
                     e1[l-2] = e1[l-2] + j + k + l;
                     e1[k-2] = e1[k-2] + j * k * l;
    //  was          e1[l-2] = j + k + l; and  e1[k-2] = j * k * l;
                  }
              }
         }
        end_time();
        x = (e1[0]+e1[1])/(SPDP)n4/(SPDP)xtra;   // was x = e1[0]+e1[1];
        pout("N4 fixed point   \0",(float)(n4*15)*(float)(xtra),
                                 2,x,secs,calibrate,4);
     
        /* Section 5, Trig functions */
        x = 0.5;
        y = 0.5;
       start_time();
         {
            for (ix=0; ix<xtra; ix++)
              {
                for(i=1; i<n5; i++)
                  {
                     x = t*atan(t2*sin(x)*cos(x)/(cos(x+y)+cos(x-y)-1.0));
                     y = t*atan(t2*sin(y)*cos(y)/(cos(x+y)+cos(x-y)-1.0));
                  }
                t = 1.0 - t;
              }
            t = t0;
         }
        end_time();
        pout("N5 sin,cos etc.  \0",(float)(n5*26)*(float)(xtra),
                                 2,y,secs,calibrate,5);
  

        /* Section 6, Procedure calls */
        x = 1.0;

        y = 1.0;
        z = 1.0;
       start_time();
         {
//            for (ix=0; ix<xtra; ix++)
//              {
  //              for(i=0; i<n6; i++)
//                  {
//                     p3(&x,&y,&z,t,t1,t2);
  //                }
		
//              }

	wrapN6(&x,&y,&z,t,t1,t2,n6,xtra);
         }
        end_time();
/*EDIT: try and get copy time to device */
        pout("N6 floating point\0",(float)(n6*6)*(float)(xtra),
                                1,z,secs,calibrate,6);
  
        /* Section 7, Array refrences */
        j = 0;
        k = 1;
        l = 2;
        e1[0] = 1.0;
        e1[1] = 2.0;
        e1[2] = 3.0;
	/* copy to device pointers */
       start_time();
	//mycudaInit(e1_d,e1);
//	cudaMalloc((void **)&e1_d,4*sizeof(SPDP));
//	cudaMemcpy(e1_d,e1,4*sizeof(SPDP),cudaMemcpyHostToDevice);
         {
            for (ix=0; ix<xtra; ix++)
              {
                for(i=0;i<n7;i++)
                  {
                     po(e1,j,k,l);
                  }
              }
         }
	//mycudaFree(e1_d,e1);
        end_time();

        pout("N7 assignments   \0",(float)(n7*3)*(float)(xtra),
                            2,e1[2],secs,calibrate,7);
        
        /* Section 8, Standard functions */
        x = 0.75;
       start_time();
         {
            for (ix=0; ix<xtra; ix++)
              {
                for(i=0; i<n8; i++)
                  {
                     x = sqrt(exp(log(x)/t1));
                  }
              }
         }
        end_time();
        pout("N8 exp,sqrt etc. \0",(float)(n8*4)*(float)(xtra),
                                2,x,secs,calibrate,8);

        return;
      }


    void pa(SPDP e[4], SPDP t, SPDP t2)
      {
/*replaced with cuda code*/
         long j;
         for(j=0;j<6;j++)
            {
               e[0] = (e[0]+e[1]+e[2]-e[3])*t;
               e[1] = (e[0]+e[1]-e[2]+e[3])*t;
               e[2] = (e[0]-e[1]+e[2]+e[3])*t;
               e[3] = (-e[0]+e[1]+e[2]+e[3])/t2;
            }

         return;
      }

    void po(SPDP e1[4], long j, long k, long l)
      {
         e1[j] = e1[k];
         e1[k] = e1[l];
         e1[l] = e1[j];
         return;
      }

    void p3(SPDP *x, SPDP *y, SPDP *z, SPDP t, SPDP t1, SPDP t2)
      {
//replaced with cuda code..
         *x = *y;
         *y = *z;
         *x = t * (*x + *y);
         *y = t1 * (*x + *y);
         *z = (*x + *y)/t2;
         return;
      }


    void pout(char title[18], float ops, int type, SPDP checknum,
              SPDP time, int calibrate, int section)
      {
        SPDP mops,mflops;

        Check = Check + checknum;
        loop_time[section] = time;
        strcpy (headings[section],title);
        TimeUsed =  TimeUsed + time;
        if (calibrate == 1)
     
          {
              results[section] = checknum;
          }
        if (calibrate == 0)
          {              
            printf("%s %24.17f    ",headings[section],results[section]);    
       
            if (type == 1)
             {
                if (time>0)
                 {
                    mflops = ops/(1000000L*time);
                 }
                else
                 {
                   mflops = 0;
                 }
                loop_mops[section] = 99999;
                loop_mflops[section] = mflops;
                printf(" %9.3f          %9.3f\n",
                 loop_mflops[section], loop_time[section]);                
             }
            else
             {
                if (time>0)
                 {
                   mops = ops/(1000000L*time);
                 }
                else
                 {
                   mops = 0;
                 }
                loop_mops[section] = mops;
                loop_mflops[section] = 0;                 
                printf("           %9.3f%9.3f\n",
                 loop_mops[section], loop_time[section]);
             }
          }
          
        return;
      }





