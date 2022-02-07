#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
// This block enables compilation of the code with and without LIKWID in place
#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#define N 10000

int main(int argc, char* argv[])
{
    int i;
    double data[N];
    LIKWID_MARKER_INIT;
#pragma omp parallel
{
    LIKWID_MARKER_REGISTER("foo");
}
#pragma omp parallel
{
    LIKWID_MARKER_START("foo");
    #pragma omp for
    for(i = 0; i < N; i++)
    {
        data[i] = omp_get_thread_num();
    }
    LIKWID_MARKER_STOP("foo");
}
    LIKWID_MARKER_CLOSE;
    return 0;
}