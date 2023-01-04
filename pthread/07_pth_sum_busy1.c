#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "timer.h"

long T;        // # of threads
long long N;   // # of elements to add, N/T must be integer
long long sum; // sum of f(i) for i=0,1,..,N-1

long flag = 0;

//////////////////////////////////////////////////////
long long f ( long long i ) {
   return i;
}
//////////////////////////////////////////////////////
long long Serial_sum() {
   long long s=0, i;
   for (i = 0; i < N; i++) {
      s += f(i);  
   }
   return s;
}
//////////////////////////////////////////////////////
void* Thread_sum(void* rank) {
   long r = (long) rank;
   long long i;
   long long i1 = (N / T) * r;
   long long i2 = (N / T) * (r+1);

   for (i = i1; i < i2; i++) {
      while (flag != r);
      sum += f(i);  
      flag = (flag+1) % T;
   }

   return NULL;
}
//////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

   T = strtol(argv[1], NULL, 10);  
   N = strtoll(argv[2], NULL, 10);
   pthread_t* handles = (pthread_t*) malloc (T*sizeof(pthread_t));
   sum = 0;
   long j;
   double t1,t2,t3;

   GET_TIME(t1);

   for (j=0; j<T; j++)  
      pthread_create( &handles[j], NULL, Thread_sum, (void*)j );  

   for (j=0; j<T; j++)  
      pthread_join( handles[j], NULL); 

   GET_TIME(t2);
   long long serial_sum = Serial_sum();
   GET_TIME(t3);
   printf(" n=%lld \n Serial:\ttime = %e sec sum = %lld \n Parallel:\ttime = %e sec sum = %lld \n", N, t3-t2, serial_sum, t2-t1, sum);
   return 0;
}

