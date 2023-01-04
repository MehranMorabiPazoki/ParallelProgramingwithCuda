//Written by Matin Hashemi

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 

#ifdef DEBUG
	#define N 4
	int x[N] = {1,2,3,4};
	int y[N] = {8,7,6,5};
	int z[N];
#else
	#define N 1000*1000*200
	int* x;
	int* y;
	int* z;
#endif

/*--------------------------------------------------------------------*/
void *add(void* rank) {
   //printf("begin: rank=%d\n",(int)rank);

   long a = (long) rank;
   long i; for (i= a*N/4; i< (a+1)*N/4; i++) 
       z[i] = x[i] + y[i];

   //printf("end: rank=%d\n",(int)rank);
   return NULL;
} 
/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {

   #ifndef DEBUG
   x = malloc( N * sizeof(int) ); 
   y = malloc( N * sizeof(int) );
   z = malloc( N * sizeof(int) );
   #endif

   pthread_t handle0, handle1, handle2, handle3;

   pthread_create( & handle0, NULL, add, (void*) (0) );  
   pthread_create( & handle1, NULL, add, (void*) (1) );  
   pthread_create( & handle2, NULL, add, (void*) (2) );  
   pthread_create( & handle3, NULL, add, (void*) (3) ); 

   pthread_join( handle0, NULL); 
   pthread_join( handle1, NULL);
   pthread_join( handle2, NULL); 
   pthread_join( handle3, NULL); 
   
   #ifdef DEBUG
   long i; for (i=0; i<N; i++) printf("%d\n",z[i]);
   #endif
   
   return 0;
}
