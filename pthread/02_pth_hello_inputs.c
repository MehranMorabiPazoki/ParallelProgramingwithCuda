/* 
 * Compile:  gcc -O2 2_pth_hello_simple.c -lpthread
 * Usage:    ./a.out
 *
 * Written by Matin Hashemi
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 

/*-------------------------------------------------------------------*/
struct Point {
   int x;
   int y;
};
/*-------------------------------------------------------------------*/
void* Hello(void* input) {
   long v = (long) input;  /* Use long in case of 64-bit system */ 
   printf( "Hello :-) %ld\n", v );
   return NULL;
}
/*-------------------------------------------------------------------*/
void* Hola(void* input) {
   int* a = (int*) input;
   printf( "Hola :) %d\n", a[0]+a[1]+a[2] );
   return NULL;
}
/*-------------------------------------------------------------------*/
void* Salaam(void* input) {
   struct Point* s = (struct Point*) input;
   printf( "Salaam ;) %d\n", (*s).y ); //s->y
   return NULL;
}
/*--------------------------------------------------------------------*/
int main() {

   unsigned long long v = 10;
   int a [3] = {4,5,6};
   struct Point s = {7,8};

   pthread_t handle1, handle2, handle3;
   pthread_create( &handle1, NULL, Hello,  (void*) v    );
   pthread_create( &handle2, NULL, Hola,   (void*) a    );
   pthread_create( &handle3, NULL, Salaam, (void*) (&s) );
   printf("Main\n");
   pthread_join(handle2, NULL); 
   pthread_join(handle1, NULL);
   pthread_join(handle3, NULL);

   return 0;
}

