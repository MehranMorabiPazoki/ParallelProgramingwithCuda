/* 
 * Compile:  gcc -O0 1_pth_hello_simple.c -lpthread
 * Usage:    ./a.out
 *
 * Sample code is taken from "An Introduction to Parallel Programming" book
 * Modified by Matin Hashemi
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> 

/*-------------------------------------------------------------------*/
void* Salaam(void* input) {
   printf("Salaam ;)\n");
   return NULL;
}
/*-------------------------------------------------------------------*/
void* Hola(void* input) {
   printf("Hola :)\n");
   return NULL;
}
/*-------------------------------------------------------------------*/
void* Hello(void* input) {
   printf("Hello :-)\n");
   return NULL;
}
/*--------------------------------------------------------------------*/
int main() {
pthread_t handle1, handle2, handle3;
   
   printf("Main\n");
   
   pthread_create( &handle1, NULL, Salaam, NULL);
   pthread_create( &handle2, NULL, Hello, NULL);
   pthread_create( &handle3, NULL, Hola, NULL);

   pthread_join(handle1, NULL); 
   pthread_join(handle2, NULL);
   pthread_join(handle3, NULL);


   return 0;
}

