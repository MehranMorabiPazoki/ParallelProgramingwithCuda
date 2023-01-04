#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
 
#include <semaphore.h>

long T; // # of threads

int  count;
sem_t barrier_sem;
sem_t count_sem;
/*--------------------------------------------------------------------*/
void *test(void* rank) {
   printf("thread %ld begin\n",(long)rank);

      sem_wait(&count_sem); //similar to lock(mutex)
      if (count == T-1) {
         count = 0;
         sem_post(&count_sem); //unlock(mutex)
         int j; for (j=0; j < T-1; j++)
            sem_post(&barrier_sem);
      } else {
         count++;
         sem_post(&count_sem); //unlock(mutex)
         sem_wait(&barrier_sem);
      }

   printf("thread %ld end\n",(long)rank);
   return NULL;
} 
/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {

   long j;
   T = strtol(argv[1], NULL, 10);
   pthread_t* handles = (pthread_t*) malloc (T*sizeof(pthread_t));

   count = 0;
   sem_init(&barrier_sem, 0, 0);
   sem_init(&count_sem, 0, 1);

   for (j=0; j<T; j++)  
      pthread_create( &handles[j], NULL, test, (void*)j );  

   for (j=0; j<T; j++)  
      pthread_join( handles[j], NULL); 

   sem_destroy(&barrier_sem);
   sem_destroy(&count_sem);
 
   return 0;
}

