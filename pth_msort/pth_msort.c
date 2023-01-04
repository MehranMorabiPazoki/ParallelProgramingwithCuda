// Include your C header files here
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pth_msort.h"

int* values_arr = NULL;
int* sorted_arr = NULL;
int length;
int left[5]={0,0,0,0,0};
int right[5]={0,0,0,0,0};

struct arr_idx {
   int* input;
   int* output;
   int first;
   int middle;
   int last;
};

void swap(int* a, int* b) 
{ 
    int t = *a; 
    *a = *b; 
    *b = t; 
} 
  
int partition (int* arr, int f, int l) 
{ 
    int p = arr[l];  
    int i = (f - 1);  
    int j;

    for (j = f; j <= l- 1; j++) 
    {     
        if (arr[j] < p) 
        { 
            i++;  

            swap(&arr[i], &arr[j]); 
        } 
    } 

    swap(&arr[i + 1], &arr[l]); 

    return (i + 1); 
} 

void quickSort(int* arr, int f, int l) 
{ 
    if (f < l) 
    { 
        int p_idx = partition(arr, f, l); 

        quickSort(arr, f, p_idx - 1); 
        
        quickSort(arr, p_idx + 1, l); 
    } 
}

void merge(int* in, int* out, int f, int m, int l)
{
    int i, j, k;
    int n1 = m - f + 1;
    int n2 = l - m;

    int* L = &in[f];
    int* R = &in[m+1]; 
 
    i = 0; 
    j = 0; 
    k = f; 

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            out[k] = L[i];
            i++;
        }
        else {
            out[k] = R[j];
            j++;
        }
        k++;
    }
 
    while (i < n1) {
        out[k] = L[i];
        i++;
        k++;
    }
 
    while (j < n2) {
        out[k] = R[j];
        j++;
        k++;
    }
	
}

int binarySearch(int arr[], int f, int l, int x) 
{ 
    if (l >= f) { 

        int mid = f + (l - f) / 2; 
  
        if (x<=arr[f]) return f;

        if (x>=arr[l]) return l+1;

        if ((x <= arr[mid]) && (x >= arr[mid-1]))  return mid; 
        
        if (arr[mid] > x)  return binarySearch(arr, f, mid - 1, x); 
   
        return binarySearch(arr, mid + 1, l, x); 
    } 
    return -1; 
}

void indexes(int* idx_arr, int* arr1, int* arr2, int n) {

	int a = n/3;

	int idx1 = binarySearch(arr1, 0, n-1, arr2[a]);
    int idx2 = binarySearch(arr1, 0, n-1, arr2[2*a]);

	idx_arr[0] = 0;
	idx_arr[1] = a;
	idx_arr[2] = 2*a;
	idx_arr[3] = idx1;
	idx_arr[4] = idx2;

	quickSort(idx_arr, 0, 4);
}

void cp() {

    int i=0;
    int j=0;

    int* arr1 = sorted_arr;
    int* arr2 = &sorted_arr[length/2];

	for(i=left[0];i<left[1];i++) {  
        values_arr[j]=arr1[i]; 
	    j++;   
    }

    for(i=right[0];i<right[1];i++) {
        values_arr[j]=arr2[i];
	    j++;     
    }

    for(i=left[1];i<left[2];i++) {
        values_arr[j]=arr1[i];
	    j++;     
    }

    for(i=right[1];i<right[2];i++) {
        values_arr[j]=arr2[i]; 
	    j++;    
    }

    for(i=left[2];i<left[3];i++) {
        values_arr[j]=arr1[i];
	    j++;     
    }

    for(i=right[2];i<right[3];i++) {
        values_arr[j]=arr2[i];
	    j++;     
    }

    for(i=left[3];i<left[4];i++) {
        values_arr[j]=arr1[i];
	    j++;     
    }

    for(i=right[3];i<right[4];i++) {
        values_arr[j]=arr2[i];
	    j++;     
    }

    for(i=left[4];i<length/2;i++) {
        values_arr[j]=arr1[i];
	    j++;     
    }

    for(i=right[4];i<length/2;i++) {
        values_arr[j]=arr2[i]; 
	    j++;    
    } 
       
}

void* func1(void* input) {
   struct arr_idx* s = (struct arr_idx*) input;
   quickSort(s->input, s->first, s ->last);
   return NULL;
}

void* func2(void* input) {
   struct arr_idx* s = (struct arr_idx*) input;
   merge(s->input, s->output, s->first, s->middle, s ->last);
   return NULL;
}

void mergeSortParallel (const int* values, unsigned int N, int* sorted) {

    length = N;
    values_arr = (int*) values;
    sorted_arr = sorted;

    pthread_t h1,h2,h3,h4,h5;

    struct arr_idx s1 = {values_arr, values_arr,  0,           0, (int)(N/4-1)};
    struct arr_idx s2 = {values_arr, values_arr, (int)(N/4),   0, (int)(N/2-1)};
    struct arr_idx s3 = {values_arr, values_arr, (int)(N/2),   0, (int)(3*N/4-1)};
    struct arr_idx s4 = {values_arr, values_arr, (int)(3*N/4), 0, (int)(N-1)};

    pthread_create(&h1, NULL, func1, (void*) (&s1) );
    pthread_create(&h2, NULL, func1, (void*) (&s2) );
    pthread_create(&h3, NULL, func1, (void*) (&s3) );
    pthread_create(&h4, NULL, func1, (void*) (&s4) );

    pthread_join(h1, NULL);
    pthread_join(h2, NULL);
    pthread_join(h3, NULL);
    pthread_join(h4, NULL);


    struct arr_idx s5 = {values_arr, sorted_arr, 0,          (int)(N/4-1), (int)(N/2-1)};
    struct arr_idx s6 = {values_arr, sorted_arr, (int)(N/2), (int)(3*N/4-1), (int)(N-1)};

    pthread_create(&h1, NULL, func2, (void*) (&s5) );
    pthread_create(&h2, NULL, func2, (void*) (&s6) );


    pthread_join(h1, NULL);
    pthread_join(h2, NULL);
    
	indexes(left, sorted_arr, &sorted_arr[N/2], N/2);
	indexes(right, &sorted_arr[N/2], sorted_arr, N/2);

    cp();
    
    struct arr_idx s7 = {values_arr, sorted_arr, left[0]+right[0], right[0]+left[1]-1, left[1]+right[1]-1};
    struct arr_idx s8 = {values_arr, sorted_arr, left[1]+right[1], right[1]+left[2]-1, left[2]+right[2]-1};
    struct arr_idx s9 = {values_arr, sorted_arr, left[2]+right[2], right[2]+left[3]-1, left[3]+right[3]-1};
    struct arr_idx s10 = {values_arr, sorted_arr, left[3]+right[3], right[3]+left[4]-1, left[4]+right[4]-1};
    struct arr_idx s11 = {values_arr, sorted_arr, left[4]+right[4], right[4]+(length/2)-1, length-1};

    pthread_create(&h1, NULL, func2, (void*) (&s7) );
    pthread_create(&h2, NULL, func2, (void*) (&s8) );
    pthread_create(&h3, NULL, func2, (void*) (&s9) );
    pthread_create(&h4, NULL, func2, (void*) (&s10) );
    pthread_create(&h5, NULL, func2, (void*) (&s11) );

    pthread_join(h1, NULL);
    pthread_join(h2, NULL);
    pthread_join(h3, NULL);
    pthread_join(h4, NULL);
    pthread_join(h5, NULL);
}
