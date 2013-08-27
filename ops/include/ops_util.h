
#ifndef __OP_UTIL_H
#define __OP_UTIL_H

/*
 * op_util.h
 *
 * Header file for the utility functions used in op_util.c
 *
 * written by: Gihan R. Mudalige, (Started 23-08-2013)
 */



#ifdef __cplusplus
extern "C" {
#endif

void* xmalloc(size_t size);

void* xrealloc(void *ptr, size_t size);

int min(int array[], int size);

int binary_search(int a[], int value, int low, int high);

int linear_search(int a[], int value, int low, int high);

void quickSort(int arr[], int left, int right);

int removeDups(int a[], int array_size);

int file_exist(char const *filename);

#ifdef __cplusplus
}
#endif

#endif /* __OP_UTIL_H */


