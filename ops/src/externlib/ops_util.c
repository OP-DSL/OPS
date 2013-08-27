/*
 * ops_util.c
 *
 * Some utility functions for the OPS
 *
 * written by: Gihan R. Mudalige, (Started 23-08-2013)
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

#include <ops_util.h>

/*******************************************************************************
* Wrapper for malloc from www.gnu.org/
*******************************************************************************/

void* xmalloc (size_t size)
{
  if(size == 0) return (void *)NULL;

  register void *value = malloc (size);
  if (value == 0) printf("Virtual memory exhausted at malloc\n");
  return value;
}

/*******************************************************************************
* Wrapper for realloc from www.gnu.org/
*******************************************************************************/

void* xrealloc (void *ptr, size_t size)
{
  if(size == 0)
  {
    free(ptr);
    return (void *)NULL;
  }

  register void *value = realloc (ptr, size);
  if (value == 0) printf ("Virtual memory exhausted at realloc\n");
  return value;
}

/*******************************************************************************
* Return the index of the min value in an array
*******************************************************************************/

int min(int array[], int size)
{
  int min = INT_MAX;
  int index = -1;
  for(int i=0; i<size; i++)
  {
    if(array[i]<min)
    {
      index = i;
      min = array[i];
    }
  }
  return index;
}

/*******************************************************************************
* Binary search an array for a given value
*******************************************************************************/

int binary_search(int a[], int value, int low, int high)
{
  if (high < low)
    return -1; // not found

  int mid = low + (high - low) / 2;
  if (a[mid] > value)
    return binary_search(a, value, low, mid-1);
  else if (a[mid] < value)
    return binary_search(a, value, mid+1, high);
  else
    return mid; // found
}

/*******************************************************************************
* Linear search an array for a given value
*******************************************************************************/

int linear_search(int a[], int value, int low, int high)
{
  for(int i = low; i<=high; i++)
  {
    if (a[i] == value) return i;
  }
  return -1;
}

/*******************************************************************************
* Quicksort an array
*******************************************************************************/

void quickSort(int arr[], int left, int right)
{
  int i = left;
  int j = right;
  int tmp;
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)i++;
    while (arr[j] > pivot)j--;
    if (i <= j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
      i++; j--;
    }
  };
  // recursion
  if (left < j)
    quickSort(arr, left, j);
  if (i < right)
    quickSort(arr, i, right);
}

/*******************************************************************************
* Remove duplicates in an array
*******************************************************************************/

int removeDups(int a[], int array_size)
{
  int i, j;
  j = 0;
  // Remove the duplicates ...
  for (i = 1; i < array_size; i++)
  {
    if (a[i] != a[j])
    {
      j++;
      a[j] = a[i]; // Move it to the front
    }
  }
  // The new array size..
  array_size = (j + 1);
  return array_size;
}

/*******************************************************************************
* Check if a file exists
*******************************************************************************/
int file_exist (char const *filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

