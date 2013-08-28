#ifndef TEST_KERNEL_H
#define TEST_KERNEL_H

void test_kernel(double **value) {
        **value = 1.890567;
}

void test_kernel2(double **value) {
        printf("%lf ", (double)**value);
}

#endif
