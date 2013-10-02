#ifndef TEST_KERNEL_H
#define TEST_KERNEL_H

void test_kernel(double **value) {
        **value = 1.890567;
}

void test_kernel2(double **value) {
        printf("%lf ", (double)**value);
}

void test_kernel3(double **value, double **value_d) {
        printf("%lf %lf\n", (double)**value, (double)**value_d);
}

#endif
