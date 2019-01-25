#include "kernels.cuh"
#include "kernels2.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void test8_cpp() {
    thrust::device_vector<double> d;
    d.resize(10);
    for (size_t i = 0; i < d.size(); ++i)
        d[i] = (double)i;

    thrust::host_vector<double> h = d;
    for (auto const& i : h)
        std::cout << i << " ";
    std::cout << std::endl;

}

#define RUN_TEST(name) \
          std::cout << std::endl; \
          std::cout << "--------------------------------------------------------" << std::endl; \
          std::cout << "           " << #name << std::endl; \
          std::cout << "--------------------------------------------------------" << std::endl; \
          name(); \
          std::cout << "--------------------------------------------------------" << std::endl; \
          std::cout << std::endl;

int main() {

    int count = 0;
    cudaGetDeviceCount(&count);
    std::cout << "Found " << count << " cuda devices." << std::endl;
    if (count == 0) {
        return 1;
    }
    //const size_t heapSpaceMB = 2*1024;
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * heapSpaceMB);

    size_t heapSize_bytes;
    cudaDeviceGetLimit(&heapSize_bytes, cudaLimitMallocHeapSize);
    std::cout << "Heap size in bytes = " << heapSize_bytes << std::endl;

    // Print vector tests
    RUN_TEST(test1);
    RUN_TEST(test2);
    RUN_TEST(test3);
    RUN_TEST(test3a);

    // Modify vector tests
    RUN_TEST(test4);
    RUN_TEST(test5);
    RUN_TEST(test6);

    // Polymorphic classes Classes
    RUN_TEST(test7);
    RUN_TEST(test9);
    RUN_TEST(test10);
    RUN_TEST(test11);

    // Thrust
    RUN_TEST(test8);
    RUN_TEST(test8_cpp);

    // Overloading news
    RUN_TEST(test12);
    RUN_TEST(test13);

    // Move/copy
    RUN_TEST(test14);

    // MaybeOwner
    RUN_TEST(test15);

    // Serialization
    RUN_TEST(test16);
    RUN_TEST(test17);
    RUN_TEST(test18);
    RUN_TEST(test19);
    RUN_TEST(test20);
    RUN_TEST(test21);
    RUN_TEST(simple_polymorphic_test);

    return 0;
}