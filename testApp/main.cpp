#include "kernels.cuh"

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
	test1();
	test2();
	test3();
	test3a();

	// Modify vector tests
	test4();
	test5();
	test6();

	// Polymorphic classes Classes
	test7();
	test9();
	//test10();
	test11();

	// Thrust
	test8();
	test8_cpp();

    return 0;
}