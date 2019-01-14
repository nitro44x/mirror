#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <numeric>

#include "simt_vector.hpp"
#include "simt_allocator.hpp"

template <typename T>
__global__ void printArray(T const* data, size_t size) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("gpu v = ");
		for (int i = 0; i < size; ++i)
			printf("%lf ", data[i]);
		printf("\n");
	}
}

HOSTDEVICE void printVector(simt::containers::vector<double> const & v) {
	printf("gpu v = ");
	for (auto const& d : v)
		printf("%lf ", d);
	printf("\n");
}

__global__ void call_printVector(simt::containers::vector<double> const& v) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printVector(v);
	}
}

__global__ void call_printVector_ref(simt::containers::vector<double> const& v) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printVector(v);
	}
}


HOSTDEVICE void setTo(simt::containers::vector<double> & v, simt::containers::vector<double>::value_type value) {
	for (auto & d : v)
		d = value;
}

__global__ void call_setTo(simt::containers::vector<double> & v, simt::containers::vector<double>::value_type value) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		setTo(v, value);
	}
}

void test1() {
	std::cout << "std::vector" << std::endl;
	std::vector<double, simt::memory::managed_allocator<double>> v(10);
	std::iota(begin(v), end(v), -4);
	std::cout << "cpu v = ";
	for (auto const& d : v)
		std::cout << d << " ";
	std::cout << std::endl;

	printArray<<<1, 1>>>(v.data(), v.size());
	cudaDeviceSynchronize();
	std::cout << std::endl;
}

void test2() {
	std::cout << "simt::containers::vector [raw ptr]" << std::endl;
	simt::containers::vector<double> simt_v(10, 3.0);
	std::iota(simt_v.begin(), simt_v.end(), -3.0);
	simt_v.push_back(4321);
	std::cout << "cpu v = ";
	for (auto const& d : simt_v)
		std::cout << d << " ";
	std::cout << std::endl;
	printArray<<<1,1>>>(simt_v.data(), simt_v.size());
	cudaDeviceSynchronize();
	std::cout << std::endl;
}

void test3() {
	std::cout << "simt::containers::vector [object]" << std::endl;
	auto simt_v_ptr = new simt::containers::vector<double>(10);
	std::iota(simt_v_ptr->begin(), simt_v_ptr->end(), -4);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	printVector(*simt_v_ptr);
	call_printVector<<<1,1>>>(*simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test3a() {
	std::cout << "simt::containers::vector [object] printByRef" << std::endl;
	auto simt_v_ptr = new simt::containers::vector<double>(10);
	std::iota(simt_v_ptr->begin(), simt_v_ptr->end(), -4);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	printVector(*simt_v_ptr);
	call_printVector_ref<<<1,1>>>(*simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test4() {
	std::cout << "modify simt::containers::vector [object] on cpu" << std::endl;
	auto simt_v_ptr = new simt::containers::vector<double>;
	simt_v_ptr->resize(10);
	setTo(*simt_v_ptr, 123);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	call_printVector<<<1,1>>>(*simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test5() {
	std::cout << "modify simt::containers::vector [object] on gpu" << std::endl;
	auto simt_v_ptr = new simt::containers::vector<double>;
	simt_v_ptr->resize(10);
	call_setTo<<<1,1>>>(*simt_v_ptr, 123);
	cudaDeviceSynchronize();
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	call_printVector<<<1,1>>>(*simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test6() {
	std::cout << "modify simt::containers::vector [object] on gpu" << std::endl;
	auto simt_v_ptr = new simt::containers::vector<double>;
	for (auto i = 0; i < 4; ++i)
		simt_v_ptr->push_back(10+i*i);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	call_printVector<<<1,1>>>(*simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

int main() {
	// Print vector tests
	test1();
	test2();
	test3();
	test3a();

	// Modify vector tests
	test4();
	test5();
	test6();

    return EXIT_SUCCESS;
}