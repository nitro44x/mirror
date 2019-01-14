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


enum class ABC_t { Unk, A, B, C };

struct encodedObj {
	ABC_t type;
	double d = 0;
	int i = 0;
};

class A {
public:
	HOSTDEVICE virtual ~A() { ; }

	HOSTDEVICE virtual void sayHi() const = 0;

	HOSTDEVICE virtual encodedObj encode() const = 0;
	HOSTDEVICE virtual void decode(encodedObj e) = 0;
};

class B : public A {
public:
	HOSTDEVICE B() { ; }
	HOSTDEVICE B(int j) : j(j) {}
	HOSTDEVICE B(encodedObj e) : B() { decode(e); }
	HOSTDEVICE ~B() override { ; }

	HOSTDEVICE void sayHi() const override {
		printf("Hello from B, j = %d\n", j);
	}

	HOSTDEVICE virtual encodedObj encode() const {
		return { ABC_t::B, 0, j };
	}

	HOSTDEVICE virtual void decode(encodedObj e) {
		j = e.i;
	}

	int j = 0;
};

class C : public A {
public:
	HOSTDEVICE C() { ; }
	HOSTDEVICE C(int j) : d(j) {}
	HOSTDEVICE C(encodedObj e) : C() { decode(e); }
	HOSTDEVICE ~C() override { ; }

	HOSTDEVICE void sayHi() const override {
		printf("Hello from C, d = %lf\n", d);
	}

	HOSTDEVICE virtual encodedObj encode() const {
		return { ABC_t::C, d, 0 };
	}

	HOSTDEVICE virtual void decode(encodedObj e) {
		d = e.d;
	}

	double d = 0;
};

__global__
void allocateDeviceObjs(simt::containers::vector<A*> & device_objs, simt::containers::vector<encodedObj> const& encoded_objs) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("device_objs size = %d\n", (int)device_objs.size());
		for (size_t i = 0; i < device_objs.size(); ++i) {
			switch (encoded_objs[i].type) {
			case ABC_t::B:
				printf("Allocating B object!\n");
				device_objs[i] = new B(encoded_objs[i]);
				break;
			case ABC_t::C:
				printf("Allocating C object!\n");
				device_objs[i] = new C(encoded_objs[i]);
				break;
			case ABC_t::A:
			case ABC_t::Unk:
			default:
				printf("Error allocating object!\n");
			}
		}
	}
}

__global__
void sayHi(simt::containers::vector<A*> & device_objs) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (size_t i = 0; i < device_objs.size(); ++i) {
			printf("Saying hi from an A* \n");
			device_objs[i]->sayHi();
		}
	}
}

__global__
void deallocateDeviceObjs(simt::containers::vector<A*> & device_objs) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (auto o : device_objs) {
			printf("Deallocating an A*\n");
			delete o;
		}
	}
}

void test7() {
	const auto N = 30;
	std::vector<A*> host_objs;
	auto encoded_objs = new simt::containers::vector<encodedObj>();
	for (auto i = 0; i < N; ++i) {
		host_objs.push_back(new B(i));
		encoded_objs->push_back(host_objs.back()->encode());
		host_objs.push_back(new C(i));
		encoded_objs->push_back(host_objs.back()->encode());
	}

	auto device_objs = new simt::containers::vector<A*>(2*N);
	allocateDeviceObjs<<<1,1>>>(*device_objs, *encoded_objs);
	check(cudaDeviceSynchronize());
	sayHi<<<1,1>>>(*device_objs);
	check(cudaDeviceSynchronize());
	deallocateDeviceObjs<<<1,1>>>(*device_objs);
	check(cudaDeviceSynchronize());

	delete encoded_objs;
	delete device_objs;
	for (auto o : host_objs)
		delete o;
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

	// Polymorphic classes Classes
	test7();

    return EXIT_SUCCESS;
}