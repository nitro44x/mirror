#include "simt_macros.hpp"
#include "simt_allocator.hpp"
#include "simt_vector.hpp"

#include <vector>
#include <numeric>

#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

	printArray<<<1,1>>>(v.data(), v.size());
	simt_sync
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
	simt_sync
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
	simt_sync
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
	simt_sync
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
	simt_sync
	delete simt_v_ptr;
	simt_sync
	std::cout << std::endl;
}

void test5() {
	std::cout << "modify simt::containers::vector [object] on gpu" << std::endl;
	auto simt_v_ptr = new simt::containers::vector<double>;
	simt_v_ptr->resize(10);
	call_setTo<<<1,1>>>(*simt_v_ptr, 123);
	simt_sync
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	call_printVector << <1, 1 >> > (*simt_v_ptr);
	simt_sync
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test6() {
	std::cout << "modify simt::containers::vector [object] on gpu" << std::endl;
	auto simt_v_ptr = new simt::containers::vector<double>;
	for (auto i = 0; i < 4; ++i)
		simt_v_ptr->push_back(10 + i * i);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	call_printVector<<<1,1>>>(*simt_v_ptr);
	simt_sync
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

	HOSTDEVICE virtual void sayHi() = 0;

	HOSTDEVICE virtual encodedObj encode() const = 0;
	HOSTDEVICE virtual void decode(encodedObj e) = 0;
	HOSTDEVICE virtual ABC_t type() const = 0;
};

class B : public A {
public:
	HOSTDEVICE B() { ; }
	HOSTDEVICE B(int j) : j(j) {}
	HOSTDEVICE B(encodedObj e) : B() { decode(e); }
	HOSTDEVICE ~B() override { ; }

	HOSTDEVICE void sayHi() override {
		//printf("Hello from B, j = %d\n", j);
		++j;
	}

	HOSTDEVICE virtual encodedObj encode() const {
		return { ABC_t::B, 0, j };
	}

	HOSTDEVICE virtual void decode(encodedObj e) {
		j = e.i;
	}

	HOSTDEVICE virtual ABC_t type() const {
		return ABC_t::B;
	}

	int j = 0;
};

class C : public A {
public:
	HOSTDEVICE C() { ; }
	HOSTDEVICE C(int j) : d(j) {}
	HOSTDEVICE C(encodedObj e) : C() { decode(e); }

	HOSTDEVICE ~C() override { ; }

	HOSTDEVICE void sayHi() override {
		//printf("Hello from C, d = %lf\n", d);
		++d;
	}

	HOSTDEVICE virtual encodedObj encode() const {
		return { ABC_t::C, d, 0 };
	}

	HOSTDEVICE virtual void decode(encodedObj e) {
		d = e.d;
	}

	HOSTDEVICE virtual ABC_t type() const {
		return ABC_t::C;
	}

	double d = 0;
};

__global__
void allocateDeviceObjs(simt::containers::vector<A*> & device_objs, simt::containers::vector<encodedObj> const& encoded_objs) {
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (; tid < encoded_objs.size(); tid += blockDim.x * gridDim.x) {
		switch (encoded_objs[tid].type) {
		case ABC_t::B:
			//printf("Allocating B object! %d\n", (int)tid);
			device_objs[tid] = new B(encoded_objs[tid]);
			break;
		case ABC_t::C:
			//printf("Allocating C object! %d\n", (int)tid);
			device_objs[tid] = new C(encoded_objs[tid]);
			break;
		case ABC_t::A:
		case ABC_t::Unk:
		default:
			printf("Error allocating object!\n");
		}

		if (nullptr == device_objs[tid])
			printf("failed allocation at tid = %u\n", tid);
	}
}

__global__
void sayHi(simt::containers::vector<A*> & device_objs) {
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
		//printf("Saying hi from an A* \n");
		device_objs[tid]->sayHi();
	}
}

__global__
void deallocateDeviceObjs(simt::containers::vector<A*> & device_objs) {
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
		//printf("Deallocating an A*\n");
		delete device_objs[tid];
	}
}

template <typename T>
__global__ void compute_sizeof(size_t * size) {
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid == 0)
		*size = sizeof(T);
}

template <typename T>
HOST size_t getDeviceSize() {
	size_t * size = nullptr;
	cudaMallocManaged((void**)&size, sizeof(size_t));
	compute_sizeof<T><<<1,1>>>(size);
	simt_sync;
	auto const result = *size;
	cudaFree(size);
	return result;
}

void test7() {
	const auto N = 10;
	std::vector<A*> host_objs;
	auto encoded_objs = new simt::containers::vector<encodedObj>();
	for (auto i = 0; i < N; ++i) {
		host_objs.push_back(new B(i));
		encoded_objs->push_back(host_objs.back()->encode());
	}

	for (auto i = 0; i < N; ++i) {
		host_objs.push_back(new C(i));
		encoded_objs->push_back(host_objs.back()->encode());
	}

	auto const nBlocks = 128;
	auto const nThreadsPerBlock = 128;
	auto device_objs = new simt::containers::vector<A*>(encoded_objs->size(), nullptr);
	allocateDeviceObjs<<<nBlocks,nThreadsPerBlock>>>(*device_objs, *encoded_objs);
	simt_sync
	delete encoded_objs;

	size_t nNulls = 0;
	for (auto const& p : *device_objs) {
		if (p == nullptr)
			++nNulls;
	}

	if (nNulls > 0) {
		std::cout << "Found " << nNulls << " nullptrs" << std::endl;
		return;
	}

	sayHi<<<nBlocks,nThreadsPerBlock>>>(*device_objs);
	simt_sync
	deallocateDeviceObjs<<<nBlocks,nThreadsPerBlock>>>(*device_objs);
	simt_sync

	delete device_objs;
	for (auto o : host_objs)
		delete o;
}

void test8() {
	thrust::device_vector<double> d;
	d.resize(10);
	for (size_t i = 0; i < d.size(); ++i)
		d[i] = (double)i;

	thrust::host_vector<double> h = d;
	for (auto const& i : h)
		std::cout << i << " ";
	std::cout << std::endl;

}

void test9() {
	std::cout << "cpu sizeof(A) = " << sizeof(A) << std::endl;
	std::cout << "gpu sizeof(A) = " << getDeviceSize<A>() << std::endl;
	std::cout << "cpu sizeof(B) = " << sizeof(B) << std::endl;
	std::cout << "gpu sizeof(B) = " << getDeviceSize<B>() << std::endl;
	std::cout << "cpu sizeof(C) = " << sizeof(C) << std::endl;
	std::cout << "gpu sizeof(C) = " << getDeviceSize<C>() << std::endl;
}

__global__
void constructDeviceObjs(simt::containers::vector<A*> & device_objs, simt::containers::vector<encodedObj> const& encoded_objs) {
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (; tid < encoded_objs.size(); tid += blockDim.x * gridDim.x) {

		switch (encoded_objs[tid].type) {
		case ABC_t::B:
			//printf("constructing B object at %p \n", device_objs[tid]);
			new(device_objs[tid]) B(encoded_objs[tid]);
			break;
		case ABC_t::C:
			//printf("constructing C object at %p \n", device_objs[tid]);
			new(device_objs[tid]) C(encoded_objs[tid]);
			break;
		case ABC_t::A:
		case ABC_t::Unk:
		default:
			printf("Error allocating object!\n");
		}

		if (nullptr == device_objs[tid])
			printf("failed allocation at tid = %u\n", tid);
	}
}

__global__
void destructDeviceObjs(simt::containers::vector<A*> & device_objs) {
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
		//printf("Deallocating an A*\n");
		device_objs[tid]->~A();
	}
}

void test10() {
	const auto N = 5;
	std::vector<A*> host_objs;
	auto encoded_objs = new simt::containers::vector<encodedObj>();
	for (auto i = 0; i < N; ++i) {
		host_objs.push_back(new B(i));
		encoded_objs->push_back(host_objs.back()->encode());
	}

	for (auto i = 0; i < N; ++i) {
		host_objs.push_back(new C(i));
		encoded_objs->push_back(host_objs.back()->encode());
	}

	auto const nBlocks = 128;
	auto const nThreadsPerBlock = 128;
	auto device_objs = new simt::containers::vector<A*>(encoded_objs->size(), nullptr);
	auto const sizeofB = getDeviceSize<B>();
	auto const sizeofC = getDeviceSize<C>();
	for(size_t i = 0; i < encoded_objs->size(); ++i)
		cudaMallocManaged((void**)&(*device_objs)[i], (*encoded_objs)[i].type == ABC_t::B ? sizeofB : sizeofC);

	constructDeviceObjs<<<nBlocks,nThreadsPerBlock>>>(*device_objs, *encoded_objs);
	simt_sync
	

	sayHi<<<nBlocks,nThreadsPerBlock>>>(*device_objs);
	simt_sync
	destructDeviceObjs<<<nBlocks,nThreadsPerBlock>>>(*device_objs);
	simt_sync


	delete encoded_objs;
	for (auto p : *device_objs)
		cudaFree(p);
	delete device_objs;

	for (auto o : host_objs)
		delete o;
}

void test11() {
	const auto N = 5;
	std::vector<A*> host_objs;
	auto encoded_objs = new simt::containers::vector<encodedObj>();
	for (auto i = 0; i < N; ++i) {
		host_objs.push_back(new B(i));
		encoded_objs->push_back(host_objs.back()->encode());
	}

	for (auto i = 0; i < N; ++i) {
		host_objs.push_back(new C(i));
		encoded_objs->push_back(host_objs.back()->encode());
	}

	auto const nBlocks = 128;
	auto const nThreadsPerBlock = 128;
	auto device_objs = new simt::containers::vector<A*>(encoded_objs->size(), nullptr);
	auto const sizeofB = getDeviceSize<B>();
	auto const sizeofC = getDeviceSize<C>();

	auto sizeofFold = [sizeofB, sizeofC](size_t currentTotal, encodedObj const& e) {
		switch (e.type) {
		case ABC_t::B:
			return currentTotal + sizeofB;
		case ABC_t::C:
			return currentTotal + sizeofC;
		default:
			assert(false);
			return size_t(0);
		}
	};

	auto totalSpaceNeeded_bytes = std::accumulate(encoded_objs->begin(), encoded_objs->end(), size_t(0), sizeofFold);
	std::cout << "total Space needed [bytes] = " << totalSpaceNeeded_bytes << std::endl;

	auto tank = new simt::containers::vector<char, simt::memory::device_allocator<char>>(totalSpaceNeeded_bytes, '\0');

	std::cout << "               Tank setup" << std::endl;
	std::cout << "--------------------------" << std::endl;

	size_t offset = 0;
	for (size_t i = 0; i < encoded_objs->size(); ++i) {
		
		(*device_objs)[i] = (A*)(tank->data() + offset);

		auto const& e = (*encoded_objs)[i];
		switch (e.type) {
		case ABC_t::B:
			offset += sizeofB;
			break;
		case ABC_t::C:
			offset += sizeofC;
			break;
		default:
			assert(false);
		}
	}

	for (size_t i = 0; i < device_objs->size(); ++i) {
		if(i < 3 || i+3 > device_objs->size())
			std::cout << "A[" << i << "] = " << (*device_objs)[i] << std::endl;
	}

	auto const tankStart = &(*(tank->begin()));
	auto const tankEnd = &(*(--(tank->end()))) + sizeof(char);
	std::cout << "tank start = " << (void*)tankStart << std::endl;
	std::cout << "tank end   = " << (void*)tankEnd << std::endl;

	constructDeviceObjs<<<nBlocks,nThreadsPerBlock>>>(*device_objs, *encoded_objs);
	simt_sync

	for(size_t i = 0; i < 10000; ++i)
		sayHi<<<nBlocks,nThreadsPerBlock>>>(*device_objs);
	std::cout << "Launched a bunch of sayHi's" << std::endl;
	simt_sync
	destructDeviceObjs<<<nBlocks,nThreadsPerBlock>>>(*device_objs);
	simt_sync


	delete encoded_objs;
	delete tank;
	delete device_objs;

	for (auto o : host_objs)
		delete o;
}

void test12() {

}
