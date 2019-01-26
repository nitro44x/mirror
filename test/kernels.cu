#include <mirror/simt_macros.hpp>
#include <mirror/simt_allocator.hpp>
#include <mirror/simt_vector.hpp>
#include <mirror/simt_serialization.hpp>
#include <mirror/simt_utilities.hpp>

#include <vector>
#include <numeric>
#include <string>
#include <map>

#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <map>

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
    simt_sync;
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
    printArray<<<1, 1>>>(simt_v.data(), simt_v.size());
    simt_sync;
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
    call_printVector<<<1, 1>>>(*simt_v_ptr);
    simt_sync;
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
    call_printVector_ref<<<1, 1>>>(*simt_v_ptr);
    simt_sync;
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
    call_printVector<<<1, 1>>>(*simt_v_ptr);
    simt_sync;
    delete simt_v_ptr;
    simt_sync;
    std::cout << std::endl;
}

void test5() {
    std::cout << "modify simt::containers::vector [object] on gpu" << std::endl;
    auto simt_v_ptr = new simt::containers::vector<double>;
    simt_v_ptr->resize(10);
    call_setTo<<<1, 1>>>(*simt_v_ptr, 123);
    simt_sync;
        std::cout << "cpu v = ";
    for (auto const& d : *simt_v_ptr)
        std::cout << d << " ";
    std::cout << std::endl;
    call_printVector<<<1, 1>>>(*simt_v_ptr);
    simt_sync;
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
    call_printVector<<<1, 1>>>(*simt_v_ptr);
    simt_sync;
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

template<typename T>
__global__
void sayHi(simt::containers::vector<T*> & device_objs) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
        //printf("Saying hi from an A* \n");
        device_objs[tid]->sayHi();
    }
}

template<typename T>
__global__
void sayHi_poly(simt::serialization::polymorphic_mirror<T> & device_objs) {
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
    allocateDeviceObjs<<<nBlocks, nThreadsPerBlock>>>(*device_objs, *encoded_objs);
    simt_sync;
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

    sayHi<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;
        deallocateDeviceObjs<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;

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
    std::cout << "gpu sizeof(A) = " << simt::utilities::getDeviceSize<A>() << std::endl;
    std::cout << "cpu sizeof(B) = " << sizeof(B) << std::endl;
    std::cout << "gpu sizeof(B) = " << simt::utilities::getDeviceSize<B>() << std::endl;
    std::cout << "cpu sizeof(C) = " << sizeof(C) << std::endl;
    std::cout << "gpu sizeof(C) = " << simt::utilities::getDeviceSize<C>() << std::endl;
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

template <typename T>
__global__
void destructDeviceObjs_test(simt::containers::vector<T*> & device_objs) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
        //printf("Deallocating an A*\n");
        device_objs[tid]->~T();
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
    auto const sizeofB = simt::utilities::getDeviceSize<B>();
    auto const sizeofC = simt::utilities::getDeviceSize<C>();
    for (size_t i = 0; i < encoded_objs->size(); ++i)
        cudaMallocManaged((void**)&(*device_objs)[i], (*encoded_objs)[i].type == ABC_t::B ? sizeofB : sizeofC);

    constructDeviceObjs<<<nBlocks, nThreadsPerBlock>>>(*device_objs, *encoded_objs);
    simt_sync;


    sayHi<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;
    destructDeviceObjs_test<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;


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
    auto const sizeofB = simt::utilities::getDeviceSize<B>();
    auto const sizeofC = simt::utilities::getDeviceSize<C>();

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

    auto tank = new simt::containers::vector<char, simt::memory::device_allocator<char>, simt::memory::OverloadNewType::eHostOnly>(totalSpaceNeeded_bytes, '\0');

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
        if (i < 3 || i + 3 > device_objs->size())
            std::cout << "A[" << i << "] = " << (*device_objs)[i] << std::endl;
    }

    auto const tankStart = &(*(tank->begin()));
    auto const tankEnd = &(*(--(tank->end()))) + sizeof(char);
    std::cout << "tank start = " << (void*)tankStart << std::endl;
    std::cout << "tank end   = " << (void*)tankEnd << std::endl;

    constructDeviceObjs<<<nBlocks, nThreadsPerBlock>>>(*device_objs, *encoded_objs);
    simt_sync;

    for (size_t i = 0; i < 100; ++i)
        sayHi<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    std::cout << "Launched a bunch of sayHi's" << std::endl;
    simt_sync;
        destructDeviceObjs_test<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;


        delete encoded_objs;
    delete tank;
    delete device_objs;

    for (auto o : host_objs)
        delete o;
}

void test12() {
    // Device only pointers to vectors don't really make sense since they must be malloced on teh host side.

    // managed data
    auto managed_managedData_v = new simt::containers::vector<int>(4);
    auto hostOnly_managedData_v = new simt::containers::vector<int, simt::memory::managed_allocator<int>, simt::memory::OverloadNewType::eHostOnly>(4);

    // device data
    auto managed_DeviceOnlyData_v = new simt::containers::vector<int, simt::memory::device_allocator<int>, simt::memory::OverloadNewType::eManaged>(4);
    auto hostOnly_DeviceOnlyData_v = new simt::containers::vector<int, simt::memory::device_allocator<int>, simt::memory::OverloadNewType::eHostOnly>(4);

    // host data
    auto managed_HostOnlyData_v = new simt::containers::vector<int, std::allocator<int>, simt::memory::OverloadNewType::eManaged>(4);
    auto hostOnly_HostOnlyData_v = new simt::containers::vector<int, std::allocator<int>, simt::memory::OverloadNewType::eHostOnly>(4);

    delete managed_managedData_v;
    delete hostOnly_managedData_v;
    delete managed_DeviceOnlyData_v;
    delete managed_HostOnlyData_v;
    delete hostOnly_DeviceOnlyData_v;
    delete hostOnly_HostOnlyData_v;
}

void test13() {
    // Device only pointers to vectors don't really make sense since they must be malloced on teh host side.

    // managed data
    auto managed_managedData_v = new simt::containers::vector<int>(4, 1);
    auto hostOnly_managedData_v = new simt::containers::vector<int, simt::memory::managed_allocator<int>, simt::memory::OverloadNewType::eHostOnly>(4, 1);

    // device data
    auto managed_DeviceOnlyData_v = new simt::containers::vector<int, simt::memory::device_allocator<int>, simt::memory::OverloadNewType::eManaged>(4, 1);
    auto hostOnly_DeviceOnlyData_v = new simt::containers::vector<int, simt::memory::device_allocator<int>, simt::memory::OverloadNewType::eHostOnly>(4, 1);

    // host data
    auto managed_HostOnlyData_v = new simt::containers::vector<int, std::allocator<int>, simt::memory::OverloadNewType::eManaged>(4, 1);
    auto hostOnly_HostOnlyData_v = new simt::containers::vector<int, std::allocator<int>, simt::memory::OverloadNewType::eHostOnly>(4, 1);

    delete managed_managedData_v;
    delete hostOnly_managedData_v;
    delete managed_DeviceOnlyData_v;
    delete managed_HostOnlyData_v;
    delete hostOnly_DeviceOnlyData_v;
    delete hostOnly_HostOnlyData_v;
}

template <typename T>
void message(std::string prefix, T const& v, bool shouldPrint) {
    if (!shouldPrint) {
        std::cout << "Skipping printing because vector is device backed" << std::endl;
        return;
    }

    if (v.memory_type != simt::memory::OverloadNewType::eDeviceOnly) {
        std::cout << prefix << " :: v = ";
        for (auto const& d : v)
            std::cout << d << " ";
        std::cout << std::endl;
    }
}

template <typename T>
void test_copy_move_stuff(bool shouldPrint = true) {
    T managedNew_managedData(4, 1);
    managedNew_managedData.push_back(10);
    message("Original", managedNew_managedData, shouldPrint);
    T managedNew_managedData_copy(managedNew_managedData);
    message("Copy Constructor", managedNew_managedData_copy, shouldPrint);
    std::cout << std::endl;

    // Can't do std::iota() since this could be a device backed vector
    // std::iota(managedNew_managedData.begin(), managedNew_managedData.end(), -4);
    managedNew_managedData = T(5, 2);
    message("Original", managedNew_managedData, shouldPrint);
    managedNew_managedData_copy = managedNew_managedData;
    message("Copy Assigned", managedNew_managedData_copy, shouldPrint);
    std::cout << std::endl;

    message("Original", managedNew_managedData, shouldPrint);
    T managedNew_managedData_move(std::move(managedNew_managedData));
    message("Move Constructed", managedNew_managedData_move, shouldPrint);
    std::cout << std::endl;

    T managedNew_managedData_moveAssign{};
    managedNew_managedData_copy.push_back(123);
    managedNew_managedData_copy.push_back(321);
    message("Original", managedNew_managedData_copy, shouldPrint);
    managedNew_managedData_moveAssign = std::move(managedNew_managedData_copy);
    message("Move Assigned", managedNew_managedData_moveAssign, shouldPrint);
}

void test14() {
    test_copy_move_stuff<simt::containers::vector<int, simt::memory::managed_allocator<int>, simt::memory::OverloadNewType::eManaged>>();
    test_copy_move_stuff<simt::containers::vector<int, simt::memory::managed_allocator<int>, simt::memory::OverloadNewType::eHostOnly>>();

    test_copy_move_stuff<simt::containers::vector<int, simt::memory::device_allocator<int>, simt::memory::OverloadNewType::eManaged>>(false);
    test_copy_move_stuff<simt::containers::vector<int, simt::memory::device_allocator<int>, simt::memory::OverloadNewType::eHostOnly>>(false);

    test_copy_move_stuff<simt::containers::vector<int, std::allocator<int>, simt::memory::OverloadNewType::eManaged>>();
    test_copy_move_stuff<simt::containers::vector<int, std::allocator<int>, simt::memory::OverloadNewType::eHostOnly>>();
}

void test15() {
    simt::memory::MaybeOwner<simt::containers::vector<double>> v1(new simt::containers::vector<double>(5, 1));
    call_printVector<<<1, 1>>>(*v1);
    simt_sync;
    simt::memory::MaybeOwner<simt::containers::vector<double>> v2(v1.get(), false);
    call_printVector<<<1, 1>>>(*v2);
    simt_sync;
    std::iota(v1->begin(), v1->end(), -10);
    call_printVector<<<1, 1>>>(*v1);
    simt_sync;
    call_printVector<<<1, 1>>>(*v2);
    simt_sync;

    simt::memory::MaybeOwner<simt::containers::vector<double>> v3(std::move(v1));
    call_printVector<<<1, 1>>>(*v3);
    simt_sync;
    auto v4 = std::move(v3);
    call_printVector<<<1, 1>>>(*v4);
    simt_sync;
    call_printVector<<<1, 1>>>(*v2);
    simt_sync;
}

void test16() {
    simt::serialization::serializer io;
    double in_d = 12.234;
    std::cout << "in: " << in_d << std::endl;
    io.write(in_d);

    in_d = 21.999;
    std::cout << "in: " << in_d << std::endl;
    io.write(in_d);

    int in_i = 5;
    std::cout << "in: " << in_i << std::endl;
    io.write(in_i);
    
    in_d = 1.532e32;
    std::cout << "in: " << in_d << std::endl;
    io.write(in_d);

    double * in_dp = &in_d;
    std::cout << "in: " << in_dp << std::endl;
    io.write(in_dp);

    char in_c = 'M';
    std::cout << "in: " << in_c << std::endl;
    io.write(in_c);

    double out_d;
    int out_i;
    double * out_dp;
    char out_c;

    std::cout << std::endl;

    io.read(&out_d);
    std::cout << "out: " << out_d << std::endl;
    io.read(&out_d);
    std::cout << "out: " << out_d << std::endl;
    io.read(&out_i);
    std::cout << "out: " << out_i << std::endl;
    io.read(&out_d);
    std::cout << "out: " << out_d << std::endl;
    io.read(&out_dp);
    std::cout << "out: " << out_dp << std::endl;
    io.read(&out_c);
    std::cout << "out: " << out_c << std::endl;

    std::cout << std::endl << "Read them again!" << std::endl;
    io.seek(simt::serialization::Position::Beginning);
    io.read(&out_d);
    std::cout << "out: " << out_d << std::endl;
    io.read(&out_d);
    std::cout << "out: " << out_d << std::endl;
    io.read(&out_i);
    std::cout << "out: " << out_i << std::endl;
    io.read(&out_d);
    std::cout << "out: " << out_d << std::endl;
    io.read(&out_dp);
    std::cout << "out: " << out_dp << std::endl;
    io.read(&out_c);
    std::cout << "out: " << out_c << std::endl;
}

struct test17_a {
    int i = -42;
    char c = 'C';
};

struct test17_b {
    double d = 123.321;
    size_t s = 98765;
    float3 f3 = { 4,3,2 };
};

void test17() {
    simt::serialization::serializer io;
    test17_a a;
    double a_d_in = 998877.66;
    test17_b b;
    float4 b_f4_in = { 99, 88, 77, 66.123f };

    std::cout << ":IN:" << std::endl << std::endl;
    std::cout << "obj A:" << std::endl;
    std::cout << "  test17_a: " << a.i << " \"" << a.c << "\"" << std::endl;
    std::cout << "  a_d_in: " << a_d_in << std::endl;

    std::cout << "obj B:" << std::endl;
    std::cout << "  test17_b: " << b.d << " " << b.s << " "
        << b.f3.x << " " << b.f3.y << " " << b.f3.z << std::endl;
    std::cout << "  b_f4_in: " << b_f4_in.x << " " << b_f4_in.y << " " << b_f4_in.z << " " << b_f4_in.w << std::endl;

    auto const objStart_a = io.mark();
    io.write(a);
    io.write(a_d_in);
    auto const objStart_b = io.mark();
    io.write(b);
    io.write(b_f4_in);

    memset(&a, 0, sizeof(a));
    double a_d_out = 0;
    memset(&b, 0, sizeof(b));
    float4 b_f4_out = { 0, 0, 0, 0 };

    std::cout << ":CLEAR:" << std::endl << std::endl;
    std::cout << "obj A:" << std::endl;
    std::cout << "  test17_a: " << a.i << " \"" << a.c << "\"" << std::endl;
    std::cout << "  a_d_out: " << a_d_out << std::endl;

    std::cout << "obj B:" << std::endl;
    std::cout << "  test17_b: " << b.d << " " << b.s << " "
        << b.f3.x << " " << b.f3.y << " " << b.f3.z << std::endl;
    std::cout << "  b_f4_out: " << b_f4_out.x << " " << b_f4_out.y << " " << b_f4_out.z << " " << b_f4_out.w << std::endl;

    auto start = io.mark_position(1);
    io.read(start, &b);
    io.read(start, &b_f4_out);

    start = io.mark_position(0);
    io.read(start, &a);
    io.read(start, &a_d_out);

    std::cout << ":OUT:" << std::endl << std::endl;
    std::cout << "obj A:" << std::endl;
    std::cout << "  test17_a: " << a.i << " \"" << a.c << "\"" << std::endl;
    std::cout << "  a_d_out: " << a_d_out << std::endl;

    std::cout << "obj B:" << std::endl;
    std::cout << "  test17_b: " << b.d << " " << b.s << " "
        << b.f3.x << " " << b.f3.y << " " << b.f3.z << std::endl;
    std::cout << "  b_f4_out: " << b_f4_out.x << " " << b_f4_out.y << " " << b_f4_out.z << " " << b_f4_out.w << std::endl;
}


struct alignas(32) test18_a {
    double d;
    size_t s;
    float3 f3;
};

struct alignas(32) test18_b {
    int i;
    char c;
};

__global__ void test18_kernel(simt::serialization::serializer const& io) {
    auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        test18_a a;
        test18_b b;
        double a_d_out;
        float4 b_f4_out;

        auto start = io.mark_position(1);
        io.read(start, &b);
        io.read(start, &b_f4_out);

        start = io.mark_position(0);
        io.read(start, &a);
        io.read(start, &a_d_out);

        printf(":OUT GPU side:\n");
        printf("obj B: \n");
        printf("  test18_b: %d %c\n", b.i, b.c);
        printf("  b_f4_in: %f %f %f %f\n", b_f4_out.x, b_f4_out.y, b_f4_out.z, b_f4_out.w);

        printf("obj A: \n");
        printf("  test18_a: %lf %u %f %f %f\n", a.d, a.s, a.f3.x, a.f3.y, a.f3.z);
        printf("  a_d_out: %lf\n", a_d_out);

    }
}

void test18() {
    auto io = new simt::serialization::serializer;
    test18_a a;
    a.d = 1.23;
    a.s = 123;
    a.f3 = { 1, 2, 3 };
    double a_d_in = 998877.66;

    test18_b b;
    b.i = 9;
    b.c = '?';
    float4 b_f4_in = { 99, 88, 77, 66.123f };

    std::cout << ":IN:" << std::endl << std::endl;
    std::cout << "obj B:" << std::endl;
    std::cout << "  test18_b: " << b.i << " \"" << b.c << "\"" << std::endl;
    std::cout << "  b_f4_in: " << b_f4_in.x << " " << b_f4_in.y << " " << b_f4_in.z << " " << b_f4_in.w << std::endl;

    std::cout << "obj A:" << std::endl;
    std::cout << "  test18_a: " << a.d << " " << a.s << " "
        << a.f3.x << " " << a.f3.y << " " << a.f3.z << std::endl;
    std::cout << "  a_d_in: " << a_d_in << std::endl;

    io->mark();
    io->write(a);
    io->write(a_d_in);
    io->mark();
    io->write(b);
    io->write(b_f4_in);

    test18_kernel<<<1,1>>>(*io);
    simt_sync;
    delete io;
}

#define TEST19_ABSTRACT_TYPES \
        ENTRY(eBase, Base)

#define TEST19_CONCRETE_TYPES \
        ENTRY(eDerived1, Derived1) \
        ENTRY(eDerived2, Derived2) \
        ENTRY(eDerived1_2, Derived1_2) \

#define TEST19_TYPES \
        TEST19_ABSTRACT_TYPES \
        TEST19_CONCRETE_TYPES

enum class Test19Types {
    //eBase = 0,
    //eDerived1,
    //eDerived2,
    //eDerived1_2
#define ENTRY(a, b) a,
    TEST19_TYPES
#undef ENTRY
    Max_
};

class Base : public simt::serialization::Serializable<Test19Types> {
public:
    HOSTDEVICE virtual ~Base() { ; }

    HOSTDEVICE virtual void sayHi() = 0;
};

class Derived1 : public Base {
public:
    HOSTDEVICE Derived1() { ; }
    HOSTDEVICE Derived1(int j) : j(j) {}
    HOSTDEVICE ~Derived1() override { ; }

    HOSTDEVICE void sayHi() override {
        printf("Hello from Derived1, j = %d\n", j);
        ++j;
    }

    HOST void write(simt::serialization::serializer & io) const override {
        printf("Writing in Derived1 %d\n", j);
        io.write(j);
    }

    HOSTDEVICE void read(simt::serialization::serializer::size_type startPosition, 
                         simt::serialization::serializer & io) override {
        io.read(startPosition, &j);
        printf("[%u] Reading in Derived1 %d\n", simt::utilities::getTID(), j);
    }

    HOSTDEVICE type_id_t type() const override {
        return Test19Types::eDerived1;
    }

private:
    int j = 0;
};

class Derived2 : public Base {
public:
    HOSTDEVICE Derived2() { ; }
    HOSTDEVICE Derived2(int j) : d(j) {}
    HOSTDEVICE ~Derived2() override { ; }

    HOSTDEVICE void sayHi() override {
        printf("Hello from Derived2, d = %lf\n", d);
        ++d;
    }

    HOST void write(simt::serialization::serializer & io) const override {
        printf("Writing in Derived2 %lf\n", d);
        io.write(d);
    }

    HOSTDEVICE void read(simt::serialization::serializer::size_type startPosition,
        simt::serialization::serializer & io) override {
        io.read(startPosition, &d);
        printf("[%u] Reading in Derived2 %lf\n", simt::utilities::getTID(), d);
    }


    HOSTDEVICE type_id_t type() const override {
        return Test19Types::eDerived2;
    }

private:
    double d = 0;
};

class Derived1_2 : public Derived1 {
public:
    HOSTDEVICE Derived1_2() { ; }

    HOST Derived1_2(size_t j, double s) : Derived1(int(j)+1), v(new simt::containers::vector<double>(j)) {
        std::iota(v->begin(), v->end(), s);
    }

    HOSTDEVICE ~Derived1_2() override { ; }

    HOSTDEVICE void sayHi() override {
        printf("Hello from Derived1_2:\n ");
        if (!v) {
            printf("  No Data");
        }
        else {
            printf("  v = ");
            for (auto & d : *v) {
                printf("%lf ", d);
                ++d;
            }
            printf("\n");
        }
        printf("SayHi in Derived1: \n,");
        Derived1::sayHi();
    }

    HOST void write(simt::serialization::serializer & io) const override {
        printf("Writing in Derived1_2 %p\n", v.get());
        io.write(v.get());
        Derived1::write(io);
    }

    HOSTDEVICE void read(simt::serialization::serializer::size_type startPosition,
        simt::serialization::serializer & io) override {
        simt::containers::vector<double> * p = nullptr;
        io.read(startPosition, &p);
        v.setData(p, false);
        printf("[%u] Reading in Derived1_2 %p\n", simt::utilities::getTID(), v.get());
        Derived1::read(startPosition, io);
    }

    HOSTDEVICE type_id_t type() const override {
        return Test19Types::eDerived1_2;
    }


private:
    simt::memory::MaybeOwner<simt::containers::vector<double>> v;
};

template <Test19Types type>
struct type_getter {};

#define ENTRY(a, b) \
template<> \
struct type_getter<Test19Types::a> { \
    using type = b; \
};
TEST19_TYPES
#undef ENTRY

HOST Base* create_obj_test19(size_t obj_idx, simt::serialization::serializer & io) {
    auto startPosition = io.mark_position(obj_idx);
    Test19Types type;
    io.read(startPosition, &type);
    Base * p = nullptr;
    switch (type) {
#define ENTRY(a, b) \
    case Test19Types::a: \
        p = new b; \
        break;
    TEST19_CONCRETE_TYPES
#undef ENTRY
#define ENTRY(a, b) \
    case Test19Types::a:
        TEST19_ABSTRACT_TYPES
#undef ENTRY
    default:
        throw;
    }

    p->read(startPosition, io);
    return p;
}

// From a vector of encodedObjs, construct a vector of the polymorphic type on the device
void test19() {
    const auto N = 5;
    std::vector<Base*> host_objs;
    for (auto i = 0; i < N; ++i) {
        host_objs.push_back(new Derived1(i));
        host_objs.push_back(new Derived2(i));
        host_objs.push_back(new Derived1_2(5, i));
    }

    simt::serialization::serializer io;
    for (auto obj : host_objs) {
        io.mark();
        io.write(obj->type());
        obj->write(io);
    }

    std::vector<Base*> new_host_objs;
    for (size_t i = 0; i < io.number_of_marks(); ++i) {
        auto p = create_obj_test19(i, io);
        new_host_objs.push_back(p);
    }

    for (auto p : new_host_objs) {
        delete p;
    }

    for (auto p : host_objs) {
        delete p;
    }
}

template <>
struct simt::serialization::polymorphic_traits<Base> {
    using size_type = std::size_t;
    using pointer = Base*;
    using enum_type = Test19Types;

    static std::map<Test19Types, size_t> cache;

    static HOST size_type sizeOf(pointer p) {
        auto const res = cache.find(p->type());
        if (res != end(cache))
            return res->second;

        switch(p->type()) {
#define ENTRY(a, b) \
        case Test19Types::a: \
            cache[Test19Types::a] = simt::utilities::getDeviceSize<type_getter<Test19Types::a>::type>(); \
            return cache[Test19Types::a];
        TEST19_CONCRETE_TYPES
#undef ENTRY
        case Test19Types::Max_:
        default:
            throw;
        }
    }

    static HOSTDEVICE void create(simt::containers::vector<Base*> & device_objs, simt::serialization::serializer & io) {
        auto tid = threadIdx.x + blockIdx.x * blockDim.x;

        for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
            auto startPosition = io.mark_position(tid);
            Test19Types type;
            io.read(startPosition, &type);

            switch (type) {

        #define ENTRY(a, b) \
        case Test19Types::a: \
            new(device_objs[tid]) b; \
            device_objs[tid]->read(startPosition, io); \
            break;
                TEST19_CONCRETE_TYPES
        #undef ENTRY

            default:
                printf("Error allocating object!\n");
            }

            if (nullptr == device_objs[tid])
                printf("failed allocation at tid = %u\n", tid);
        }
    }
};

std::map<Test19Types, size_t> simt::serialization::polymorphic_traits<Base>::cache{};


template <typename BaseClass>
__global__
void constructDeviceTest19Objs(simt::containers::vector<BaseClass*> & device_objs, simt::serialization::serializer & io) {
    simt::serialization::polymorphic_traits<BaseClass>::create(device_objs, io);
}

void test20() {
    const auto N = 2;
    std::vector<Base*> host_objs;
    for (auto i = 0; i < N; ++i) {
        host_objs.push_back(new Derived1(i));
        host_objs.push_back(new Derived2(i));
        host_objs.push_back(new Derived1_2(5, i));
    }

    simt::memory::MaybeOwner<simt::serialization::serializer> io(new simt::serialization::serializer);

    for (auto obj : host_objs) {
        io->mark();
        io->write(obj->type());
        obj->write(*io);
    }

    std::vector<Base*> new_host_objs;
    for (size_t i = 0; i < io->number_of_marks(); ++i) {
        auto p = create_obj_test19(i, *io);
        new_host_objs.push_back(p);
    }

    auto sizeofFold = [](size_t currentTotal, Base * p) {
        return currentTotal + simt::serialization::polymorphic_traits<Base>::sizeOf(p);
    };

    auto totalSpaceNeeded_bytes = std::accumulate(host_objs.begin(), host_objs.end(), size_t(0), sizeofFold);

    auto tank = new simt::containers::vector<char, simt::memory::device_allocator<char>, simt::memory::OverloadNewType::eHostOnly>(totalSpaceNeeded_bytes, '\0');

    std::cout << "               Tank setup" << std::endl;
    std::cout << "--------------------------" << std::endl;

    simt::memory::MaybeOwner<simt::containers::vector<Base*>> device_objs(new simt::containers::vector<Base*>(host_objs.size(), nullptr));
    size_t offset = 0;
    for (size_t i = 0; i < host_objs.size(); ++i) {

        (*device_objs)[i] = (Base*)(tank->data() + offset);

        switch (host_objs[i]->type()) {
#define ENTRY(a, b) \
        case Test19Types::a: \
            offset += simt::utilities::getDeviceSize<type_getter<Test19Types::a>::type>(); \
            break;
            TEST19_CONCRETE_TYPES
#undef ENTRY
        default:
            assert(false);
        }
    }

    auto const nBlocks = 128;
    auto const nThreadsPerBlock = 128;
    constructDeviceTest19Objs<<<nBlocks, nThreadsPerBlock>>>(*device_objs, *io);
    simt_sync;

    sayHi<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;

    destructDeviceObjs_test<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;

    for (auto p : new_host_objs) {
        delete p;
    }

    for (auto p : host_objs) {
        delete p;
    }
}

void test21() {
    const auto N = 20;
    std::vector<Base*> host_objs;
    for (auto i = 0; i < N; ++i) {
        host_objs.push_back(new Derived1(i));
        host_objs.push_back(new Derived2(i));
        host_objs.push_back(new Derived1_2(5, i));
    }

    simt::serialization::polymorphic_mirror<Base> device_objs(host_objs);

    auto const nBlocks = 128;
    auto const nThreadsPerBlock = 128;
    sayHi<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;
}