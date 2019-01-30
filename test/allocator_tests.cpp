#include <catch/catch.hpp>

#include <mirror/simt_allocator.hpp>
#include <mirror/simt_utilities.hpp>

#include <cuda_runtime_api.h>

using namespace Catch;
using namespace Catch::literals;

__global__ void setArrayTo(double * data, size_t n, double value) {
    if (!mirror::getTID()) {
        for (auto i = 0u; i < n; ++i)
            data[i] = value;
    }
}

TEST_CASE("std::vector can use UMA allocator", "[allocator]") {
    std::vector<double, mirror::managed_allocator<double>> v;
    REQUIRE_NOTHROW(v.resize(10));

    setArrayTo<<<1,1>>>(v.data(), v.size(), 123);
    mirror_sync;

    for (auto const& value : v)
        REQUIRE(value == 123.0_a);
}


TEST_CASE("std::vector cannot use device-only allocator", "[allocator]") {
    /* 
    This code is left in as a demonstration of what you cannot do. MSVC uses the provided
    allocator to new a vector as well as the data underneath it (I think). It then uses a
    placement new construction of the vector, which explodes since the memory device side.

    std::vector<double, mirror::device_allocator<double>> * v;
    REQUIRE_THROWS(v = new std::vector<double, mirror::device_allocator<double>>());
    delete v;
    */
}


template <typename T>
struct someData : public T {
    double a;
    int b;
    char c;
};

template <typename T>
__global__ void setSomeData(someData<T> * data, char value) {
    if (!mirror::getTID()) {
        data->a = (double)value;
        data->b = (int)value;
        data->c = value;
    }
}

TEST_CASE("Overloading new/delete with UMA", "[overload_new_delete]") {
    auto data = new someData<mirror::Managed>;
    char value = 'A';
    setSomeData<<<1,1>>>(data, value);
    mirror_sync;

    REQUIRE(data->a == (double)value);
    REQUIRE(data->b == (int)value);
    REQUIRE(data->c == value);

    delete data;
}

TEST_CASE("Overloading new/delete with device-only", "[overload_new_delete]") {
    auto data = new someData<mirror::DeviceOnly>;
    char value = 'A';
    setSomeData<<<1,1>>>(data, value);
    mirror_sync;

    someData<mirror::HostOnly> data_host;

    // Actually a bit scary that this works...
    cudaMemcpy(&data_host, data, sizeof(someData<mirror::DeviceOnly>), cudaMemcpyDeviceToHost);

    REQUIRE(data_host.a == (double)value);
    REQUIRE(data_host.b == (int)value);
    REQUIRE(data_host.c == value);

    delete data;
}

TEST_CASE("MaybeOwner sematics work with standard containers", "[maybe_owner]") {
    using TestType = std::string;
    mirror::MaybeOwner<std::vector<TestType>> v(new std::vector<TestType>(5));

    REQUIRE(v->size() == 5);
    REQUIRE(v->capacity() >= 5);
    REQUIRE(v.isOwned() == true);

    SECTION("Move assignment of MaybeOwner moves the ownership.") {
        auto v2 = std::move(v);
        REQUIRE(v2.isOwned() == true);
        REQUIRE(v.isOwned() == false);
    }

    SECTION("Move constructor of MaybeOwner moves the ownership.") {
        mirror::MaybeOwner<std::vector<TestType>> v2(std::move(v));
        REQUIRE(v2.isOwned() == true);
        REQUIRE(v.isOwned() == false);
    }
}

static size_t nAlloc = 0;
static size_t nDealloc = 0;

void resetAllocCounts() {
    nAlloc = 0;
    nDealloc = 0;
}

template <typename T>
struct counting_allocator {

    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    using value_type = T;
    using pointer = T * ;
    using const_pointer = const T*;
    using reference = T & ;
    using const_reference = const T&;

    template< class U > struct rebind { typedef counting_allocator<U> other; };
    counting_allocator() = default;

    template <class U> constexpr counting_allocator(const counting_allocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        nAlloc++;
        return (T*)malloc(n * sizeof(T));
    }

    void deallocate(T* p, std::size_t) noexcept {
        nDealloc++;
        free(p);
    }
};

TEST_CASE("MaybeOwner deletes owned data", "[maybe_owner]") {
    using TestType = double;
    using TestArray = std::vector<TestType, counting_allocator<TestType>>;
    
    SECTION("MaybeOwner deallocates up data if it owns it") {
        resetAllocCounts();
        {
            mirror::MaybeOwner<TestArray> v(new TestArray(5));
        }
        REQUIRE(nAlloc == nDealloc);
        REQUIRE(nAlloc > 0);
    }

    SECTION("MaybeOwner does not deallocates up data if it doest not own it (move assignment)") {
        resetAllocCounts();
        size_t tmp_nAllocs;
        {
            mirror::MaybeOwner<TestArray> v2;
            {
                mirror::MaybeOwner<TestArray> v(new TestArray(5));
                tmp_nAllocs = nAlloc;
                resetAllocCounts();
                v2 = std::move(v);
            }
            REQUIRE(nAlloc == 0);
            REQUIRE(nDealloc == 0);
        }

        REQUIRE(tmp_nAllocs == nDealloc);
        REQUIRE(nDealloc > 0);
    }

    SECTION("MaybeOwner does not deallocates up data if it doest not own it (move constructor)") {
        resetAllocCounts();
        {
            mirror::MaybeOwner<TestArray> v(new TestArray(5));
            REQUIRE(nAlloc > 0);
            REQUIRE(nDealloc == 0);
            size_t tmp_alloc = nAlloc;
            size_t tmp_dealloc = nDealloc;
            mirror::MaybeOwner<TestArray> v2(std::move(v));
            REQUIRE(tmp_alloc == nAlloc);
            REQUIRE(tmp_dealloc == nDealloc);
        }
        REQUIRE(nAlloc == nDealloc);
        REQUIRE(nAlloc > 0);
    }

}

TEST_CASE("MaybeOwner returns expected types", "[maybe_owner]") {
    using TestType = double;
    using TestArray = std::vector<TestType, counting_allocator<TestType>>;
    mirror::MaybeOwner<TestArray> v(new TestArray(5));

    REQUIRE(typeid(v.get()).name() == typeid(TestArray*).name());
    REQUIRE(typeid(*v).name() == typeid(TestArray&).name());
    REQUIRE(typeid(v.operator->()).name() == typeid(TestArray*).name());
    REQUIRE(typeid(v->begin()).name() == typeid(TestArray::iterator).name());
    REQUIRE(typeid(v->end()).name() == typeid(TestArray::iterator).name());
}