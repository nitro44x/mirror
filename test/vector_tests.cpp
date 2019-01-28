#include <catch/catch.hpp>

#include <mirror/simt_allocator.hpp>
#include <mirror/simt_vector.hpp>

using namespace Catch;

TEMPLATE_TEST_CASE("UMA newed vectors can be sized and resized", "[vector]", 
    mirror::managed_allocator<int>, 
    mirror::device_allocator<int>,
    std::allocator<int>) {

    using VectorType = mirror::vector<int, TestType, mirror::OverloadNewType::eManaged>;

    auto v = new VectorType(5);

    REQUIRE(v->size() == 5);
    REQUIRE(v->capacity() >= 5);

    SECTION("resizing bigger changes size and capacity") {
        v->resize(10);

        REQUIRE(v->size() == 10);
        REQUIRE(v->capacity() >= 10);
    }
    SECTION("resizing smaller changes size but not capacity") {
        v->resize(0);

        REQUIRE(v->size() == 0);
        REQUIRE(v->capacity() >= 5);
    }
    SECTION("reserving smaller does not change size or capacity") {
        v->reserve(0);

        REQUIRE(v->size() == 5);
        REQUIRE(v->capacity() >= 5);
    }
    SECTION("reserving larger does not change size, but increases capacity") {
        v->reserve(10);

        REQUIRE(v->size() == 5);
        REQUIRE(v->capacity() >= 10);
    }
    SECTION("push_back increases the size by one and the capcity by >= 1") {
        v->push_back(10);
        REQUIRE(v->size() == 6);
        REQUIRE(v->capacity() >= 6);
    }

    delete v;
}

TEMPLATE_TEST_CASE("HostOnly newed vectors can be sized and resized", "[vector]",
    mirror::managed_allocator<int>,
    mirror::device_allocator<int>,
    std::allocator<int>) {

    using VectorType = mirror::vector<int, TestType, mirror::OverloadNewType::eHostOnly>;

    auto v = new VectorType(5);

    REQUIRE(v->size() == 5);
    REQUIRE(v->capacity() >= 5);

    SECTION("resizing bigger changes size and capacity") {
        v->resize(10);

        REQUIRE(v->size() == 10);
        REQUIRE(v->capacity() >= 10);
    }
    SECTION("resizing smaller changes size but not capacity") {
        v->resize(0);

        REQUIRE(v->size() == 0);
        REQUIRE(v->capacity() >= 5);
    }
    SECTION("reserving smaller does not change size or capacity") {
        v->reserve(0);

        REQUIRE(v->size() == 5);
        REQUIRE(v->capacity() >= 5);
    }
    SECTION("reserving larger does not change size, but increases capacity") {
        v->reserve(10);

        REQUIRE(v->size() == 5);
        REQUIRE(v->capacity() >= 10);
    }
    SECTION("push_back increases the size by one and the capcity by >= 1") {
        v->push_back(10);
        REQUIRE(v->size() == 6);
        REQUIRE(v->capacity() >= 6);
    }

    delete v;
}

TEST_CASE("Benchmark vector push_back", "[vector][benchmark]") {
    using VectorType = mirror::vector<int>;
    static const int size = 500000;
    VectorType v;

    BENCHMARK("Load up a vector") {
        v = VectorType();
        for (int i = 0; i < size; ++i)
            v.push_back(i);
        simt_sync;
    }
    REQUIRE(v.size() == size);

    BENCHMARK("Init a vector with a value") {
        v = VectorType(size, 123);
        simt_sync;
    }
    REQUIRE(v.size() == size);
}

TEMPLATE_TEST_CASE("UMA newed vectors construct with default value", "[vector]",
    mirror::managed_allocator<int>,
    mirror::device_allocator<int>,
    std::allocator<int>) {

    using VectorType = mirror::vector<int, TestType, mirror::OverloadNewType::eManaged>;
    int setValue = 123;
    auto v = new VectorType(5, setValue);

    if (std::is_same<mirror::device_allocator<int>, typename VectorType::allocator_type>::value) {
        std::vector<int> host_data(5);
        cudaMemcpy(host_data.data(), v->data(), sizeof(int)*v->size(), cudaMemcpyDeviceToHost);

        for (auto const& value : host_data)
            REQUIRE(value == setValue);
    }
    else {
        for (auto const& value : *v)
            REQUIRE(value == setValue);
    }

    delete v;
}


TEMPLATE_TEST_CASE("HostOnly newed vectors construct with default value", "[vector]",
    mirror::managed_allocator<int>,
    mirror::device_allocator<int>,
    std::allocator<int>) {

    using VectorType = mirror::vector<int, TestType, mirror::OverloadNewType::eHostOnly>;
    int setValue = 123;
    auto v = new VectorType(5, setValue);

    if (std::is_same<mirror::device_allocator<int>, typename VectorType::allocator_type>::value) {
        std::vector<int> host_data(5);
        cudaMemcpy(host_data.data(), v->data(), sizeof(int)*v->size(), cudaMemcpyDeviceToHost);

        for (auto const& value : host_data)
            REQUIRE(value == setValue);
    }
    else {
        for (auto const& value : *v)
            REQUIRE(value == setValue);
    }

    delete v;
}


TEMPLATE_TEST_CASE("MaybeOwner can be used with vectors", "[vector][maybe_owner]",
    mirror::managed_allocator<int>,
    mirror::device_allocator<int>,
    std::allocator<int>) {

    using VectorType = mirror::vector<int, TestType, mirror::OverloadNewType::eManaged>;
    int setValue = 123;
    mirror::MaybeOwner<VectorType> v(new VectorType(5, setValue));

    if (std::is_same<mirror::device_allocator<int>, typename VectorType::allocator_type>::value) {
        std::vector<int> host_data(5);
        cudaMemcpy(host_data.data(), v->data(), sizeof(int)*v->size(), cudaMemcpyDeviceToHost);

        for (auto const& value : host_data)
            REQUIRE(value == setValue);
    }
    else {
        for (auto const& value : *v)
            REQUIRE(value == setValue);
    }
}
