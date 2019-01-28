#include <catch/catch.hpp>

#include <mirror/simt_allocator.hpp>
#include <mirror/simt_utilities.hpp>
#include <mirror/simt_serialization.hpp>

#include <cuda_runtime_api.h>

TEMPLATE_TEST_CASE("Can serialize basic types and get them back out", "[serialization]", 
    size_t,
    int,
    float,
    double,
    char) {
    mirror::serializer io;

    TestType in = 5;

    io.write(in);
    io.write(&in);

    TestType out = 0;
    TestType * p_out = nullptr;

    io.seek(mirror::Position::Beginning);
    io.read(&out);
    io.read(&p_out);

    REQUIRE(out == in);
    REQUIRE(p_out == &in);
}

TEST_CASE("Can mark locations within serializer and read them out of order", "[serialization]") {
    struct a_t {
        double i;
        int j;
    };

    struct b_t {
        char c;
        size_t u;
    };

    a_t a = { 1.23, 5 };
    b_t b = { 'D', 2 };

    mirror::serializer io;
    auto a_pos = io.mark();
    io.write(a.i);
    io.write(a.j);
    auto b_pos = io.mark();
    io.write(b.c);
    io.write(b.u);

    REQUIRE(b_pos == io.mark_position(1));
    REQUIRE(a_pos == io.mark_position(0));
    REQUIRE(2 == io.number_of_marks());

    b_t b_out;
    io.read(b_pos, &b_out.c);
    io.read(b_pos, &b_out.u);
    REQUIRE(b_out.c == b.c);
    REQUIRE(b_out.u == b.u);

    a_t a_out;
    io.read(a_pos, &a_out.i);
    io.read(a_pos, &a_out.j);
    REQUIRE(a_out.i == a.i);
    REQUIRE(a_out.j == a.j);
}