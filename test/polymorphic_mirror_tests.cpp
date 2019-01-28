#include <catch/catch.hpp>

#include <mirror/simt_allocator.hpp>
#include <mirror/simt_utilities.hpp>
#include <mirror/simt_serialization.hpp>

#include <cuda_runtime_api.h>


#define SIMPLETest_ABSTRACT_TYPES \
        ENTRY(eSimpleBaseTest, SimpleBaseTest)

#define SIMPLETest_CONCRETE_TYPES \
        ENTRY(eSimpleDerivedTest, SimpleDerivedTest)

#define ALLTest_SIMPLE_TYPES \
        SIMPLETest_ABSTRACT_TYPES \
        SIMPLETest_CONCRETE_TYPES

enum SimpleTypes {
#define ENTRY(a, b) a,
    ALLTest_SIMPLE_TYPES
#undef ENTRY
    Max_
};

class SimpleBaseTest : public simt::serialization::Serializable<SimpleTypes> {
public:
    HOSTDEVICE virtual ~SimpleBaseTest() { ; }

    HOSTDEVICE virtual double getValue() const = 0;
};

class SimpleDerivedTest : public SimpleBaseTest {
public:
    HOSTDEVICE SimpleDerivedTest() { ; }
    HOSTDEVICE SimpleDerivedTest(double j) : j(j) {}
    HOSTDEVICE ~SimpleDerivedTest() override { ; }

    HOSTDEVICE double getValue() const override {
        return j;
    }

    HOST void write(simt::serialization::serializer & io) const override {
        io.write(j);
    }

    HOSTDEVICE void read(simt::serialization::serializer::size_type startPosition,
        simt::serialization::serializer & io) override {
        io.read(startPosition, &j);
    }

    HOSTDEVICE type_id_t type() const override {
        return SimpleTypes::eSimpleDerivedTest;
    }

private:
    double j = 0;
};

template <>
struct simt::serialization::polymorphic_traits<SimpleBaseTest> {
    using size_type = std::size_t;
    using pointer = SimpleBaseTest * ;
    using enum_type = SimpleTypes;

    static size_t cache[enum_type::Max_];

    static HOST size_type sizeOf(pointer p) {
        if (cache[p->type()])
            return cache[p->type()];

        switch (p->type()) {
        #define ENTRY(a, b) \
        case enum_type::a: \
            cache[enum_type::a] = simt::utilities::getDeviceSize<b>(); \
            return cache[enum_type::a];
            ALLTest_SIMPLE_TYPES
        #undef ENTRY
        case enum_type::Max_:
        default:
            throw;
        }
    }

    static HOSTDEVICE void create(simt::containers::vector<pointer> & device_objs, simt::serialization::serializer & io) {
        auto tid = simt::utilities::getTID();
        auto stride = simt::utilities::gridStride();

        for (; tid < device_objs.size(); tid += stride) {
            auto startPosition = io.mark_position(tid);
            enum_type type;
            io.read(startPosition, &type);

            switch (type) {

            #define ENTRY(a, b) \
            case enum_type::a: \
                simt::serialization::construct_obj<b>(device_objs[tid]); \
                break;
                    SIMPLETest_CONCRETE_TYPES
            #undef ENTRY

            default:
                printf("Error allocating object!\n");
            }

            if (nullptr == device_objs[tid])
                printf("failed allocation at tid = %u\n", tid);

            device_objs[tid]->read(startPosition, io);
        }
    }
};

size_t simt::serialization::polymorphic_traits<SimpleBaseTest>::cache[enum_type::Max_];

__global__
void getValues(simt::containers::vector<SimpleBaseTest*> const* objs, simt::containers::vector<double> & outVals) {
    auto tid = simt::utilities::getTID();
    if (tid == 0) {
        size_t i = 0;
        for (auto const& o : *objs) {
            outVals[i++] = o->getValue();
        }
    }
}

TEST_CASE("Can mirror polymorphic objects") {
    std::vector<SimpleBaseTest*> host_objs;
    host_objs.push_back(new SimpleDerivedTest(123));

    simt::serialization::polymorphic_mirror<SimpleBaseTest> device_objs(host_objs);
    simt::memory::MaybeOwner<simt::containers::vector<double>> out(new simt::containers::vector<double>(host_objs.size(), 0));
    getValues<<<1, 1>>>(device_objs.get(), *out);
    simt_sync;

    size_t i = 0;
    for (auto const& o : host_objs)
        REQUIRE(o->getValue() == out[i++]);
}