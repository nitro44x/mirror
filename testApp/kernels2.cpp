#include "kernels2.hpp"

#include "simt_macros.hpp"
#include "simt_vector.hpp"
#include "simt_serialization.hpp"
#include "simt_utilities.hpp"

#include <vector>

#include <device_launch_parameters.h>

#define SIMPLE_ABSTRACT_TYPES \
        ENTRY(eSimpleBase, SimpleBase)

#define SIMPLE_CONCRETE_TYPES \
        ENTRY(eSimpleDerived1, SimpleDerived1) \
        ENTRY(eSimpleDerived2, SimpleDerived2) \
        ENTRY(eSimpleDerived1_2, SimpleDerived1_2) \

#define ALL_SIMPLE_TYPES \
        SIMPLE_ABSTRACT_TYPES \
        SIMPLE_CONCRETE_TYPES

enum SimpleTypes {
#define ENTRY(a, b) a,
    ALL_SIMPLE_TYPES
#undef ENTRY
    Max_
};

class SimpleBase : public simt::serialization::Serializable<SimpleTypes> {
public:
    HOSTDEVICE virtual ~SimpleBase() { ; }

    HOSTDEVICE virtual void sayHi() = 0;
};

class SimpleDerived1 : public SimpleBase {
public:
    HOSTDEVICE SimpleDerived1() { ; }
    HOSTDEVICE SimpleDerived1(int j) : j(j) {}
    HOSTDEVICE ~SimpleDerived1() override { ; }

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
        return SimpleTypes::eSimpleDerived1;
    }

private:
    int j = 0;
};

class SimpleDerived2 : public SimpleBase {
public:
    HOSTDEVICE SimpleDerived2() { ; }
    HOSTDEVICE SimpleDerived2(int j) : d(j) {}
    HOSTDEVICE ~SimpleDerived2() override { ; }

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
        return SimpleTypes::eSimpleDerived2;
    }

private:
    double d = 0;
};

class SimpleDerived1_2 : public SimpleDerived1 {
public:
    HOSTDEVICE SimpleDerived1_2() { ; }

    HOST SimpleDerived1_2(size_t j, double s) : SimpleDerived1(int(j) + 1), v(new simt::containers::vector<double>(j)) {
        std::iota(v->begin(), v->end(), s);
    }

    HOSTDEVICE ~SimpleDerived1_2() override { ; }

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
        SimpleDerived1::sayHi();
    }

    HOST void write(simt::serialization::serializer & io) const override {
        printf("Writing in Derived1_2 %p\n", v.get());
        io.write(v.get());
        SimpleDerived1::write(io);
    }

    HOSTDEVICE void read(simt::serialization::serializer::size_type startPosition,
        simt::serialization::serializer & io) override {
        simt::containers::vector<double> * p = nullptr;
        io.read(startPosition, &p);
        v.setData(p, false);
        printf("[%u] Reading in Derived1_2 %p\n", simt::utilities::getTID(), v.get());
        SimpleDerived1::read(startPosition, io);
    }

    HOSTDEVICE type_id_t type() const override {
        return SimpleTypes::eSimpleDerived1_2;
    }


private:
    simt::memory::MaybeOwner<simt::containers::vector<double>> v;
};

template <>
struct simt::serialization::polymorphic_traits<SimpleBase> {
    using size_type = std::size_t;
    using pointer = SimpleBase * ;
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
            ALL_SIMPLE_TYPES
        #undef ENTRY
        case enum_type::Max_:
        default:
            throw;
        }
    }

    static HOSTDEVICE void create(simt::containers::vector<pointer> & device_objs, simt::serialization::serializer & io) {
        auto tid = simt::utilities::getTID();

        for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
            auto startPosition = io.mark_position(tid);
            enum_type type;
            io.read(startPosition, &type);

            switch (type) {

            #define ENTRY(a, b) \
            case enum_type::a: \
                simt::serialization::construct_obj<b>(device_objs[tid]); \
                break;
                    SIMPLE_CONCRETE_TYPES
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

size_t simt::serialization::polymorphic_traits<SimpleBase>::cache[enum_type::Max_];


template<typename T>
__global__
void sayHi(simt::containers::vector<T*> & device_objs) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
        //printf("Saying hi from an A* \n");
        device_objs[tid]->sayHi();
    }
}

void simple_polymorphic_test() {
    const auto N = 5;
    std::vector<SimpleBase*> host_objs;
    for (auto i = 0; i < N; ++i) {
        host_objs.push_back(new SimpleDerived1(i));
        host_objs.push_back(new SimpleDerived2(i));
        host_objs.push_back(new SimpleDerived1_2(5, i));
    }

    simt::serialization::polymorphic_mirror<SimpleBase> device_objs(host_objs);

    auto const nBlocks = 128;
    auto const nThreadsPerBlock = 128;
    sayHi<<<nBlocks, nThreadsPerBlock>>>(*device_objs);
    simt_sync;
}