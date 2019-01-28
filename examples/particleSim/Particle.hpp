#pragma once

#include <mirror/simt_macros.hpp>
#include <mirror/simt_serialization.hpp>
#include <mirror/simt_utilities.hpp>

#include <device_launch_parameters.h>

#define PARTICLE_ABSTRACT_TYPES \
        ENTRY(eParticleBase, Particle)

#define PARTICLE_CONCRETE_TYPES \
        ENTRY(eParticleSquare, ParticleSquare) \
        ENTRY(eParticleCircle, ParticleCircle) \
        ENTRY(eParticleTriangle, ParticleTriangle)

#define ALL_PARTICLE_TYPES \
        PARTICLE_ABSTRACT_TYPES \
        PARTICLE_CONCRETE_TYPES

enum ParticleTypes {
#define ENTRY(a, b) a,
    ALL_PARTICLE_TYPES
#undef ENTRY
    Max_
};

class Particle : public mirror::Serializable<ParticleTypes> {
public:
    HOSTDEVICE virtual ~Particle() { ; }
    HOSTDEVICE virtual double area() const = 0;
    HOSTDEVICE virtual double mass() const = 0;
};


class ParticleSquare : public Particle {
public:
    HOSTDEVICE ParticleSquare() { ; }
    HOSTDEVICE ParticleSquare(double L);
    HOSTDEVICE ~ParticleSquare() override;

    HOSTDEVICE double area() const override;

    HOSTDEVICE double mass() const override;

    HOST void write(mirror::serializer & io) const override;

    HOSTDEVICE void read(mirror::serializer::size_type startPosition,
        mirror::serializer & io) override;

    HOSTDEVICE type_id_t type() const override;

private:
    double m_L;
};

class ParticleCircle : public Particle {
public:
    HOSTDEVICE ParticleCircle() { ; }
    HOSTDEVICE ParticleCircle(double radius);
    HOSTDEVICE ~ParticleCircle() override;

    HOSTDEVICE double area() const override;

    HOSTDEVICE double mass() const override;

    HOST void write(mirror::serializer & io) const override;

    HOSTDEVICE void read(mirror::serializer::size_type startPosition,
        mirror::serializer & io) override;

    HOSTDEVICE type_id_t type() const override;

private:
    double m_radius;
};

class ParticleTriangle : public Particle {
public:
    HOSTDEVICE ParticleTriangle() { ; }
    HOSTDEVICE ParticleTriangle(double base, double height);
    HOSTDEVICE ~ParticleTriangle() override;

    HOSTDEVICE double area() const override;

    HOSTDEVICE double mass() const override;

    HOST void write(mirror::serializer & io) const override;

    HOSTDEVICE void read(mirror::serializer::size_type startPosition,
        mirror::serializer & io) override;

    HOSTDEVICE type_id_t type() const override;

private:
    double m_base;
    double m_height;
};


template <>
struct mirror::polymorphic_traits<Particle> {
    using size_type = std::size_t;
    using pointer = Particle * ;
    using enum_type = ParticleTypes;

    static size_t cache[enum_type::Max_];

    static HOST size_type sizeOf(pointer p) {
        if (cache[p->type()])
            return cache[p->type()];

        switch (p->type()) {
        #define ENTRY(a, b) \
        case enum_type::a: \
            cache[enum_type::a] = mirror::getDeviceSize<b>(); \
            return cache[enum_type::a];
            ALL_PARTICLE_TYPES
        #undef ENTRY
        default:
            throw std::runtime_error("Invalid type: cannot find the sizeof it");
        }
    }

    static HOSTDEVICE void create(mirror::vector<pointer> & device_objs, mirror::serializer & io) {
        auto tid = mirror::getTID();
        auto const stride = mirror::gridStride();

        for (; tid < device_objs.size(); tid += stride) {
            auto startPosition = io.mark_position(tid);
            enum_type type;
            io.read(startPosition, &type);

            switch (type) {

            #define ENTRY(a, b) \
            case enum_type::a: \
                mirror::construct_obj<b>(device_objs[tid]); \
                break;
                PARTICLE_CONCRETE_TYPES
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
