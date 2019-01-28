#include "Particle.hpp"

#include <device_launch_parameters.h>

#include <mirror/simt_macros.hpp>
#include <mirror/simt_allocator.hpp>
#include <mirror/simt_vector.hpp>
#include <mirror/simt_serialization.hpp>
#include <mirror/simt_utilities.hpp>

ParticleSquare::ParticleSquare(double L) : m_L(L) {}

ParticleSquare::~ParticleSquare() { ; }

HOSTDEVICE double ParticleSquare::area() const { return m_L * m_L; }

HOSTDEVICE double ParticleSquare::mass() const { return 1.0; }

HOST void ParticleSquare::write(mirror::serializer & io) const {
    io.write(m_L);
}

HOSTDEVICE void ParticleSquare::read(mirror::serializer::size_type startPosition, mirror::serializer & io) {
    io.read(startPosition, &m_L);
}

HOSTDEVICE ParticleSquare::type_id_t ParticleSquare::type() const {
    return ParticleTypes::eParticleSquare;
}

ParticleCircle::ParticleCircle(double radius) : m_radius(radius) {}

ParticleCircle::~ParticleCircle() { ; }

HOSTDEVICE double ParticleCircle::area() const { return 3.1415 * m_radius * m_radius; }

HOSTDEVICE double ParticleCircle::mass() const { return 1.0; }

HOST void ParticleCircle::write(mirror::serializer & io) const {
    io.write(m_radius);
}

HOSTDEVICE void ParticleCircle::read(mirror::serializer::size_type startPosition, mirror::serializer & io) {
    io.read(startPosition, &m_radius);
}

HOSTDEVICE ParticleCircle::type_id_t ParticleCircle::type() const {
    return ParticleTypes::eParticleCircle;
}

ParticleTriangle::ParticleTriangle(double base, double height) : m_base(base), m_height(height) {}

ParticleTriangle::~ParticleTriangle() { ; }

HOSTDEVICE double ParticleTriangle::area() const { return 0.5 * m_base * m_height; }

HOSTDEVICE double ParticleTriangle::mass() const { return 1.0; }

HOST void ParticleTriangle::write(mirror::serializer & io) const {
    io.write(m_base);
    io.write(m_height);
}

HOSTDEVICE void ParticleTriangle::read(mirror::serializer::size_type startPosition, mirror::serializer & io) {
    io.read(startPosition, &m_base);
    io.read(startPosition, &m_height);
}

HOSTDEVICE ParticleTriangle::type_id_t ParticleTriangle::type() const {
    return ParticleTypes::eParticleTriangle;
}

size_t mirror::polymorphic_traits<Particle>::cache[enum_type::Max_];
