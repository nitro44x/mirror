#include "Particle.hpp"

#include <mirror/simt_macros.hpp>
#include <mirror/simt_allocator.hpp>
#include <mirror/simt_vector.hpp>
#include <mirror/simt_serialization.hpp>
#include <mirror/simt_utilities.hpp>

#include <device_launch_parameters.h>

#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

/*
 *  Particle Particle test
 *  
 *  Launch severnal different types of particles into the air
 *  Each particle will be affected by gravity and drag
 *  Each particle will have a different area calculation (polymorphic)
 *
 */

class DataStore : public mirror::Managed {
    using Real = double;
    using vector_t = mirror::vector<Real>;

public:
    HOST DataStore(size_t nParticles) : m_nParticles(nParticles), 
        m_x(new vector_t(nParticles)), 
        m_y(new vector_t(nParticles)), 
        m_z(new vector_t(nParticles)), 
        m_Vx(new vector_t(nParticles)), 
        m_Vy(new vector_t(nParticles)), 
        m_Vz(new vector_t(nParticles)) {}

    HOSTDEVICE Real x(size_t idx) const { return m_x[idx]; }
    HOSTDEVICE Real y(size_t idx) const { return m_y[idx]; }
    HOSTDEVICE Real z(size_t idx) const { return m_z[idx]; }
    HOSTDEVICE Real Vx(size_t idx) const { return m_Vx[idx]; }
    HOSTDEVICE Real Vy(size_t idx) const { return m_Vy[idx]; }
    HOSTDEVICE Real Vz(size_t idx) const { return m_Vz[idx]; }

    HOSTDEVICE void x(size_t idx, Real value) { m_x[idx] = value; }
    HOSTDEVICE void y(size_t idx, Real value) { m_y[idx] = value; }
    HOSTDEVICE void z(size_t idx, Real value) { m_z[idx] = value; }
    HOSTDEVICE void Vx(size_t idx, Real value) { m_Vx[idx] = value; }
    HOSTDEVICE void Vy(size_t idx, Real value) { m_Vy[idx] = value; }
    HOSTDEVICE void Vz(size_t idx, Real value) { m_Vz[idx] = value; }

    HOSTDEVICE size_t number_of_particles() const { return m_nParticles; }

    HOST void setup_random_distribution() {
        std::mt19937_64 engine(100);
        std::normal_distribution<double> norm_pos(0, 1);
        std::normal_distribution<double> norm_vel(0, 10);

        for (size_t i = 0; i < m_nParticles; ++i) {
            m_x[i] = norm_pos(engine);
            m_y[i] = norm_pos(engine);
            m_z[i] = norm_pos(engine) + 10;
            m_Vx[i] = norm_vel(engine);
            m_Vy[i] = norm_vel(engine);
            m_Vz[i] = norm_vel(engine) + 5;
        }
    }

    HOST void setup_simple_distribution() {
        for (size_t i = 0; i < m_nParticles; ++i) {
            m_x[i] = 0;
            m_y[i] = 0;
            m_z[i] = 0;
            m_Vx[i] = 1;
            m_Vy[i] = 1;
            m_Vz[i] = 1;
        }
    }

    HOST std::vector<double> snapshot() const {
        std::vector<double> s(m_x->size() * 6);
        for (size_t i = 0; i < m_nParticles; ++i) {
            s[6 * i + 0] = m_x[i];
            s[6 * i + 1] = m_y[i];
            s[6 * i + 2] = m_z[i];
            s[6 * i + 3] = m_Vx[i];
            s[6 * i + 4] = m_Vy[i];
            s[6 * i + 5] = m_Vz[i];
        }
        return std::move(s);
    }

private:
    size_t m_nParticles;
    mirror::MaybeOwner<vector_t> m_x;
    mirror::MaybeOwner<vector_t> m_y;
    mirror::MaybeOwner<vector_t> m_z;

    mirror::MaybeOwner<vector_t> m_Vx;
    mirror::MaybeOwner<vector_t> m_Vy;
    mirror::MaybeOwner<vector_t> m_Vz;
};


template <typename ParticleContainer>
void integrateTo_cpu(double dt, ParticleContainer const* particles, DataStore * store) {

    #pragma omp parallel for
    for (int i = 0; i < particles->size(); ++i) {
        integrateTo(dt, particles, store, i);
    }
}

template <typename ParticleContainer>
DEVICE void integrateTo_gpu(double dt, ParticleContainer const* particles, DataStore * store) {
    auto tid = mirror::getTID();
    auto stride = mirror::gridStride();

    for (; tid < particles->size(); tid += stride) {
        integrateTo(dt, particles, store, tid);
    }
}

template <typename ParticleContainer>
HOSTDEVICE void integrateTo(double dt, ParticleContainer const* particles, DataStore * store, size_t tid) {
    double const rho = 1.0;
    double const g = -9.8;

    double const x = store->x(tid);
    double const y = store->y(tid);
    double const z = store->z(tid);
    double const Vx = store->Vx(tid);
    double const Vy = store->Vy(tid);
    double const Vz = store->Vz(tid);

    double const area = (*particles)[tid]->area();
    double const mass = (*particles)[tid]->mass();
    double const massInv = 1.0 / mass;

    double const xNew = x + dt * Vx;
    double const yNew = y + dt * Vy;
    double const zNew = z + dt * Vz;

    if (zNew > 0) {
        double const VxNew = Vx + dt * (-0.5 * Vx * rho * area) * massInv;
        double const VyNew = Vy + dt * (-0.5 * Vy * rho * area) * massInv;
        double const VzNew = Vz + dt * (mass * g - 0.5 * Vz * rho * area) * massInv;

        store->x(tid, xNew);
        store->y(tid, yNew);
        store->z(tid, zNew);
        store->Vx(tid, VxNew);
        store->Vy(tid, VyNew);
        store->Vz(tid, VzNew);
    }
    else {
        store->x(tid, xNew);
        store->y(tid, yNew);
        store->z(tid, zNew);
        store->Vx(tid, 0);
        store->Vy(tid, 0);
        store->Vz(tid, 0);
    }
}

__global__ void call_integrateTo(double dt, mirror::vector<Particle*> const* particles, DataStore * store) {
    integrateTo_gpu(dt, particles, store);
}

void print_particle_state(std::ostream & out, DataStore const* store) {
    for (size_t i = 0; i < store->number_of_particles(); ++i) {
        out << store->x(i) << ", ";
        out << store->y(i) << ", ";
        out << store->z(i) << ", ";
        out << store->Vx(i) << ", ";
        out << store->Vy(i) << ", ";
        out << store->Vz(i);

        if (i + 1 != store->number_of_particles())
            out << ", ";
    }
    out << std::endl;
}

void print_headers(std::ostream & out, size_t nParticles) {
    for (size_t i = 0; i < nParticles; ++i) {
        out << "x"  << ", ";
        out << "y"  << ", ";
        out << "z"  << ", ";
        out << "Vx" << ", ";
        out << "Vy" << ", ";
        out << "Vz";

        if (i + 1 != nParticles)
            out << ", ";
    }
    out << std::endl;
}

class simulation {
public:

    simulation(size_t N) : host_particles(), device_particles() {
        for (auto i = 0; i < N; ++i)
            host_particles.push_back(new ParticleCircle(1.0));
        for (auto i = 0; i < N; ++i)
            host_particles.push_back(new ParticleSquare(2.0));
        for (auto i = 0; i < N; ++i)
            host_particles.push_back(new ParticleTriangle(5, 2));

        device_particles = new mirror::polymorphic_mirror<Particle>(host_particles);

        nParticles = host_particles.size();
        store = new DataStore(nParticles);
        store->setup_random_distribution();
    }

    void checkpoint(double time) {
        last_checkpoint = std::make_pair( time, store->snapshot());
    }

    ~simulation() {
        delete store;
        delete device_particles;
    }

    size_t nParticles;
    std::vector<Particle*> host_particles;
    mirror::polymorphic_mirror<Particle> * device_particles;
    std::pair<double, std::vector<double>> last_checkpoint;
    DataStore * store = nullptr;
};

void simple_particle_test_gpu(size_t N, size_t outputInterval) {
    auto start = std::chrono::high_resolution_clock::now();
    simulation sim(N);
    auto setup = std::chrono::high_resolution_clock::now();

    auto const nBlocks = 128;
    auto const nThreadsPerBlock = 128;

    //auto & out = std::cout;
    //std::ofstream out("particle_trajectories.csv");
    //print_headers(out, sim.nParticles);

    size_t nIterations = outputInterval;
    double const dt = 1e-4;
    double currentTime = 0;
    for(size_t iteration = 0; iteration < nIterations; ++iteration) {
        currentTime += dt;
        call_integrateTo<<<nBlocks, nThreadsPerBlock>>>(dt, sim.device_particles->get(), sim.store);

        if (iteration % 5 == 0) {
            mirror_sync;
            //std::cout << "iter " << iteration << ": " << currentTime << std::endl;
            sim.checkpoint(currentTime);
        }
    }

    mirror_sync;

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Setup Time : " << std::chrono::duration_cast<std::chrono::seconds>(setup - start).count() << std::endl;
    std::cout << "Run Time : " << std::chrono::duration_cast<std::chrono::seconds>(stop - setup).count() << std::endl;
}


void simple_particle_test_cpu(size_t N, size_t outputInterval) {
    auto start = std::chrono::high_resolution_clock::now();
    simulation sim(N);
    auto setup = std::chrono::high_resolution_clock::now();

    //auto & out = std::cout;
    //std::ofstream out("particle_trajectories.csv");
    //print_headers(out, sim.nParticles);

    size_t nIterations = outputInterval;
    double const dt = 1e-4;
    double currentTime = 0;
    for (size_t iteration = 0; iteration < nIterations; ++iteration) {
        currentTime += dt;
        integrateTo_cpu(dt, &(sim.host_particles), sim.store);

        if (iteration % 5 == 0) {
            //std::cout << "iter " << iteration << ": " << currentTime << std::endl;
            sim.checkpoint(currentTime);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Setup Time : " << std::chrono::duration_cast<std::chrono::seconds>(setup - start).count() << std::endl;
    std::cout << "Run Time : " << std::chrono::duration_cast<std::chrono::seconds>(stop - setup).count() << std::endl;
}

int main() {
    size_t const N = 10000;
    size_t const checkpointIterval = 100;
    simple_particle_test_cpu(N, checkpointIterval);
    simple_particle_test_gpu(N, checkpointIterval);
}