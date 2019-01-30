#include <catch/catch.hpp>

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

using namespace Catch;

/*
 *  Particle Particle test
 *  
 *  Launch severnal different types of particles into the air
 *  Each particle will be affected by gravity and drag
 *  Each particle will have a different area calculation (polymorphic)
 *
 */

template <typename Real = double, typename Alloc = mirror::managed_allocator<Real>>
class DataStore : public mirror::Managed {
    using vector_t = mirror::vector<Real, Alloc>;

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

        if (!std::is_same<Alloc, mirror::device_allocator<Real>>::value) {
            for (size_t i = 0; i < m_nParticles; ++i) {
                m_x[i] = norm_pos(engine);
                m_y[i] = norm_pos(engine);
                m_z[i] = norm_pos(engine) + 10;
                m_Vx[i] = norm_vel(engine);
                m_Vy[i] = norm_vel(engine);
                m_Vz[i] = norm_vel(engine) + 5;
            }
        }
        else {
            std::vector<Real> x(m_x->size());
            std::vector<Real> y(m_y->size());
            std::vector<Real> z(m_z->size());
            std::vector<Real> Vx(m_Vx->size());
            std::vector<Real> Vy(m_Vy->size());
            std::vector<Real> Vz(m_Vz->size());
            for (size_t i = 0; i < m_nParticles; ++i) {
                x[i] = norm_pos(engine);
                y[i] = norm_pos(engine);
                z[i] = norm_pos(engine) + 10;
                Vx[i] = norm_vel(engine);
                Vy[i] = norm_vel(engine);
                Vz[i] = norm_vel(engine) + 5;
            }

            mirror_check(cudaMemcpy(m_x->data(), x.data(), sizeof(Real) * x.size(), cudaMemcpyHostToDevice));
            mirror_check(cudaMemcpy(m_y->data(), y.data(), sizeof(Real) * y.size(), cudaMemcpyHostToDevice));
            mirror_check(cudaMemcpy(m_z->data(), z.data(), sizeof(Real) * z.size(), cudaMemcpyHostToDevice));
            mirror_check(cudaMemcpy(m_Vx->data(), Vx.data(), sizeof(Real) * Vx.size(), cudaMemcpyHostToDevice));
            mirror_check(cudaMemcpy(m_Vy->data(), Vy.data(), sizeof(Real) * Vy.size(), cudaMemcpyHostToDevice));
            mirror_check(cudaMemcpy(m_Vz->data(), Vz.data(), sizeof(Real) * Vz.size(), cudaMemcpyHostToDevice));
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
        if (!std::is_same<Alloc, mirror::device_allocator<Real>>::value) {
            for (size_t i = 0; i < m_nParticles; ++i) {
                s[6 * i + 0] = m_x[i];
                s[6 * i + 1] = m_y[i];
                s[6 * i + 2] = m_z[i];
                s[6 * i + 3] = m_Vx[i];
                s[6 * i + 4] = m_Vy[i];
                s[6 * i + 5] = m_Vz[i];
            }
        }
        else {
            std::vector<Real> x(m_x->size());
            std::vector<Real> y(m_y->size());
            std::vector<Real> z(m_z->size());
            std::vector<Real> Vx(m_Vx->size());
            std::vector<Real> Vy(m_Vy->size());
            std::vector<Real> Vz(m_Vz->size());

            size_t offset = 0;
            mirror_check(cudaMemcpy(s.data() + offset, m_x->data(), sizeof(Real) * x.size(), cudaMemcpyDeviceToHost));
            offset += m_x->size();
            mirror_check(cudaMemcpy(s.data() + offset, m_y->data(), sizeof(Real) * y.size(), cudaMemcpyDeviceToHost));
            offset += m_x->size();
            mirror_check(cudaMemcpy(s.data() + offset, m_z->data(), sizeof(Real) * z.size(), cudaMemcpyDeviceToHost));
            offset += m_x->size();
            mirror_check(cudaMemcpy(s.data() + offset, m_Vx->data(), sizeof(Real) * Vx.size(), cudaMemcpyDeviceToHost));
            offset += m_x->size();
            mirror_check(cudaMemcpy(s.data() + offset, m_Vy->data(), sizeof(Real) * Vy.size(), cudaMemcpyDeviceToHost));
            offset += m_x->size();
            mirror_check(cudaMemcpy(s.data() + offset, m_Vz->data(), sizeof(Real) * Vz.size(), cudaMemcpyDeviceToHost));
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

#define STRIDE_64K 65536
template <typename Real, typename Alloc>
__global__ void call_integrateTo_byWarp(double dt, mirror::managed_vector<Particle*> const* particles, DataStore<Real, Alloc> * store) {
    int lane_id = threadIdx.x & 31;
    size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
    int warps_per_grid = (blockDim.x * gridDim.x) >> 5;
    auto const size = particles->size();
    size_t warp_total = (size + STRIDE_64K - 1) / STRIDE_64K;

    size_t n = size / sizeof(Real);

    for (; warp_id < warp_total; warp_id += warps_per_grid) {
        #pragma unroll
        for (int rep = 0; rep < STRIDE_64K / sizeof(Real) / 32; rep++) {
            size_t ind = warp_id * STRIDE_64K / sizeof(Real) + rep * 32 + lane_id;
            if (ind < n) {
                integrateTo(dt, particles, store, ind);
            }
        }
    }
}

template <typename ParticleContainer, typename Real, typename Alloc>
void integrateTo_cpu(double dt, ParticleContainer const* particles, DataStore<Real, Alloc> * store) {

    #pragma omp parallel for
    for (int i = 0; i < particles->size(); ++i) {
        integrateTo(dt, particles, store, i);
    }
}

template <typename ParticleContainer, typename Real, typename Alloc>
DEVICE void integrateTo_gpu(double dt, ParticleContainer const* particles, DataStore<Real, Alloc> * store) {
    auto tid = mirror::getTID();
    auto stride = mirror::gridStride();

    for (; tid < particles->size(); tid += stride) {
        integrateTo(dt, particles, store, tid);
    }
}

template <typename ParticleContainer, typename Real, typename Alloc>
HOSTDEVICE void integrateTo(double dt, ParticleContainer const* particles, DataStore<Real, Alloc> * store, size_t tid) {
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

template <typename Real, typename Alloc>
__global__ void call_integrateTo(double dt, mirror::managed_vector<Particle*> const* particles, DataStore<Real, Alloc> * store) {
    integrateTo_gpu(dt, particles, store);
}

template <typename Real, typename Alloc>
void print_particle_state(std::ostream & out, DataStore<Real, Alloc> const* store) {
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

template <typename Real = double, typename Alloc = mirror::managed_allocator<Real>>
class simulation {
public:

    simulation(size_t N, bool setupProblem = true) : host_particles(), device_particles() {
        for (auto i = 0; i < N; ++i)
            host_particles.push_back(new ParticleCircle(1.0));
        for (auto i = 0; i < N; ++i)
            host_particles.push_back(new ParticleSquare(2.0));
        for (auto i = 0; i < N; ++i)
            host_particles.push_back(new ParticleTriangle(5, 2));

        device_particles = new mirror::polymorphic_mirror<Particle>(host_particles);

        nParticles = host_particles.size();
        store = new DataStore<Real, Alloc>(nParticles);
        if(setupProblem)
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
    mirror::host_vector<Particle*> host_particles;
    mirror::polymorphic_mirror<Particle> * device_particles;
    std::pair<double, std::vector<double>> last_checkpoint;
    DataStore<Real, Alloc> * store = nullptr;
};


TEST_CASE("Simple particle trajectory benchmarks", "[particleSim][benchmark]") {
    
#ifdef NDEBUG
    size_t const nParticles = 1000000;
    size_t const nIterations = 10000;
    size_t const checkpointInterval = 1000;
#else
    size_t const nParticles = 10;
    size_t const nIterations = 10;
    size_t const checkpointInterval = 5;
#endif

    auto const nBlocks = 256;
    auto const nThreadsPerBlock = 256;
    double const dt = 1e-4;
    double currentTime = 0;

    BENCHMARK("Setup Problem (polymorphic mirroring)") {
        simulation<double> sim(nParticles, false);
    }

    {
        simulation<double> sim(nParticles);
        BENCHMARK("GPU with UMA") {
            for (size_t iteration = 0; iteration < nIterations; ++iteration) {
                currentTime += dt;
                call_integrateTo<<<nBlocks, nThreadsPerBlock>>>(dt, sim.device_particles->get(), sim.store);

                if (iteration % checkpointInterval == 0) {
                    mirror_sync;
                    sim.checkpoint(currentTime);
                }
            }
        }
    }

    {
        simulation<double, mirror::device_allocator<double>> sim(nParticles);
        BENCHMARK("GPU with manual sync") {
            for (size_t iteration = 0; iteration < nIterations; ++iteration) {
                currentTime += dt;
                call_integrateTo<<<nBlocks, nThreadsPerBlock>>>(dt, sim.device_particles->get(), sim.store);

                if (iteration % checkpointInterval == 0) {
                    mirror_sync;
                    sim.checkpoint(currentTime);
                }
            }
        }
    }

    {
        simulation<double> sim(nParticles);
        BENCHMARK("GPU with UMA by Warp") {
            for (size_t iteration = 0; iteration < nIterations; ++iteration) {
                currentTime += dt;
                call_integrateTo_byWarp<<<nBlocks, nThreadsPerBlock>>>(dt, sim.device_particles->get(), sim.store);

                if (iteration % checkpointInterval == 0) {
                    mirror_sync;
                    sim.checkpoint(currentTime);
                }
            }
        }
    }

    {
        simulation<double> sim(nParticles);
        BENCHMARK("CPU with OMP") {
            for (size_t iteration = 0; iteration < nIterations; ++iteration) {
                currentTime += dt;
                integrateTo_cpu(dt, &(sim.host_particles), sim.store);

                if (iteration % checkpointInterval == 0) {
                    sim.checkpoint(currentTime);
                }
            }
        }
    }
}