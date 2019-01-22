#pragma once

#include "simt_macros.hpp"
#include "simt_allocator.hpp"

#include <algorithm>

namespace simt {
    namespace containers {

        template <typename T>
        class vectorIter {
        public:
            using value_type = T;
            using pointer = T * ;
            using const_pointer = const T*;
            using reference = T & ;
            using const_reference = const T&;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;

        public:
            HOSTDEVICE vectorIter(pointer p, size_type o) : p(p + o) {}

            HOSTDEVICE bool operator==(vectorIter const& other) { return other.p == p; }
            HOSTDEVICE bool operator!=(vectorIter const& other) { return !(*this == other); }
            HOSTDEVICE reference operator*() { return *p; }
            HOSTDEVICE vectorIter & operator++() { ++p; return *this; }
            HOSTDEVICE vectorIter operator++(int) {
                vectorIter clone(*this);
                ++p;
                return clone;
            }

            HOSTDEVICE vectorIter & operator--() { --p; return *this; }
            HOSTDEVICE vectorIter operator--(int) {
                vectorIter clone(*this);
                --p;
                return clone;
            }

        private:
            pointer p = nullptr;
        };

        template <typename T>
        __global__ void setAllTo(T * start, T * stop, T value) {
            auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            auto const N = (stop - start) / sizeof(T);
            for (; tid < N; tid += blockDim.x * gridDim.x) {
                start[tid] = value;
            }
        }

        template <typename T>
        __global__ void setIndexTo(T * location, T value) {
            auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid == 0) {
                *location = value;
            }
        }


        template <typename T> struct allowed_new_overloads_vector { using is_whitelisted = std::false_type; };
        template <> struct allowed_new_overloads_vector<simt::memory::HostOnly> { using is_whitelisted = std::true_type; };
        template <> struct allowed_new_overloads_vector<simt::memory::Managed> { using is_whitelisted = std::true_type; };

        template <typename T,
            class Alloc = simt::memory::managed_allocator<T>,
            simt::memory::OverloadNewType New_t = simt::memory::OverloadNewType::eManaged>
            class vector final : public simt::memory::Overload_trait_t<New_t>::type {
            public:
                using value_type = T;
                using allocator_type = Alloc;
                using pointer = T * ;
                using const_pointer = const T*;
                using reference = T & ;
                using const_reference = const T&;
                using size_type = std::size_t;
                using difference_type = std::ptrdiff_t;
                using iterator = vectorIter<T>;
                using const_iterator = const iterator;
                using reverse_iterator = std::reverse_iterator<iterator>;
                using const_reverse_iterator = std::reverse_iterator<const_iterator>;

                using is_whitelisted = allowed_new_overloads_vector<simt::memory::Overload_trait_t<New_t>::type>::is_whitelisted;
                static_assert(is_whitelisted::value, "Invalid new/delete overload for a vector, cannot use device only overload");


                static const simt::memory::OverloadNewType memory_type = New_t;

            public:
                vector() = default;

                HOST vector(size_type nElements) : m_alloc(), m_data(m_alloc.allocate(nElements)), m_size(nElements), m_capacity(nElements) {}

                HOST vector(size_type nElements, value_type initValue) : vector(nElements) {
                    // \todo Find a way to do this at compile time.
                    if (std::is_same<simt::memory::device_allocator<T>, allocator_type>::value) {
                        setAllTo << <128, 128 >> > (m_data, m_data + sizeof(T)*m_size, initValue);
                        simt_sync;
                    }
                    else {
                        for (size_type i = 0; i < nElements; ++i)
                            m_data[i] = initValue;
                    }
                }

                template<typename _T,
                    typename _Alloc,
                    simt::memory::OverloadNewType _New_t>
                    HOST vector(vector<_T, _Alloc, _New_t> * encodedObjs) {
                    static_assert(std::is_trivially_copyable<_T>::value, "Cannot use non-trivial copyable encoded objects");
                }

                HOST ~vector() {
                    m_alloc.deallocate(m_data, m_size);
                }

                HOST vector(vector const& other) : m_alloc(), m_data(m_alloc.allocate(other.m_size)), m_size(other.m_size), m_capacity(other.m_size) {
                    internal_memcpy(m_data, other.m_data, m_size);
                }

                HOST vector& operator=(vector const& other) {
                    if (&other == this)
                        return *this;

                    m_alloc.deallocate(m_data, m_size);
                    m_size = other.m_size;
                    m_data = m_alloc.allocate(m_size);
                    m_capacity = m_size;
                    internal_memcpy(m_data, other.m_data, m_size);

                    return *this;
                }

                HOST vector(vector && other) : m_alloc(std::move(other.m_alloc)), m_data(std::move(other.m_data)),
                    m_size(std::move(other.m_size)), m_capacity(std::move(other.m_capacity)) {
                    other.m_data = nullptr;
                }

                HOST vector& operator=(vector && other) {
                    if (&other == this)
                        return *this;

                    m_alloc.deallocate(m_data, m_size);
                    m_alloc = std::move(other.m_alloc);
                    m_size = std::move(other.m_size);
                    m_capacity = std::move(other.m_capacity);
                    m_data = std::move(other.m_data);
                    other.m_data = nullptr;
                    return *this;
                }


                HOST void resize(size_type nElements) {
                    if (nElements > capacity()) {
                        pointer tmp = m_alloc.allocate(nElements);
                        internal_memcpy(tmp, m_data, size());
                        m_alloc.deallocate(m_data, capacity());
                        m_data = tmp;
                        m_capacity = nElements;
                    }

                    m_size = nElements;
                }

                HOST void push_back(value_type value) {
                    auto const requiredCapcity = size() + 1;
                    if (requiredCapcity > capacity())
                        grow(calculate_growth(requiredCapcity));
                    internal_setValue(size(), value);
                    ++m_size;
                }

                HOSTDEVICE pointer data() const { return m_data; }
                HOSTDEVICE size_type size() const { return m_size; }
                HOSTDEVICE size_type capacity() const { return m_capacity; }

                HOSTDEVICE reference operator[](size_type index) { return *(m_data + index); }
                HOSTDEVICE const_reference operator[](size_type index) const { return *(m_data + index); }

                HOSTDEVICE iterator begin() { return iterator(m_data, 0); }
                HOSTDEVICE iterator end() { return iterator(m_data, m_size); }

                HOSTDEVICE const_iterator begin() const { return iterator(m_data, 0); }
                HOSTDEVICE const_iterator end() const { return iterator(m_data, m_size); }

            private:

                HOST void grow(size_type newCapacity) {
                    pointer tmp = m_alloc.allocate(newCapacity);
                    internal_memcpy(tmp, m_data, std::min(size(), newCapacity));
                    m_alloc.deallocate(m_data, capacity());
                    m_data = tmp;
                    m_capacity = newCapacity;
                }

                HOST size_type calculate_growth(size_type requiredCapcity) {
                    auto const newCapacity = capacity() + capacity() / 2;
                    if (newCapacity < requiredCapcity)
                        return requiredCapcity;
                    return newCapacity;
                }

                HOST static void internal_memcpy(pointer dst, pointer src, size_type nElements) {
                    // \todo Find a way to do this at compile time.
                    if (std::is_same<simt::memory::device_allocator<T>, allocator_type>::value) {
                        cudaMemcpy(dst, src, sizeof(T) * nElements, cudaMemcpyDeviceToDevice);
                    }
                    else {
                        memcpy(dst, src, sizeof(T) * nElements);
                    }
                }

                HOST void internal_setValue(size_type index, T value) {
                    // \todo Find a way to do this at compile time.
                    if (std::is_same<simt::memory::device_allocator<T>, allocator_type>::value) {
                        setIndexTo << <1, 1 >> > (data() + index, value);
                        simt_sync;
                    }
                    else {
                        this->operator[](size()) = value;
                    }
                }

                allocator_type m_alloc{};
                pointer m_data = nullptr;
                size_type m_size = 0;
                size_type m_capacity = 0;
        };
    }
}
