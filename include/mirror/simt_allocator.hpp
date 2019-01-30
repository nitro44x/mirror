#pragma once

#include <cuda_runtime.h>

#include <mirror/simt_macros.hpp>

namespace mirror {

        /*
            https://stackoverflow.com/a/53033942
            Broadly speaking, an Allocator type is used when an object of one type (typically a container) needs to
            manage memory to hold an object or objects of some other type. Overloading operator new and operator delete
            within a class is used when objects of that type need some special memory management.
        */

        template <class T>
        struct managed_allocator {
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;

            using value_type = T;
            using pointer = T * ;
            using const_pointer = const T*;
            using reference = T & ;
            using const_reference = const T&;

            template< class U > struct rebind { typedef managed_allocator<U> other; };
            managed_allocator() = default;

            template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}

            HOSTDEVICE T* allocate(std::size_t n) {
                #ifndef __CUDA_ARCH__
                void* out = nullptr;
                mirror_check(cudaMallocManaged(&out, n * sizeof(T)));
                return static_cast<T*>(out);
                #else
                return nullptr;
                #endif

            }

            HOSTDEVICE void deallocate(T* p, std::size_t) noexcept {
                #ifndef __CUDA_ARCH__
                mirror_check(cudaFree(p));
                #endif
            }

        };

        template <class T>
        struct device_allocator {
            typedef std::size_t size_type;
            typedef std::ptrdiff_t difference_type;

            using value_type = T;
            using pointer = T * ;
            using const_pointer = const T*;
            using reference = T & ;
            using const_reference = const T&;

            template< class U > struct rebind { typedef device_allocator<U> other; };
            device_allocator() = default;

            template <class U> constexpr device_allocator(const device_allocator<U>&) noexcept {}

            HOSTDEVICE T* allocate(std::size_t n) {
                #ifndef __CUDA_ARCH__
                void* out = nullptr;
                mirror_check(cudaMalloc(&out, n * sizeof(T)));
                mirror_check(cudaMemset(out, 0, n * sizeof(T)));
                return static_cast<T*>(out);
                #else
                return nullptr;
                #endif      
            }

            HOSTDEVICE void deallocate(T* p, std::size_t) noexcept {
                #ifndef __CUDA_ARCH__
                mirror_check(cudaFree(p));
                #endif          
            }

        };

        class Managed {
        public:
            void *operator new(size_t len) {
                void *ptr;
                mirror_check(cudaMallocManaged(&ptr, len));
                mirror_sync;
                return ptr;
            }

            void operator delete(void *ptr) {
                mirror_sync;
                mirror_check(cudaFree(ptr));
            }
        };

        class DeviceOnly {
        public:
            void *operator new(size_t len) {
                void *ptr;
                mirror_check(cudaMalloc(&ptr, len));
                mirror_sync;
                return ptr;
            }

            void operator delete(void *ptr) {
                mirror_sync;
                mirror_check(cudaFree(ptr));
            }
        };

        class HostOnly {
        public:
            void *operator new(size_t len) {
                return malloc(len);
            }

            void operator delete(void *ptr) {
                free(ptr);
            }
        };

        enum class OverloadNewType {
            eManaged = 0,
            eDeviceOnly,
            eHostOnly
        };

        template<OverloadNewType T> struct force_overload_specialization : public std::false_type {};

        template <OverloadNewType T>
        struct Overload_trait_t {
            using type = HostOnly;
            static_assert(force_overload_specialization<T>::value, "Must choose how to overload new/delete");
        };

        template <>
        struct Overload_trait_t<OverloadNewType::eManaged> {
            using type = Managed;
        };

        template <>
        struct Overload_trait_t<OverloadNewType::eDeviceOnly> {
            using type = DeviceOnly;
        };

        template <>
        struct Overload_trait_t<OverloadNewType::eHostOnly> {
            using type = HostOnly;
        };

        template <typename T>
        class MaybeOwner {
        public:
            using value_type = T::value_type;
            using pointer = T::pointer;
            using const_pointer = T::const_pointer;
            using reference = T::reference;
            using const_reference = T::const_reference;
            using size_type = T::size_type;
            using difference_type = T::difference_type;
            using iterator = T::iterator;
            using const_iterator = T::const_iterator;
            using reverse_iterator = T::reverse_iterator;
            using const_reverse_iterator = T::const_reverse_iterator;

        public:
            HOSTDEVICE MaybeOwner() { ; }
            HOSTDEVICE MaybeOwner(T* p, bool takeOwnership = true) : m_data(p), m_owner(takeOwnership) {}

            // This being HOST only might mean we are safe on the GPU side, but it would 
            // mean we are depending on the destructor of T to be HOST only as well
            HOSTDEVICE ~MaybeOwner() {
                #ifndef __CUDA_ARCH__
                if (m_owner && m_data)
                    delete m_data;
                #endif
            }

            HOSTDEVICE MaybeOwner(MaybeOwner const&) = delete;
            HOSTDEVICE MaybeOwner& operator=(MaybeOwner const&) = delete;

            HOSTDEVICE MaybeOwner(MaybeOwner && other) : m_data(std::move(other.m_data)), m_owner(std::move(other.m_owner)) {
                other.m_owner = false;
            }

            HOSTDEVICE MaybeOwner& operator=(MaybeOwner && other) {
                if (*this == other)
                    return *this;

                m_data = std::move(other.m_data);
                m_owner = std::move(other.m_owner);
                other.m_owner = false;

                return *this;
            }

            HOSTDEVICE T* get() const {
                return m_data;
            }

            HOSTDEVICE reference operator[](size_type index) {
                return m_data->operator[](index);
            }

            HOSTDEVICE const_reference operator[](size_type index) const {
                return m_data->operator[](index);
            }

            HOSTDEVICE T& operator*() {
                return *m_data;
            }

            HOSTDEVICE bool operator==(MaybeOwner const& other) { return other.m_data == m_data; }
            HOSTDEVICE bool operator!=(MaybeOwner const& other) { return !(*this == other); }

            HOSTDEVICE bool operator!() const { return !m_data; }
            HOSTDEVICE bool isOwned() const { return m_owner; }

            HOSTDEVICE T* operator->() { return get(); }
            HOSTDEVICE T* operator->() const { return get(); }

            HOSTDEVICE iterator begin() { return m_data->begin(); }
            HOSTDEVICE iterator end() { return m_data->end(); }

            HOSTDEVICE const_iterator begin() const { return m_data->begin(); }
            HOSTDEVICE const_iterator end() const { return m_data->end(); }

        private:
            T* m_data = nullptr;
            bool m_owner = true;
        };
}