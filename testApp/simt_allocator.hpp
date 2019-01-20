#pragma once

#include <cuda_runtime.h>

#include "simt_macros.hpp"

namespace simt {

	namespace memory {
		/*
			https://stackoverflow.com/a/53033942
			Broadly speaking, an Allocator type is used when an object of one type (typically a container) needs to
			manage memory to hold an object or objects of some other type. Overloading operator new and operator delete
			within a class is used when objects of that type need some special memory management.
		*/

		template <class T>
		struct managed_allocator {
			typedef std::size_t size_type;
			typedef std::ptrdiff_t difference_type;

			using value_type = T;
			using pointer = T * ;
			using const_pointer = const T*;
			using reference = T & ;
			using const_reference = const T&;

			template< class U > struct rebind { typedef managed_allocator<U> other; };
			managed_allocator() = default;

			template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}

			HOST T* allocate(std::size_t n) {
				void* out = nullptr;
				simt_check(cudaMallocManaged(&out, n * sizeof(T)));
				memset(out, 0, n * sizeof(T));
				return static_cast<T*>(out);
			}

			HOST void deallocate(T* p, std::size_t) noexcept {
				simt_check(cudaFree(p));
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

			HOST T* allocate(std::size_t n) {
				void* out = nullptr;
				simt_check(cudaMalloc(&out, n * sizeof(T)));
				simt_check(cudaMemset(out, 0, n * sizeof(T)));
				return static_cast<T*>(out);
			}

			HOST void deallocate(T* p, std::size_t) noexcept {
				simt_check(cudaFree(p));
			}

		};

		class Managed {
		public:
			void *operator new(size_t len) {
				void *ptr;
				simt_check(cudaMallocManaged(&ptr, len));
				simt_sync;
				return ptr;
			}

			void operator delete(void *ptr) {
				simt_sync;
				simt_check(cudaFree(ptr));
			}
		};

		class DeviceOnly {
		public:
			void *operator new(size_t len) {
				void *ptr;
				simt_check(cudaMalloc(&ptr, len));
				simt_sync;
				return ptr;
			}

			void operator delete(void *ptr) {
				simt_sync;
				simt_check(cudaFree(ptr));
			}
		};

		// Provided for symmetry
		class HostOnly { };

		enum class OverloadNewType {
			eManaged = 0,
			eDeviceOnly,
			eHostOnly
		};

		template<OverloadNewType T> struct force_specialization : public std::false_type {};

		template <OverloadNewType T>
		struct Overload_trait_t {
			using type = HostOnly;
			static_assert(force_specialization<T>::value, "Must choose how to overload new/delete");
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

	}
}