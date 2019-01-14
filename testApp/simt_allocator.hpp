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
				//static_assert(!std::is_trivially_copyable<T>::value, "Managed Allocator can only be used for trivially copiable types");

				void* out = nullptr;
				check(cudaMallocManaged(&out, n * sizeof(T)));
				memset(out, 0, n * sizeof(T));
				return static_cast<T*>(out);
			}

			HOST void deallocate(T* p, std::size_t) noexcept {
				check(cudaFree(p));
			}

		};

		class Managed {
		public:
			void *operator new(size_t len) {
				void *ptr;
				cudaMallocManaged(&ptr, len);
				cudaDeviceSynchronize();
				return ptr;
			}

			void operator delete(void *ptr) {
				cudaDeviceSynchronize();
				cudaFree(ptr);
			}
		};
	}
}