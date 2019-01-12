#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>

#include <vector>
#include <numeric>


#define check(ans) { assert_((ans), __FILE__, __LINE__); }
void assert_(cudaError_t code, const char *file, int line) {
	if (code == cudaSuccess) return;
	std::cerr << "check failed: " << cudaGetErrorString(code) << " : " << file << '@' << line << std::endl;
	abort();
}

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

namespace simt {

	template <class T>
	struct managed_allocator {
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;
													   
		typedef T value_type;
		typedef T* pointer;// (deprecated in C++17)(removed in C++20)	T*
		typedef const T* const_pointer;// (deprecated in C++17)(removed in C++20)	const T*
		typedef T& reference;// (deprecated in C++17)(removed in C++20)	T&
		typedef const T& const_reference;// (deprecated in C++17)(removed in C++20)	const T&
																			   
		template< class U > struct rebind { typedef managed_allocator<U> other; };
		managed_allocator() = default;

		template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}

		T* allocate(std::size_t n) {
			void* out = nullptr;
			check(cudaMallocManaged(&out, n * sizeof(T)));
			return static_cast<T*>(out);
		}

		void deallocate(T* p, std::size_t) noexcept {
			check(cudaFree(p));
		}

	};

	namespace containers {

		template <typename T, class Alloc = simt::managed_allocator<T>>
		class vector {
		public:

			class vectorIter;

			using value_type = T;
			using allocator_type = Alloc;
			using pointer = T * ;
			using const_pointer = const T*;
			using reference = T & ;
			using const_reference = const T&;
			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;
			using iterator = vectorIter;
			using const_iterator = const vectorIter;
			using reverse_iterator = std::reverse_iterator<iterator>;
			using const_reverse_iterator = std::reverse_iterator<const_iterator>;


			class vectorIter {
			public:
				HOSTDEVICE vectorIter(vector<T> & v, size_type o) : m_vector(v), m_offset(o) {}

				HOSTDEVICE bool operator==(vectorIter const& other) { return other.m_vector.data() == m_vector.data() && other.m_offset == m_offset; }
				HOSTDEVICE bool operator!=(vectorIter const& other) { return !(*this == other); }
				HOSTDEVICE reference operator*() { return m_vector[m_offset]; }
				HOSTDEVICE vectorIter & operator++() { ++m_offset; return *this; }
				HOSTDEVICE vectorIter operator++(int) {
					vectorIter clone(*this);
					++m_offset;
					return clone;
				}

			private:
				vector<T, Alloc> & m_vector;
				size_type m_offset;
			};

		public:
			vector(size_type nElements) : m_alloc(), m_data(m_alloc.allocate(nElements)), m_size(nElements), m_capacity(nElements) {}

			vector(size_type nElements, value_type initValue) : vector(nElements) {
				for (size_type i = 0; i < nElements; ++i)
					m_data[i] = initValue;
			}

			~vector() {
				m_alloc.deallocate(m_data, m_size);
			}

			void resize(size_type nElements) {
				pointer tmp = m_alloc.allocate(nElements);
				memcpy(tmp, m_data, nElements > m_size ? m_size : nElements);
				m_data = tmp;
				m_size = nElements;
				m_capacity = nElements;
			}

			HOSTDEVICE pointer data() const { return m_data; }
			HOSTDEVICE size_type size() const { return m_size; }
			HOSTDEVICE size_type capacity() const { return m_capacity; }

			HOSTDEVICE reference operator[](size_type index) { return *(m_data + index); }

			HOSTDEVICE iterator begin() { return iterator(*this, 0); }
			HOSTDEVICE iterator end() { return iterator(*this, m_size); }

		private:
			allocator_type m_alloc;
			pointer m_data;
			size_type m_size;
			size_type m_capacity;

			//friend class iterator;
		};
	}
}


template <typename T>
__global__ void printArray(T* data, size_t size) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("gpu v = ");
		for (int i = 0; i < size; ++i)
			printf("%lf ", data[i]);
		printf("\n");
	}
}

template <typename VectorContainer>
__host__ __device__ void printVector(VectorContainer * v) {
	printf("gpu v = ");
	for (auto const& d : *v)
		printf("%lf ", d);
	printf("\n");
}

template <typename VectorContainer>
__global__ void call_printVector(VectorContainer * v) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printVector(v);
	}
}

//template <typename T>
//__global__ void muckWithVector(simt::containers::vector)

int main() {
	std::cout << "std::vector" << std::endl;
	std::vector<double, simt::managed_allocator<double>> v(10);
	std::iota(begin(v), end(v), -4);
	std::cout << "cpu v = ";
	for (auto const& d : v)
		std::cout << d << " ";
	std::cout << std::endl;

	printArray<<<1,1>>>(v.data(), v.size());
	cudaDeviceSynchronize();
	std::cout << std::endl;

	std::cout << "simt::containers::vector [raw ptr]" << std::endl;
	simt::containers::vector<double, simt::managed_allocator<double>> simt_v(10, 3.0);
	std::iota(simt_v.begin(), simt_v.end(), -3.0);
	std::cout << "cpu v = ";
	for (auto const& d : simt_v)
		std::cout << d << " ";
	std::cout << std::endl;
	printArray<<<1,1>>>(simt_v.data(), simt_v.size());
	cudaDeviceSynchronize();
	std::cout << std::endl;
	
	std::cout << "simt::containers::vector [object]" << std::endl;
	typedef simt::containers::vector<double, simt::managed_allocator<double>> managed_vector;
	simt::managed_allocator<managed_vector> alloc;
	auto simt_v_ptr = alloc.allocate(1);
	simt_v_ptr->resize(10);
	std::iota(simt_v_ptr->begin(), simt_v_ptr->end(), -4);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	printVector(simt_v_ptr);
	call_printVector<<<1,1>>>(simt_v_ptr);
	cudaDeviceSynchronize();
	simt_v_ptr->~vector();
	alloc.deallocate(simt_v_ptr, 0);
	std::cout << std::endl;

	
    
    return EXIT_SUCCESS;
}