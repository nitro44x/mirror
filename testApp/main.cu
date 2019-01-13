#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
#define HOST __host__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define HOST
#define DEVICE
#endif

namespace simt {

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

	namespace containers {

		template <typename T, class Alloc = simt::managed_allocator<T>>
		class vector : public Managed {
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
			vector() = default;
			vector(size_type nElements) : m_alloc(), m_data(m_alloc.allocate(nElements)), m_size(nElements), m_capacity(nElements) {}

			vector(size_type nElements, value_type initValue) : vector(nElements) {
				for (size_type i = 0; i < nElements; ++i)
					m_data[i] = initValue;
			}

			~vector() {
				m_alloc.deallocate(m_data, m_size);
			}

			HOST void resize(size_type nElements) {
				if (nElements > capacity()) {
					pointer tmp = m_alloc.allocate(nElements);
					memcpy(tmp, m_data, sizeof(T) * size());
					m_data = tmp;
					m_capacity = nElements;
				}
				
				m_size = nElements;
			}

			HOST void push_back(value_type value) {
				auto const currentSize = size();
				if (currentSize <= capacity())
					grow();
				this->operator[](currentSize) = value;
				++m_size;
			}

			HOSTDEVICE pointer data() const { return m_data; }
			HOSTDEVICE size_type size() const { return m_size; }
			HOSTDEVICE size_type capacity() const { return m_capacity; }

			HOSTDEVICE reference operator[](size_type index) { return *(m_data + index); }
			HOSTDEVICE const_reference operator[](size_type index) const { return *(m_data + index); }

			HOSTDEVICE iterator begin() { return iterator(*this, 0); }
			HOSTDEVICE iterator end() { return iterator(*this, m_size); }

		private:
			void grow() {
				auto const newCapacity = capacity() * 2;
				pointer tmp = m_alloc.allocate(newCapacity);
				memcpy(tmp, m_data, sizeof(T) * size());
				m_data = tmp;
				m_capacity = newCapacity;
			}

			allocator_type m_alloc{};
			pointer m_data = nullptr;
			size_type m_size = 0;
			size_type m_capacity = 0;
		};
	}

	namespace management {
		
		class Store {
		public:

		private:


		};
	}
}

using managed_vector = simt::containers::vector<double, simt::managed_allocator<double>>;

template <typename T>
__global__ void printArray(T* data, size_t size) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("gpu v = ");
		for (int i = 0; i < size; ++i)
			printf("%lf ", data[i]);
		printf("\n");
	}
}

HOSTDEVICE void printVector(managed_vector * v) {
	printf("gpu v = ");
	for (auto const& d : *v)
		printf("%lf ", d);
	printf("\n");
}

__global__ void call_printVector(managed_vector * v) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printVector(v);
	}
}

__global__ void call_printVector_ref(managed_vector & v) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printVector(&v);
	}
}


HOSTDEVICE void setTo(managed_vector * v, managed_vector::value_type value) {
	for (auto & d : *v)
		d = value;
}

__global__ void call_setTo(managed_vector * v, managed_vector::value_type value) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		setTo(v, value);
	}
}

void test1() {
	std::cout << "std::vector" << std::endl;
	std::vector<double, simt::managed_allocator<double>> v(10);
	std::iota(begin(v), end(v), -4);
	std::cout << "cpu v = ";
	for (auto const& d : v)
		std::cout << d << " ";
	std::cout << std::endl;

	printArray<<<1, 1>>>(v.data(), v.size());
	cudaDeviceSynchronize();
	std::cout << std::endl;
}

void test2() {
	std::cout << "simt::containers::vector [raw ptr]" << std::endl;
	simt::containers::vector<double, simt::managed_allocator<double>> simt_v(10, 3.0);
	std::iota(simt_v.begin(), simt_v.end(), -3.0);
	simt_v.push_back(4321);
	std::cout << "cpu v = ";
	for (auto const& d : simt_v)
		std::cout << d << " ";
	std::cout << std::endl;
	printArray<<<1,1>>>(simt_v.data(), simt_v.size());
	cudaDeviceSynchronize();
	std::cout << std::endl;
}

void test3() {
	std::cout << "simt::containers::vector [object]" << std::endl;
	auto simt_v_ptr = new managed_vector(10);
	std::iota(simt_v_ptr->begin(), simt_v_ptr->end(), -4);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	printVector(simt_v_ptr);
	call_printVector<<<1,1>>>(simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test3a() {
	std::cout << "simt::containers::vector [object] printByRef" << std::endl;
	auto simt_v_ptr = new managed_vector(10);
	std::iota(simt_v_ptr->begin(), simt_v_ptr->end(), -4);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	printVector(simt_v_ptr);
	call_printVector_ref<<<1,1>>>(*simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test4() {
	std::cout << "modify simt::containers::vector [object] on cpu" << std::endl;
	auto simt_v_ptr = new managed_vector;
	simt_v_ptr->resize(10);
	setTo(simt_v_ptr, 123);
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	call_printVector<<<1,1>>>(simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

void test5() {
	std::cout << "modify simt::containers::vector [object] on gpu" << std::endl;
	auto simt_v_ptr = new managed_vector;
	simt_v_ptr->resize(10);
	call_setTo<<<1,1>>>(simt_v_ptr, 123);
	cudaDeviceSynchronize();
	std::cout << "cpu v = ";
	for (auto const& d : *simt_v_ptr)
		std::cout << d << " ";
	std::cout << std::endl;
	call_printVector<<<1,1>>>(simt_v_ptr);
	cudaDeviceSynchronize();
	delete simt_v_ptr;
	std::cout << std::endl;
}

int main() {
	// Print vector tests
	test1();
	test2();
	test3();
	test3a();

	// Modify vector tests
	test4();
	test5();

    return EXIT_SUCCESS;
}