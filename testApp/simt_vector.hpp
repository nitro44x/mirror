#pragma once

#include "simt_macros.hpp"
#include "simt_allocator.hpp"

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

		private:
			pointer p = nullptr;
		};

		template <typename T, class Alloc = simt::memory::managed_allocator<T>>
		class vector : public simt::memory::Managed {
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

			HOSTDEVICE iterator begin() { return iterator(m_data, 0); }
			HOSTDEVICE iterator end() { return iterator(m_data, m_size); }

			HOSTDEVICE const_iterator begin() const { return iterator(m_data, 0); }
			HOSTDEVICE const_iterator end() const { return iterator(m_data, m_size); }

		private:

			// \todo Use geometric growth strat: see std::vector::_Calculate_growth()
			HOST void grow() {
				auto const newCapacity = capacity() == 0 ? 1 : capacity() * 2;
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
}