#pragma once

#include "simt_allocator.hpp"
#include "simt_vector.hpp"

namespace simt {
    namespace seralization {

        template <typename T>
        __global__ void compute_sizeof(size_t * size) {
            auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid == 0)
                *size = sizeof(T);
        }

        template <typename T>
        HOST size_t determineSizeOf() {
            size_t * tmp = nullptr;
            cudaMallocManaged((void**)&tmp, sizeof(size_t));
            compute_sizeof<T><<<1, 1>>>(tmp);
            simt_sync;
            auto const deviceSizeOf = *tmp;
            cudaFree(size);
            auto const hostSizeOf = sizeof(T);
            return hostSizeOf > deviceSizeOf ? hostSizeOf : deviceSizeOf;
        }

        enum class Position {
            Beginning,
            End
        };

        class serializer final : public simt::memory::Managed {
        public:
            using buffer_t = simt::containers::vector<char>;
            using indices_t = simt::containers::vector<size_t>;

            using size_type = simt::containers::vector<char>::size_type;
            using index_size_type = simt::containers::vector<size_t>::size_type;

            using value_type = serializer;
            using pointer = serializer*;
            using const_pointer = const pointer;
            using reference = serializer&;
            using const_reference = const serializer&;
            //using size_type = T::size_type;
            using difference_type = std::ptrdiff_t;
            using iterator = void;
            using const_iterator = const iterator;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        public:
            HOST serializer() : m_data(new buffer_t), m_currentIndex(0), m_objIndices(new indices_t) {
                m_data->reserve(1024);
            }

            // todo: get max device/host sizeof and push [size|data w/ max size]
            template <typename T>
            HOST void write(T const value) {
                char arr[sizeof(T)];
                memcpy(arr, &value, sizeof(T));
                for (size_t i = 0; i < sizeof(T); ++i)
                    m_data->push_back(arr[i]);
                m_currentIndex += sizeof(T);
            }

            // todo: readin [size | data ]
            template <typename T>
            HOSTDEVICE void read(T* value) {
                char arr[sizeof(T)];
                for (size_t i = 0; i < sizeof(T); ++i)
                    arr[i] = m_data[m_currentIndex + i];
                m_currentIndex += sizeof(T);
                memcpy(value, arr, sizeof(T));
            }

            // todo: readin [size | data ]
            template <typename T>
            HOSTDEVICE void read(size_type & startingPosition, T* value) const {
                char arr[sizeof(T)];
                for (size_t i = 0; i < sizeof(T); ++i)
                    arr[i] = m_data[startingPosition + i];
                memcpy(value, arr, sizeof(T));
                startingPosition += sizeof(T);
            }

            HOSTDEVICE void seek(Position p) {
                switch (p) {
                case Position::Beginning:
                    m_currentIndex = 0;
                    return;
                case Position::End:
                    m_currentIndex = m_data->size();
                    return;
                }
            }

            HOST size_type mark() {
                m_objIndices->push_back(m_currentIndex);
                return m_currentIndex;
            }

            HOSTDEVICE size_type mark_position(indices_t::size_type mark) const {
                return m_objIndices[mark];
            }

            HOSTDEVICE size_type number_of_marks() const {
                return m_objIndices->size();
            }

        private:
            simt::memory::MaybeOwner<buffer_t> m_data;
            buffer_t::size_type m_currentIndex;
            simt::memory::MaybeOwner<indices_t> m_objIndices;
        };

        template <typename T>
        class Serializable {
        public:
            using type_id_t = T;

            HOSTDEVICE virtual ~Serializable() { ; }

            HOST virtual void write(simt::seralization::serializer & io) const = 0;
            HOSTDEVICE virtual void read(simt::seralization::serializer::size_type startPosition,
                                         simt::seralization::serializer & io) = 0;
            HOSTDEVICE virtual type_id_t type() const = 0;
        };

    }
}