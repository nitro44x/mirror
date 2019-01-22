#pragma once

#include "simt_allocator.hpp"
#include "simt_vector.hpp"

namespace simt {
    namespace seralization {


        enum class Position {
            Beginning,
            End
        };

        class serializer {

            using buffer_t = simt::containers::vector<char>;
            using indices_t = simt::containers::vector<size_t>;

        public:
            HOST serializer() : m_data(new buffer_t), m_currentIndex(0), m_objIndices(new indices_t) {}

            template <typename T>
            HOST void write(T const value) {
                char arr[sizeof(T)];
                memcpy(arr, &value, sizeof(T));
                for (size_t i = 0; i < sizeof(T); ++i)
                    m_data->push_back(arr[i]);
                m_currentIndex += sizeof(T);
            }

            template <typename T>
            HOSTDEVICE void read(T* value) {
                char arr[sizeof(T)];
                for (size_t i = 0; i < sizeof(T); ++i)
                    arr[i] = m_data[m_currentIndex + i];
                m_currentIndex += sizeof(T);
                memcpy(value, arr, sizeof(T));
            }

            template <typename T>
            HOSTDEVICE void read(buffer_t::size_type & startingPosition, T* value) {
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

            HOST buffer_t::size_type mark() {
                m_objIndices->push_back(m_currentIndex);
                return m_currentIndex;
            }

            HOSTDEVICE buffer_t::size_type mark_position(indices_t::size_type mark) const {
                return m_objIndices[mark];
            }

        private:
            simt::memory::MaybeOwner<buffer_t> m_data;
            buffer_t::size_type m_currentIndex;
            simt::memory::MaybeOwner<indices_t> m_objIndices;
        };

    }
}