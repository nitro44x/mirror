#pragma once

#include <mirror/simt_allocator.hpp>
#include <mirror/simt_vector.hpp>
#include <mirror/simt_utilities.hpp>

#include <vector>
#include <numeric>

namespace mirror {

        enum class Position {
            Beginning,
            End
        };

        class serializer final : public mirror::Managed {
        public:
            using buffer_t = mirror::vector<char>;
            using indices_t = mirror::vector<size_t>;

            using size_type = mirror::vector<char>::size_type;
            using index_size_type = mirror::vector<size_t>::size_type;

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
            HOST serializer() : m_data(new buffer_t), m_currentIndex(0), m_objIndices(new indices_t) {}

            HOST serializer(size_t startingBufferSize) : serializer() {
                m_data->reserve(startingBufferSize);
            }

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
            mirror::MaybeOwner<buffer_t> m_data;
            buffer_t::size_type m_currentIndex;
            mirror::MaybeOwner<indices_t> m_objIndices;
        };

        template <typename T>
        class Serializable {
        public:
            using type_id_t = T;

            HOSTDEVICE virtual ~Serializable() { ; }

            HOST virtual void write(mirror::serializer & io) const = 0;
            HOSTDEVICE virtual void read(mirror::serializer::size_type startPosition,
                                         mirror::serializer & io) = 0;
            HOSTDEVICE virtual type_id_t type() const = 0;
        };

        template <typename BaseClass>
        struct polymorphic_traits {
            /*
            This trait must be specialized for the base class that you intend on mirroring.
            See polymorphic_mirror_tests.cpp for an example.

            using size_type = std::size_t;
            using pointer = BaseClass*;
            using type = BaseClass;          // Top Level object (that inherites from mirror::Serializable
            using enum_type = BaseClassType; // Enum for each class in the object tree.
            static HOST size_type sizeOf(pointer p) { return 0; }
            static HOSTDEVICE void create(mirror::vector<BaseClass*> & device_objs, mirror::serializer & io) {}
            */

            static_assert(force_specialization<BaseClass>::value, "Must provide polymorphic_traits");
        };

        template <typename T>
        __global__
            void constructDeviceObjs(mirror::vector<T*> & device_objs, mirror::serializer & io) {
            polymorphic_traits<T>::create(device_objs, io);
        }

        template <typename T>
        __global__
        void destructDeviceObjs(mirror::vector<T*> & device_objs) {
            auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            for (; tid < device_objs.size(); tid += blockDim.x * gridDim.x) {
                device_objs[tid]->~T();
            }
        }

        template <typename T>
        HOSTDEVICE void construct_obj(void* where) {
            new(where) T;
        }

        template <typename T, size_t startingBufferSize = 1024 * 1024>
        class polymorphic_mirror final {
        public:
            using pointer = T*;
            using const_pointer = const pointer;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;

            using array_value_type = pointer;
            using array_value_pointer = array_value_type * ;
            using array_type = mirror::vector<pointer>;
            using array_pointer = array_type * ;
            using array_reference = array_type &;
            using const_array_reference = const array_type &;
            using iterator = array_type::iterator;
            using const_iterator = array_type::const_iterator;
            using reverse_iterator = array_type::reverse_iterator;
            using const_reverse_iterator = array_type::const_reverse_iterator;

            static_assert(std::is_base_of<Serializable<polymorphic_traits<T>::enum_type>, T>::value, "Object must inherit from mirror::Serializable");

        public:
            HOST polymorphic_mirror(std::vector<pointer> const& host_objs) : m_deviceObjs(new mirror::vector<pointer>(host_objs.size(), nullptr)) {
                mirror::MaybeOwner<serializer> io(new serializer(startingBufferSize));

                for (auto obj : host_objs) {
                    io->mark();
                    io->write(obj->type());
                    obj->write(*io);
                }

                auto sizeofFold = [](size_t currentTotal, pointer p) {
                    return currentTotal + polymorphic_traits<T>::sizeOf(p);
                };

                auto totalSpaceNeeded_bytes = std::accumulate(host_objs.begin(), host_objs.end(), size_t(0), sizeofFold);

                m_tank.setData(new tank_type(totalSpaceNeeded_bytes, '\0'));

                size_t offset = 0;
                for (size_t i = 0; i < host_objs.size(); ++i) {
                    (*m_deviceObjs)[i] = (pointer)(m_tank->data() + offset);
                    offset += polymorphic_traits<T>::sizeOf(host_objs[i]);
                }
                    
                constructDeviceObjs<<<nBlocks, nThreadsPerBlock>>>(*m_deviceObjs, *io);
                simt_sync;
            }

            HOST ~polymorphic_mirror() {
                destructDeviceObjs<<<nBlocks, nThreadsPerBlock>>>(*m_deviceObjs);
                simt_sync;
            }

            HOSTDEVICE array_pointer get() const {
                return m_deviceObjs.get();
            }

            HOSTDEVICE array_reference operator*() {
                return *m_deviceObjs;
            }

            HOSTDEVICE array_reference operator[](size_type index) {
                return m_deviceObjs->operator[](index);
            }

            HOSTDEVICE const_array_reference operator[](size_type index) const {
                return m_deviceObjs->operator[](index);
            }

            //HOSTDEVICE bool operator==(polymorphic_mirror const& other) { return other.m_tank == m_tank; }
            //HOSTDEVICE bool operator!=(polymorphic_mirror const& other) { return !(*this == other); }

            HOSTDEVICE bool operator!() const { return !m_tank; }

            HOSTDEVICE array_type* operator->() { return &m_deviceObjs; }
            HOSTDEVICE array_type* operator->() const { return &m_deviceObjs; }

            HOSTDEVICE iterator begin() { return m_deviceObjs->begin(); }
            HOSTDEVICE iterator end() { return m_deviceObjs->end(); }

            HOSTDEVICE const_iterator begin() const { return m_deviceObjs->begin(); }
            HOSTDEVICE const_iterator end() const { return m_deviceObjs->end(); }

        private:
            constexpr static size_t nBlocks = 128;
            constexpr static size_t nThreadsPerBlock = 128;

            using tank_type = mirror::vector<char, mirror::device_allocator<char>, mirror::OverloadNewType::eHostOnly>;
            mirror::MaybeOwner<tank_type> m_tank;

            mirror::MaybeOwner<array_type> m_deviceObjs;
        };
}