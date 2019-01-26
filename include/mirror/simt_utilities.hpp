#pragma once

#include <mirror/simt_macros.hpp>

#include <cuda_runtime_api.h>

namespace simt {
    namespace utilities {

        HOSTDEVICE int getTID();

        template <typename T>
        __global__ void compute_sizeof(size_t * size) {
            auto tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid == 0)
                *size = sizeof(T);
        }

        template <typename T>
        HOST size_t getDeviceSize() {
            size_t * size = nullptr;
            cudaMallocManaged((void**)&size, sizeof(size_t));
            compute_sizeof<T><<<1,1>>> (size);
            simt_sync;
            auto const result = *size;
            cudaFree(size);
            return result;
        }
    }
}
