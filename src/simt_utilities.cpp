#include <mirror/simt_utilities.hpp>
#include <mirror/simt_macros.hpp>

#include <cuda_runtime.h>

#include <iostream>


void assert_(cudaError_t code, const char *file, int line) {
    if (code == cudaSuccess) return;
    std::cerr << "check failed: " << cudaGetErrorString(code) << " : " << file << '@' << line << std::endl;
    abort();
}

namespace simt {
    namespace utilities {
        HOSTDEVICE int getTID() {
            #ifdef __CUDA_ARCH__
            return (int)(threadIdx.x + blockIdx.x * blockDim.x);
            #else
            return 0;
            #endif
        }

        HOSTDEVICE int gridStride() {
            #ifdef __CUDA_ARCH__
            return (int)(blockDim.x * gridDim.x);
            #else
            return 1;
            #endif
        }
    }
}