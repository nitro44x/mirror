#pragma once

#include <iostream>

#include <cuda_runtime.h>


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