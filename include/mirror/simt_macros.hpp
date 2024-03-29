#pragma once

#include <iostream>

#include <driver_types.h>

#define mirror_check(ans) { assert_((ans), __FILE__, __LINE__); }
void assert_(cudaError_t code, const char *file, int line);

#define mirror_sync mirror_check(cudaDeviceSynchronize());

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define HOST
#define DEVICE
#endif

