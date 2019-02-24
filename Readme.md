# Mirror

This is a sandbox project to mirror classes with virtual methods from a host to a CUDA device. I wanted to let the rubber meet the road, so to speak, with my understanding of memory management on CUDA devices. My interest originated when I wanted to run a kernel for each element of a std::vector<BaseClass*> object. Since classes with virtual methods must be constructed on the device side (i.e. cannot be memcpy'ed), more framework is required. 

So far, the project provides:

1. Custom allocators for host, device, and managed (unified) memory types
2. Serialization concepts (via inheritance for now) to mirror objects on the host to the device
3. Vector types that use the custom allocators
4. Smart pointer like objects (see MaybeOwn) that look and feel like pointers, but allow you to control when allocation/deallocation occurs (so you can guarantee it happens on the host)
5. polymorphic_mirror type that allocates a memory pool for the base class pointers, and automatically constructs the polymorphic classes device side.

Full disclosure: It didn't work the way I wanted it too, but it looks like it’s something I can build off.
