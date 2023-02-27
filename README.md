# std_vector_unified
This is a test repository for using ``std::vector&lt;T, unified_allocator&lt;T>>``, where unified_allocator is using cuda unified memory. It also uncludes a simple ``cuda_universal_vector<T, bool>`` that wraps cuda raw pointers with ``<T, false>`` is called when using device memory and ``<T, true>`` is called when using unified memory.
To use include ``unified_allocator.h`` in your files.
to make some tests:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make all
```
