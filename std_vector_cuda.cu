// Copyright © 2023 Evstigneev Nikolay Mikhaylovitch

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:

// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>
#include <vector>
#include <iostream>
#include "unified_allocator.h"
 
namespace detail
{
namespace kern
{

const unsigned int blocksize = 256;

template<class Data>
__global__ void process_data(std::size_t N, Data data_)
{
    std::size_t idx=blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N) return;
    data_[idx].a /= data_[idx].b;
    data_[idx].g.first *= data_[idx].g.second;

}


}
template<class Data>
void process_data(std::size_t N, Data data_)
{
    std::size_t nthreads = kern::blocksize;
    size_t k1 = ( N + nthreads -1 )/nthreads;
    dim3 dimGrid(k1, 1, 1);
    dim3 dimBlock(kern::blocksize, 1, 1);
    kern::process_data<Data><<<dimGrid, dimBlock>>>(N, data_);
}


}

template <class T>
struct some_data 
{   
    using int_t = std::pair<std::size_t, T>; //std::pair is a simple example of a data type that cuda kernels can support

    some_data() = default;
    some_data(T a_, T b_, int_t g_):
    a(a_), b(b_), g(g_) {}

    T a;
    T b;
    int_t g;
};

int main(int argc, char const *argv[]) 
{
    using type = double; //basic POD data type
    using data_t = some_data<type>; //some class that contains data that will be used in HOST and DEVICE operations
    using allocator_t = scfd_simple::unified_allocator<data_t>; // typedef unified allocator
    using data_pair_t = typename data_t::int_t;
    if(argc != 3)
    {
        std::cout << "usage: " << argv[0] << " N GPU_ID" << std::endl;
        std::cout << "  where N is the size of a cube," << std::endl;
        std::cout << "  GPU_ID is the GPU ID value to be used." << std::endl;
        return 0;
    }

    int device_id = std::stoi(argv[2]);
    cudaDeviceProp device_prop;
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&device_prop, device_id) );     
    std::cout << "using CUDA device number " << device_id << ": " << device_prop.name << std::endl;
    CUDA_SAFE_CALL( cudaSetDevice(device_id) );
    std::size_t N = std::stoi(argv[1]);

    //custom allocator with cuda unified memory
    std::vector<data_t, allocator_t> v;

    for(int j = 0; j<N/2;j++)
    {
        v.push_back( {-static_cast<type>(j),static_cast<type>(j), {j, 1.0/static_cast<type>(j)} } );
    }
    for(int j = N/2; j<N;j++)
    {
        v.emplace_back(-static_cast<type>(j), static_cast<type>(j), data_pair_t(j, 1.0/static_cast<type>(j)) );
    }

    std::cout << std::endl;
    auto& x = v[N/2];
    auto& y = v[N-1];
    std::cout << "a = " << x.a << ", b = " << x.b << ", g = (" << x.g.first << "," << x.g.second << ")" << std::endl;
    std::cout << "a = " << y.a << ", b = " << y.b << ", g = (" << y.g.first << "," << y.g.second << ")" << std::endl;
    std::cout << "executing a cuda kernel on data pointer in the std::vector<data_t, std::unified_alocator<data_t> >:" << std::endl;

    // WARNING! Cuda kernels will only use raw pointer to data, no other STL features will be supported. 
    // That is why one can only pass vector.data() to cuda kernels.
    detail::process_data(N, v.data() );
    // Always call cudaDeviceSynchronize() after the kernel is executed to force universal memory to be synced, unless you know what you are doing with cudaStreams.
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::cout << "a = " << x.a << ", b = " << x.b << ", g = (" << x.g.first << "," << x.g.second << ")" << std::endl;
    std::cout << "a = " << y.a << ", b = " << y.b << ", g = (" << y.g.first << "," << y.g.second << ")" << std::endl;

    //std::vector with a standard allocator    
    std::vector<data_t> v2( v.size() ); 
    // std::vector<data_t> v2 = v; // v2 = v will not work because of different allocators! call std::copy if you need to make a deep copy.
    std::copy(v.begin(), v.end(), v2.begin());
    

    auto& x1 = v2[N/2];
    auto& y1 = v2[N-1];
    std::cout << "copy to the std::vector<data_t, std::alocator<data_t> >:" << std::endl;
    std::cout << "a = " << x1.a << ", b = " << x1.b << ", g = (" << x1.g.first << "," << x1.g.second << ")" << std::endl;
    std::cout << "a = " << y1.a << ", b = " << y1.b << ", g = (" << y1.g.first << "," << y1.g.second << ")" << std::endl;


    // testing a simple unified vector utility
    using simple_cuda_device_vec_t = scfd_simple::cuda_universal_vector<data_t, false>; //simple universal vector type that uses device memory
    simple_cuda_device_vec_t vec(N); //constructor
    vec.copy_to_this_vector( v2.data() ); // v2->vec
    detail::process_data(vec.size(), vec.data() ); // execute cuda kernel, use vec.data() to access raw pointer.
    vec.copy_from_this_vector( v2.data() ); // vec->v2

    std::cout << "copy from the cuda_universal_vector<data_t, false>:" << std::endl;
    std::cout << "a = " << x1.a << ", b = " << x1.b << ", g = (" << x1.g.first << "," << x1.g.second << ")" << std::endl;
    std::cout << "a = " << y1.a << ", b = " << y1.b << ", g = (" << y1.g.first << "," << y1.g.second << ")" << std::endl;    

    using simple_cuda_unified_vec_t = scfd_simple::cuda_universal_vector<data_t, true>; //simple universal vector type that uses device memory
    simple_cuda_unified_vec_t vec_u(N); //constructor
    vec_u.copy_to_this_vector( v1.data() ); // v2->vec
    detail::process_data(vec_u.size(), vec_u.data() ); // execute cuda kernel, use vec.data() to access raw pointer.
    vec_u.copy_from_this_vector( v1.data() ); // vec->v2

    std::cout << "copy from the cuda_universal_vector<data_t, true>:" << std::endl;
    std::cout << "a = " << x.a << ", b = " << x.b << ", g = (" << x.g.first << "," << x.g.second << ")" << std::endl;
    std::cout << "a = " << x.a << ", b = " << y.b << ", g = (" << y.g.first << "," << y.g.second << ")" << std::endl;    

    return 0;
}