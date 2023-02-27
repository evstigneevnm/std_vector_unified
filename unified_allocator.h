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


#ifndef __SCFD_UNIFIED_ALLOCATOR_H__cffee004ccc7b218a0f27a28379ce6b41c2df464fde007ba59de2f02b43ac505
#define __SCFD_UNIFIED_ALLOCATOR_H__cffee004ccc7b218a0f27a28379ce6b41c2df464fde007ba59de2f02b43ac505

#include <limits>
#ifdef __CUDACC__
    #include "cuda_safe_call.h"
#else
    #include <new>
    #include <cstdlib>
#endif


namespace scfd_simple
{

#ifdef __CUDACC__
    template<class T, bool Universal = true>
    class cuda_universal_vector
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        cuda_universal_vector() noexcept = default;
        cuda_universal_vector(std::size_t size):
        size_(size)
        {
            if constexpr (Universal)
            {
                CUDA_SAFE_CALL( cudaMallocManaged((void**)&data_, sizeof(T)*size_) );
                #ifdef __SCFD_DEBUG_LEVEL_2
                    std::cout << "cuda_universal_vector::managed(" << size_ << ");" << std::endl;
                #endif
            }
            else
            {
                CUDA_SAFE_CALL( cudaMalloc((void**)&data_, sizeof(T)*size_) );
                #ifdef __SCFD_DEBUG_LEVEL_2
                    std::cout << "cuda_universal_vector::device(" << size_ << ");" << std::endl;                
                #endif
            }
        }
        ~cuda_universal_vector() noexcept
        {
            if(data_ != nullptr)
            {
                cudaFree(data_);
                #ifdef __SCFD_DEBUG_LEVEL_2
                    std::cout << "~cuda_universal_vector" << std::endl;
                #endif
            }
        }
        void init(std::size_t size)
        {
            if(size_ == 0)
            {
                size_ = size;
                if constexpr (Universal)
                {                
                    CUDA_SAFE_CALL( cudaMallocManaged((void**)&data_, sizeof(T)*size_) );
                    #ifdef __SCFD_DEBUG_LEVEL_2
                        std::cout << "cuda_universal_vector::init_managed(" << size_ << ");" << std::endl;
                    #endif
                }
                else
                {
                    CUDA_SAFE_CALL( cudaMalloc((void**)&data_, sizeof(T)*size_) );
                    #ifdef __SCFD_DEBUG_LEVEL_2
                        std::cout << "cuda_universal_vector::init_device(" << size_ << ");" << std::endl;                    
                    #endif
                }

            }
        }
        [[nodiscard]] constexpr T* data() 
        {
            return data_;
        }
        [[nodiscard]] constexpr size_type size() const noexcept
        {
            return size_;
        }

        void copy_to_this_vector(const T* other) const
        {
            CUDA_SAFE_CALL( cudaMemcpy(data_, other, sizeof(T)*size_, cudaMemcpyHostToDevice ) );
        }
        void copy_from_this_vector(T* other) const
        {
            CUDA_SAFE_CALL( cudaMemcpy(other, data_, sizeof(T)*size_, cudaMemcpyDeviceToHost ) );
        }

    private:
        T* data_ = nullptr;
        std::size_t size_ = 0;

    };
#endif

template<class T>
struct unified_allocator
{
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    // depricated in c++17:
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    unified_allocator() noexcept = default;
    unified_allocator(const unified_allocator&) noexcept = default;
    template <class U> unified_allocator(const unified_allocator<U>&) noexcept {}    
    ~unified_allocator() = default;


    [[nodiscard]] constexpr T* allocate( std::size_t n )
    {
        if(std::numeric_limits<std::size_t>::max() / sizeof(T) < n) throw std::bad_array_new_length();
        T* ptr = nullptr;
        #ifdef __CUDACC__
            CUDA_SAFE_CALL( cudaMallocManaged((void**)&ptr, n*sizeof(T)) );
            #ifdef __SCFD_DEBUG_LEVEL_2
                std::cout << "cuda alloc size = " << n << std::endl;
            #endif
        #else
            ptr = reinterpret_cast<T*>(std::malloc(n * sizeof(T) ));
            if(ptr == nullptr) throw std::bad_alloc(); 
            #ifdef __SCFD_DEBUG_LEVEL_2
                std::cout << "host alloc size = " << n << std::endl;
            #endif
        #endif
        return ptr;
    }
    constexpr void deallocate( T* p, std::size_t n ) noexcept // n?
    {
        #ifdef __CUDACC__
            cudaFree(p);
            #ifdef __SCFD_DEBUG_LEVEL_2
                std::cout << "cuda dealloc size = " << n << std::endl;
            #endif
        #else
            std::free(p);
            #ifdef __SCFD_DEBUG_LEVEL_2
                std::cout << "host dealloc size = " << n << std::endl;
            #endif
        #endif 
    }


};
template<class T1, class T2>
bool operator==(const unified_allocator <T1>&, const unified_allocator <T2>&) noexcept
{ 
    return true; 
}
template<class T1, class T2>
bool operator!=(const unified_allocator <T1>&, const unified_allocator <T2>&) noexcept
{ 
    return false; 
}

}


#endif // __UNIFIED_ALLOCATOR_H__cffee004ccc7b218a0f27a28379ce6b41c2df464fde007ba59de2f02b43ac505