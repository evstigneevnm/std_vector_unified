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
 
template <class T>
struct some_data
{   
    using int_t = std::pair<std::size_t, T>;

    some_data() = default;
    some_data(T a_, T b_, int_t g_):
    a(a_), b(b_), g(g_) {}

    T a;
    T b;
    std::pair<std::size_t, T> g;
};

int main(int argc, char const *argv[]) 
{
    using type = double;
    using data_t = some_data<type>;
    using allocator = scfd_simple::unified_allocator<data_t>;
    std::vector<data_t, allocator> v1;

    for(int j = 0; j<1000;j++)
    {
        v1.push_back( {-static_cast<type>(j),static_cast<type>(j), {static_cast<type>(j),1.0/static_cast<type>(j)} } );
    }

    std::cout << std::endl;
    for(auto& x: v1)
    {
        std::cout << "a = " << x.a << ", b = " << x.b << ", g = (" << x.g.first << "," << x.g.second << ")" << std::endl;
    }
    std::vector<data_t> v2( v1.size() );
    std::copy(v1.begin(), v1.end(), v2.begin());
    // std::vector<data_t> v3 = v1; // will not work because of different allocators! call std::copy if you need to make a deep copy.

    return 0;
}