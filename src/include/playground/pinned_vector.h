#ifndef THATSZUCS_PLAYGROUND_PINNED_VECTOR_H
#define THATSZUCS_PLAYGROUND_PINNED_VECTOR_H

#include "cuda_runtime.h" // "cuda_runtime_api.h" is C header

#include <stdexcept>
#include <vector>

namespace playground {

/* The allocator class */
template <typename T>
class pinned_alloc {
  public:
    using value_type = T;
    using pointer = value_type *;
    using size_type = std::size_t;

    pinned_alloc() noexcept = default;

    template <typename U>
    pinned_alloc(pinned_alloc<U> const &) noexcept
    {
    }

    auto allocate(size_type n, const void * = 0) -> value_type *
    {
        value_type *tmp;
        auto error = cudaMallocHost((void **)&tmp, n * sizeof(T));
        if (error != cudaSuccess) {
            throw std::runtime_error{cudaGetErrorString(error)};
        }
        return tmp;
    }

    auto deallocate(pointer p, size_type n) -> void
    {
        if (p) {
            auto error = cudaFreeHost(p);
            if (error != cudaSuccess) {
                throw std::runtime_error{cudaGetErrorString(error)};
            }
        }
    }
};

/* Equality operators */
template <class T, class U>
auto operator==(pinned_alloc<T> const &, pinned_alloc<U> const &) -> bool
{
    return true;
}

template <class T, class U>
auto operator!=(pinned_alloc<T> const &, pinned_alloc<U> const &) -> bool
{
    return false;
}

/* Template alias for convenient creating of a vector backed by pinned memory
 */
template <typename T>
using pinned_vector = std::vector<T, pinned_alloc<T>>;

} // namespace playground

#endif

/*
 *  This code is part of the playground project.
 *  Copyright (c) 2024 ThatSzucs
 *
 *  Distributed under the MIT license. See accompanying license file
 *  or copy at https://opensource.org/licenses/MIT.
 */
