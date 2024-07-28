#ifndef THATSZUCS_PLAYGROUND_PYARRAY_UTILS_H
#define THATSZUCS_PLAYGROUND_PYARRAY_UTILS_H

#include "playground/pinned_vector.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace playground {

template <typename T>
py::array_t<T> as_numpy(const std::vector<T> &myvec)
{
    return py::array_t<T>({myvec.size()}, // Shape
                          {sizeof(T)},    // Stride
                          myvec.data(),   // Data pointer
                          py::none());
}

template <typename T>
py::array_t<T> as_numpy(const std::vector<T, pinned_alloc<T>> &myvec)
{
    return py::array_t<T>({myvec.size()}, // Shape
                          {sizeof(T)},    // Stride
                          myvec.data(),   // Data pointer
                          py::none());
}

} // namespace playground

#endif

/*
 *  This code is part of the playground project.
 *  Copyright (c) 2024 ThatSzucs
 *
 *  Distributed under the MIT license. See accompanying license file
 *  or copy at https://opensource.org/licenses/MIT.
 */
